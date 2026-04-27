"""
多模态故障检测系统主入口

三阶段训练流程：
1. CSD Transformer 预训练（对比学习）
2. VQ-VAE 工况预训练（离散状态编码）- v3.2 更新
   或 Ti-MAE 工况预训练（掩码重构）- 旧方案
3. 条件联合微调

v3.2 更新：新增 VQ-VAE 双通道工况编码器
v3.1 更新：将 SPDNet 替换为 CSD Pair-Token Transformer
"""

import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

# 导入模块
from data.preprocessing import VirtualChannelExpander, CSDMatrixBuilder
from data.dataset import MultiModalDataset, create_dataloaders, create_dataloaders_from_preprocessed
from data.augmentation import OrthogonalityPerturbation

# 模型
from models.full_model import (
    MultiModalAnomalyDetector,
    CSDPretrainModel,
    TiMAEPretrainModel,
    VQVAEPretrainModel,  # VQ-VAE 新方案
    SPDPretrainModel  # 保留以兼容旧权重
)
from models.csd_transformer import CSDTransformerEncoder
from models.timae.timae import TiMAEWithPhysicsLoss
from models.vqvae import DualChannelVQVAE  # VQ-VAE 新方案

# 训练器
from trainers import TiMAEPretrainer, JointTrainer, VQVAEPretrainer
from trainers.csd_pretrainer import CSDPretrainer

# 推理
from inference import AnomalyScorer, POTThreshold


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(data_path: str, config: dict, stage: str = 'joint'):
    """
    准备数据
    
    Args:
        data_path: 数据路径（支持 .pt, .npy, .npz 格式）
        config: 配置字典
        stage: 训练阶段 ('csd', 'timae', 'joint')，决定 batch_size
        
    Returns:
        train_loader, val_loader
    """
    # 根据阶段选择 batch_size
    stage_config_map = {
        'csd': 'csd_pretrain',
        'spd': 'csd_pretrain',  # 兼容旧参数
        'timae': 'timae_pretrain',
        'joint': 'joint'
    }
    stage_key = stage_config_map.get(stage, 'joint')
    
    # 优先从新配置读取，兼容旧配置
    if stage_key in config['training']:
        batch_size = config['training'][stage_key].get('batch_size', 32)
    elif 'spd_pretrain' in config['training'] and stage in ['csd', 'spd']:
        batch_size = config['training']['spd_pretrain'].get('batch_size', 32)
    else:
        batch_size = 32
    
    print(f"Stage: {stage}, Batch size: {batch_size}")
    
    # 根据文件格式选择加载方式
    if data_path.endswith('.pt'):
        # PyTorch 预处理格式（由 preprocess_dataset.py 生成）
        print(f"Loading preprocessed data from {data_path}")
        
        # 只有 CSD 预训练阶段需要对比学习的顺序采样
        use_sequential = (stage in ['csd', 'spd'])
        augmentation = None
        
        if use_sequential:
            # 创建正交性扰动增强器（用于生成对比学习的负样本）
            augmentation = OrthogonalityPerturbation(
                perturbation_scale=0.1,
                mode='pv_coupling',  # 仅扰动 P-V 耦合块
                preserve_hermitian=True
            )
        
        train_loader, val_loader = create_dataloaders_from_preprocessed(
            preprocessed_path=data_path,
            train_ratio=config['data'].get('train_ratio', 0.8),
            batch_size=batch_size,
            num_workers=config['training'].get('num_workers', 4),
            use_sequential=use_sequential,
            augmentation=augmentation
        )
    else:
        # NumPy 格式（原始数据）
        print(f"Loading raw data from {data_path}")
        data = np.load(data_path)
        
        if isinstance(data, np.lib.npyio.NpzFile):
            # 如果是 npz 文件
            data = data['data']
        
        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(
            data=data,
            train_ratio=config['data'].get('train_ratio', 0.8),
            window_size=config['data']['window_size'],
            stride=config['data']['stride'],
            batch_size=batch_size,
            precompute_csd=config['data'].get('precompute_csd', True),
            fs=config['data']['fs']
        )
    
    return train_loader, val_loader


def stage1_csd_pretrain(train_loader, val_loader, config, checkpoint_dir):
    """
    阶段一：CSD Transformer 预训练
    
    使用对比学习训练 CSD Pair-Token Transformer 编码器：
    - 正样本对：相邻时间窗口的 CSD 矩阵
    - 负样本对：正交性扰动后的 CSD 矩阵
    - 损失函数：InfoNCE（余弦相似度 + 温度参数）
    
    v3.1 更新：替代原有的 SPDNet 黎曼流形学习
    """
    print("\n" + "=" * 50)
    print("Stage 1: CSD Transformer Pre-training (Contrastive Learning)")
    print("=" * 50)
    
    # 获取 CSD Transformer 配置
    # 优先使用新配置，兼容旧配置
    if 'csd_encoder' in config:
        csd_config = config['csd_encoder']
    elif 'spd_encoder' in config:
        # 从旧配置转换
        spd_config = config['spd_encoder']
        csd_config = {
            'matrix_size': 16,  # CSD 原始是 16x16
            'token_dim': 4,
            'd_model': spd_config.get('projection_dim', 128),
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1,
            'projection_dim': spd_config.get('projection_dim', 128)
        }
    else:
        csd_config = {
            'matrix_size': 16,
            'token_dim': 4,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1,
            'projection_dim': 128
        }
    
    # 获取训练配置
    if 'csd_pretrain' in config['training']:
        train_config = config['training']['csd_pretrain']
    elif 'spd_pretrain' in config['training']:
        train_config = config['training']['spd_pretrain']
    else:
        train_config = {
            'epochs': 100,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'patience': 20,
            'temperature': 0.1
        }
    
    # 创建 CSD Transformer 编码器
    model = CSDTransformerEncoder(
        matrix_size=csd_config.get('matrix_size', 16),
        token_dim=csd_config.get('token_dim', 4),
        d_model=csd_config.get('d_model', 128),
        n_heads=csd_config.get('n_heads', 4),
        n_layers=csd_config.get('n_layers', 3),
        dropout=csd_config.get('dropout', 0.1),
        use_cls_token=csd_config.get('use_cls_token', True),
        projection_dim=csd_config.get('projection_dim', 128)
    )
    
    print(f"CSD Transformer Encoder created:")
    print(f"  - Matrix size: {csd_config.get('matrix_size', 16)}x{csd_config.get('matrix_size', 16)}")
    print(f"  - Token dim: {csd_config.get('token_dim', 4)}")
    print(f"  - Model dim: {csd_config.get('d_model', 128)}")
    print(f"  - Heads: {csd_config.get('n_heads', 4)}, Layers: {csd_config.get('n_layers', 3)}")
    print(f"  - Output dim: {model.output_dim}")
    
    # 创建 CSD 预训练器
    trainer = CSDPretrainer(
        model=model,
        device=config['device'],
        learning_rate=train_config.get('learning_rate', 1e-3),
        weight_decay=train_config.get('weight_decay', 1e-4),
        temperature=train_config.get('temperature', 0.1),
        loss_type='infonce'
    )
    
    # 训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config.get('epochs', 100),
        checkpoint_dir=checkpoint_dir,
        patience=train_config.get('patience', 20)
    )
    
    return trainer.model, history


# 保留旧函数名以兼容
def stage1_spd_pretrain(train_loader, val_loader, config, checkpoint_dir):
    """阶段一：SPD 预训练（已弃用，重定向到 CSD 预训练）"""
    print("\nNote: stage1_spd_pretrain is deprecated. Using stage1_csd_pretrain instead.")
    return stage1_csd_pretrain(train_loader, val_loader, config, checkpoint_dir)


def stage1_5_timae_pretrain(train_loader, val_loader, config, checkpoint_dir):
    """阶段 1.5：Ti-MAE 工况预训练（旧方案）"""
    print("\n" + "=" * 50)
    print("Stage 1.5: Ti-MAE Context Pre-training (Legacy)")
    print("=" * 50)
    
    # 创建模型
    model = TiMAEWithPhysicsLoss(
        seq_len=config['timae']['seq_len'],
        in_channels=config['timae']['in_channels'],
        patch_size=config['timae']['patch_size'],
        d_model=config['timae']['d_model'],
        n_heads=config['timae']['n_heads'],
        n_layers=config['timae']['n_layers'],
        d_ff=config['timae']['d_ff'],
        dropout=config['timae']['dropout'],
        point_ratio=config['timae']['point_ratio'],
        block_ratio=config['timae']['block_ratio'],
        block_size=config['timae']['block_size'],
        lambda_smooth=config['timae']['lambda_smooth']
    )
    
    # 创建训练器
    trainer = TiMAEPretrainer(
        model=model,
        device=config['device'],
        learning_rate=config['training']['timae_pretrain']['learning_rate'],
        weight_decay=config['training']['timae_pretrain']['weight_decay'],
        lambda_smooth=config['timae']['lambda_smooth']
    )
    
    # 训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['timae_pretrain']['epochs'],
        checkpoint_dir=checkpoint_dir,
        patience=config['training']['timae_pretrain']['patience']
    )
    
    return trainer.model, history


def stage1_5_vqvae_pretrain(train_loader, val_loader, config, checkpoint_dir):
    """
    阶段 1.5：VQ-VAE 工况预训练（新方案）
    
    使用 VQ-VAE 替代 Ti-MAE，实现离散化的工况状态编码。
    
    核心优势：
    - 码本直接对应研究报告 5.1 节的"参考状态嵌入 R"
    - 早期卷积融合学习 P1/P2 的共模+差模特征
    - 统计量注入保留绝对能级信息
    """
    print("\n" + "=" * 50)
    print("Stage 1.5: VQ-VAE Context Pre-training")
    print("=" * 50)
    
    # 获取 VQ-VAE 配置
    vqvae_config = config.get('vqvae', {})
    train_config = config['training'].get('vqvae_pretrain', {})
    
    # 创建模型
    model = DualChannelVQVAE(
        seq_len=vqvae_config.get('seq_len', config.get('data', {}).get('window_size', 1024)),
        in_channels=vqvae_config.get('in_channels', 2),
        d_model=vqvae_config.get('d_model', 128),
        n_embeddings=vqvae_config.get('n_embeddings', 16),
        encoder_channels=vqvae_config.get('encoder_channels', [32, 64, 128]),
        commitment_cost=vqvae_config.get('commitment_cost', 0.25),
        decay=vqvae_config.get('decay', 0.99),
        use_decoder=vqvae_config.get('use_decoder', True),
        lambda_recon=vqvae_config.get('lambda_recon', 0.1),
        dropout=vqvae_config.get('dropout', 0.1)
    )
    
    print(f"VQ-VAE Model created:")
    print(f"  - Sequence length: {vqvae_config.get('seq_len', 1024)}")
    print(f"  - Input channels: {vqvae_config.get('in_channels', 2)}")
    print(f"  - Output dim: {vqvae_config.get('d_model', 128)}")
    print(f"  - Codebook size: {vqvae_config.get('n_embeddings', 16)}")
    print(f"  - Encoder channels: {vqvae_config.get('encoder_channels', [32, 64, 128])}")
    
    # 创建训练器
    trainer = VQVAEPretrainer(
        model=model,
        device=config['device'],
        learning_rate=train_config.get('learning_rate', 1e-3),
        weight_decay=train_config.get('weight_decay', 1e-4),
        lambda_recon=train_config.get('lambda_recon', 0.1)
    )
    
    # 训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config.get('epochs', 50),
        checkpoint_dir=checkpoint_dir,
        patience=train_config.get('patience', 15)
    )
    
    # 打印码本使用统计
    print("\nCodebook Usage Statistics:")
    usage = model.get_codebook_usage()
    active_codes = (usage > 0.01).sum().item()
    print(f"  - Active codes: {active_codes}/{vqvae_config.get('n_embeddings', 16)}")
    print(f"  - Usage distribution: {usage.cpu().numpy()}")
    
    return trainer.model, history


def stage2_joint_finetune(
    train_loader, 
    val_loader, 
    config, 
    checkpoint_dir,
    csd_checkpoint=None,
    timae_checkpoint=None,
    spd_checkpoint=None,  # 兼容旧参数
    pressure_encoder='timae'  # 'timae' 或 'vqvae'
):
    """阶段二：条件联合微调"""
    print("\n" + "=" * 50)
    print("Stage 2: Conditional Joint Fine-tuning")
    print(f"  Pressure encoder: {pressure_encoder.upper()}")
    print("=" * 50)
    
    # 兼容旧参数
    if spd_checkpoint and not csd_checkpoint:
        csd_checkpoint = spd_checkpoint
    
    # 获取 CSD 配置
    if 'csd_encoder' in config:
        csd_config = config['csd_encoder']
    elif 'spd_encoder' in config:
        spd_config = config['spd_encoder']
        csd_config = {
            'matrix_size': 16,
            'token_dim': 4,
            'd_model': spd_config.get('projection_dim', 128),
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1,
            'projection_dim': spd_config.get('projection_dim', 128)
        }
    else:
        csd_config = {}
    
    # 获取 VQ-VAE 配置（如果使用）
    vqvae_config = config.get('vqvae', {})
    
    # 创建完整模型
    model = MultiModalAnomalyDetector(
        # CSD Transformer 配置
        csd_matrix_size=csd_config.get('matrix_size', 16),
        csd_token_dim=csd_config.get('token_dim', 4),
        csd_d_model=csd_config.get('d_model', 128),
        csd_n_heads=csd_config.get('n_heads', 4),
        csd_n_layers=csd_config.get('n_layers', 3),
        csd_projection_dim=csd_config.get('projection_dim', 128),
        # Ti-MAE 配置（即使使用 VQ-VAE，也需要这些参数用于兼容）
        seq_len=config['timae']['seq_len'],
        pressure_channels=config['timae']['in_channels'],
        patch_size=config['timae']['patch_size'],
        timae_d_model=config['timae']['d_model'],
        timae_n_heads=config['timae']['n_heads'],
        timae_n_layers=config['timae']['n_layers'],
        timae_d_ff=config['timae']['d_ff'],
        point_ratio=config['timae']['point_ratio'],
        block_ratio=config['timae']['block_ratio'],
        block_size=config['timae']['block_size'],
        lambda_smooth=config['timae']['lambda_smooth'],
        # 融合配置
        n_reference_states=config['fusion']['n_reference_states'],
        fusion_d_model=config['fusion']['d_model'],
        # 门控配置
        warmup_epochs=config['gating']['warmup_epochs'],
        dropout=config['fusion']['dropout'],
        # 压力编码器选择
        pressure_encoder=pressure_encoder,
        vqvae_config=vqvae_config
    )
    
    # 创建训练器
    trainer = JointTrainer(
        model=model,
        device=config['device'],
        learning_rate=config['training']['joint']['learning_rate'],
        weight_decay=config['training']['joint']['weight_decay'],
        warmup_epochs=config['gating']['warmup_epochs'],
        freeze_pretrained=config['training']['joint']['freeze_pretrained']
    )
    
    # 加载预训练权重
    if csd_checkpoint:
        trainer.load_pretrained_weights(csd_checkpoint=csd_checkpoint)
    if timae_checkpoint:
        trainer.load_pretrained_weights(timae_checkpoint=timae_checkpoint)
    
    # 训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['joint']['epochs'],
        checkpoint_dir=checkpoint_dir,
        patience=config['training']['joint']['patience'],
        unfreeze_after=config['training']['joint']['unfreeze_after']
    )
    
    return trainer.model, history


def stage3_threshold_calibration(model, val_loader, config, device):
    """阶段三：阈值校准"""
    print("\n" + "=" * 50)
    print("Stage 3: Threshold Calibration")
    print("=" * 50)
    
    # 创建评分器
    scorer = AnomalyScorer(model, device=device)
    
    # 在验证集上拟合正常分布
    scorer.fit_normal_distribution(val_loader)
    
    # 创建阈值
    if config['inference']['threshold_method'] == 'pot':
        threshold = POTThreshold(
            q=config['inference']['pot']['q'],
            level=config['inference']['pot']['level']
        )
    else:
        from inference.threshold import StatisticalThreshold
        threshold = StatisticalThreshold(k=config['inference']['statistical']['k'])
    
    # 拟合阈值
    threshold.fit(scorer.normal_stats['scores'])
    
    return scorer, threshold


def main():
    parser = argparse.ArgumentParser(description='多模态故障检测系统')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--data', type=str, required=True,
                        help='数据文件路径')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'csd', 'spd', 'vqvae', 'timae', 'joint', 'inference'],
                        help='运行阶段 (spd 已弃用等同于 csd; vqvae 为新方案替代 timae)')
    parser.add_argument('--pressure-encoder', type=str, default='temporal_vqvae',
                        choices=['temporal_vqvae', 'vqvae', 'timae'],
                        help='压力编码器类型：temporal_vqvae（v3.2推荐）、vqvae（旧版）或 timae（旧方案）')
    parser.add_argument('--csd-checkpoint', type=str, default=None,
                        help='CSD Transformer 预训练检查点')
    parser.add_argument('--spd-checkpoint', type=str, default=None,
                        help='SPD 预训练检查点（已弃用，等同于 --csd-checkpoint）')
    parser.add_argument('--vqvae-checkpoint', type=str, default=None,
                        help='VQ-VAE 预训练检查点')
    parser.add_argument('--timae-checkpoint', type=str, default=None,
                        help='Ti-MAE 预训练检查点')
    parser.add_argument('--joint-checkpoint', type=str, default=None,
                        help='联合训练检查点')
    
    args = parser.parse_args()
    
    # 兼容旧参数
    if args.spd_checkpoint and not args.csd_checkpoint:
        args.csd_checkpoint = args.spd_checkpoint
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建目录
    checkpoint_dir = config['checkpoint']['save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # 根据阶段执行
    # 阶段一：CSD Transformer 预训练
    if args.stage in ['all', 'csd', 'spd']:
        print("\n" + "=" * 50)
        print("Preparing data for CSD Transformer pre-training...")
        train_loader, val_loader = prepare_data(args.data, config, stage='csd')
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        
        csd_model, csd_history = stage1_csd_pretrain(
            train_loader, val_loader, config, checkpoint_dir
        )
        args.csd_checkpoint = f"{checkpoint_dir}/csd_pretrain_best.pt"
    
    # 阶段 1.5：工况预训练（VQ-VAE 或 Ti-MAE）
    # VQ-VAE 新方案（默认）
    if args.stage in ['all', 'vqvae'] or (args.stage == 'all' and args.pressure_encoder == 'vqvae'):
        print("\n" + "=" * 50)
        print("Preparing data for VQ-VAE pre-training...")
        train_loader, val_loader = prepare_data(args.data, config, stage='timae')  # 数据加载与 timae 相同
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        
        vqvae_model, vqvae_history = stage1_5_vqvae_pretrain(
            train_loader, val_loader, config, checkpoint_dir
        )
        args.vqvae_checkpoint = f"{checkpoint_dir}/vqvae_pretrain_best.pt"
        # 同时设置 timae_checkpoint 以兼容联合训练
        args.timae_checkpoint = args.vqvae_checkpoint
    
    # Ti-MAE 旧方案（通过 --pressure-encoder timae 或 --stage timae 触发）
    elif args.stage == 'timae' or (args.stage == 'all' and args.pressure_encoder == 'timae'):
        print("\n" + "=" * 50)
        print("Preparing data for Ti-MAE pre-training (Legacy)...")
        train_loader, val_loader = prepare_data(args.data, config, stage='timae')
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        
        timae_model, timae_history = stage1_5_timae_pretrain(
            train_loader, val_loader, config, checkpoint_dir
        )
        args.timae_checkpoint = f"{checkpoint_dir}/timae_pretrain_best.pt"
    
    # 阶段二：联合微调
    if args.stage in ['all', 'joint']:
        # 如果提供了 vqvae_checkpoint 但没有 timae_checkpoint，使用 vqvae_checkpoint
        if args.vqvae_checkpoint and not args.timae_checkpoint:
            args.timae_checkpoint = args.vqvae_checkpoint
        
        print("\n" + "=" * 50)
        print("Preparing data for Joint fine-tuning...")
        train_loader, val_loader = prepare_data(args.data, config, stage='joint')
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        
        full_model, joint_history = stage2_joint_finetune(
            train_loader, val_loader, config, checkpoint_dir,
            csd_checkpoint=args.csd_checkpoint,
            timae_checkpoint=args.timae_checkpoint,
            pressure_encoder=args.pressure_encoder
        )
        args.joint_checkpoint = f"{checkpoint_dir}/joint_best.pt"
    
    # 阶段三：推理
    if args.stage in ['all', 'inference']:
        # 加载模型
        if args.joint_checkpoint:
            from trainers.utils import load_checkpoint
            
            # 获取配置
            if 'csd_encoder' in config:
                csd_config = config['csd_encoder']
            else:
                csd_config = {}
            
            vqvae_config = config.get('vqvae', {})
            
            full_model = MultiModalAnomalyDetector(
                csd_matrix_size=csd_config.get('matrix_size', 16),
                csd_token_dim=csd_config.get('token_dim', 4),
                csd_d_model=csd_config.get('d_model', 128),
                csd_n_heads=csd_config.get('n_heads', 4),
                csd_n_layers=csd_config.get('n_layers', 3),
                csd_projection_dim=csd_config.get('projection_dim', 128),
                seq_len=config['timae']['seq_len'],
                pressure_channels=config['timae']['in_channels'],
                patch_size=config['timae']['patch_size'],
                timae_d_model=config['timae']['d_model'],
                timae_n_heads=config['timae']['n_heads'],
                timae_n_layers=config['timae']['n_layers'],
                # 压力编码器选择
                pressure_encoder=args.pressure_encoder,
                vqvae_config=vqvae_config
            )
            load_checkpoint(args.joint_checkpoint, full_model, device=config['device'])
        
        scorer, threshold = stage3_threshold_calibration(
            full_model, val_loader, config, config['device']
        )
        
        print(f"\nFinal threshold: {threshold.get_threshold():.4f}")
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
