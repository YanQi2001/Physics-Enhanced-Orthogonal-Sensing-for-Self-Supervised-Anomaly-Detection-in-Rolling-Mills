"""
阶段二：条件联合微调

加载预训练权重，联合训练完整模型，使用门控加权损失。

v3.1 更新：支持 CSD Transformer 预训练权重加载
- 原方案：SPDNet (BiMap/ReEig/LogEig) - 基于黎曼流形
- 新方案：CSD Transformer - 基于 Self-Attention
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from tqdm import tqdm

from models.full_model import MultiModalAnomalyDetector
from losses.physics_loss import ConsistencyLoss, SynergyLoss
from .utils import MetricTracker, EarlyStopping, save_checkpoint, load_checkpoint


class JointTrainer:
    """
    条件联合训练器
    
    特点：
    1. 加载预训练的 CSD Transformer 和 Ti-MAE 权重
    2. 使用门控加权损失
    3. 支持门控预热和渐进启用
    
    v3.1 更新：
    - 支持 CSD Transformer 预训练权重加载（替代 SPDNet）
    - 兼容旧的 SPD 权重加载（用于历史模型）
    
    注意：对比学习已移至阶段一（CSD预训练），联合阶段专注于跨模态融合。
    
    Args:
        model: 完整的多模态异常检测模型
        device: 训练设备
        learning_rate: 学习率
        weight_decay: 权重衰减
        warmup_epochs: 门控预热期
        freeze_pretrained: 是否冻结预训练层
    """
    
    def __init__(
        self,
        model: MultiModalAnomalyDetector,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        freeze_pretrained: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.freeze_pretrained = freeze_pretrained
        
        # 如果冻结预训练层
        if freeze_pretrained:
            self._freeze_pretrained_layers()
        
        # v3.4 更新：门控网络使用单独的参数组，学习率 10 倍于主网络
        gating_param_ids = set(id(p) for p in model.gated_loss.parameters())
        gating_params = [p for p in model.gated_loss.parameters() if p.requires_grad]
        other_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) not in gating_param_ids]
        
        self.optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': learning_rate},
            {'params': gating_params, 'lr': learning_rate * 10, 'weight_decay': 0.0},
        ], weight_decay=weight_decay, betas=(0.9, 0.999))
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 损失函数（不包含对比学习，对比学习在阶段一完成）
        # v3.3 更新：仅保留 consistency 和 synergy 两项损失
        self.consistency_loss = ConsistencyLoss()
        self.synergy_loss = SynergyLoss()
        
        # 指标追踪
        self.metrics = MetricTracker()
        
        # 当前 epoch
        self.current_epoch = 0
        self.total_epochs = 100
    
    def _freeze_pretrained_layers(self):
        """冻结预训练层"""
        # 冻结 CSD Transformer 编码器
        # 注意：model.csd_encoder 是 CSDTransformerEncoder 实例
        for param in self.model.csd_encoder.parameters():
            param.requires_grad = False
        
        # 冻结 Ti-MAE
        for param in self.model.timae.timae.parameters():
            param.requires_grad = False
        
        print("Pretrained layers frozen (CSD Transformer + Ti-MAE)")
    
    def _unfreeze_all(self):
        """解冻所有层"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("All layers unfrozen")
    
    def load_pretrained_weights(
        self,
        csd_checkpoint: str = None,
        timae_checkpoint: str = None,
        spd_checkpoint: str = None  # 兼容旧参数
    ):
        """
        加载预训练权重
        
        v3.1 更新：支持 CSD Transformer 权重加载
        
        Args:
            csd_checkpoint: CSD Transformer 预训练检查点路径
            timae_checkpoint: Ti-MAE 预训练检查点路径
            spd_checkpoint: SPD 预训练检查点路径（已弃用，等同于 csd_checkpoint）
        """
        # 兼容旧参数
        if spd_checkpoint and not csd_checkpoint:
            csd_checkpoint = spd_checkpoint
        
        if csd_checkpoint:
            checkpoint = torch.load(csd_checkpoint, map_location=self.device)
            model_state = checkpoint.get('model_state_dict', checkpoint)
            
            # 尝试直接加载（CSD Transformer 格式）
            # CSD Transformer 的 checkpoint 中模型状态直接对应 CSDTransformerEncoder
            try:
                # 新格式：直接加载 CSDTransformerEncoder 权重
                self.model.csd_encoder.load_state_dict(model_state, strict=False)
                print(f"Loaded CSD Transformer weights from {csd_checkpoint}")
            except Exception as e:
                # 旧格式：尝试提取编码器部分
                print(f"Trying alternative weight loading method... ({e})")
                encoder_state = {}
                for k, v in model_state.items():
                    if k.startswith('encoder.'):
                        new_key = k.replace('encoder.', '')
                        encoder_state[new_key] = v
                    else:
                        encoder_state[k] = v
                
                self.model.csd_encoder.load_state_dict(encoder_state, strict=False)
                print(f"Loaded pretrained weights from {csd_checkpoint} (alternative format)")
        
        if timae_checkpoint:
            checkpoint = torch.load(timae_checkpoint, map_location=self.device)
            model_state = checkpoint.get('model_state_dict', checkpoint)
            
            # 检查是否为时序 VQ-VAE checkpoint
            is_temporal_vqvae = any('encoder.conv' in k or 'quantizer.embedding' in k for k in model_state.keys())
            
            if is_temporal_vqvae:
                # 时序 VQ-VAE v3.2 格式
                # 权重需要映射到 self.model.timae.vqvae
                if hasattr(self.model.timae, 'vqvae'):
                    # TemporalVQVAEWithPhysicsLoss 包装类
                    self.model.timae.vqvae.load_state_dict(model_state, strict=False)
                    
                    # 加载数据统计量
                    if 'data_mean' in checkpoint:
                        self.model.timae.vqvae.data_mean = checkpoint['data_mean'].to(self.device)
                        self.model.timae.vqvae.data_std = checkpoint['data_std'].to(self.device)
                    
                    print(f"Loaded Temporal VQ-VAE v3.2 weights from {timae_checkpoint}")
                else:
                    # 直接加载到 timae（可能是 TemporalVQVAE）
                    self.model.timae.load_state_dict(model_state, strict=False)
                    
                    if 'data_mean' in checkpoint:
                        self.model.timae.data_mean = checkpoint['data_mean'].to(self.device)
                        self.model.timae.data_std = checkpoint['data_std'].to(self.device)
                    
                    print(f"Loaded Temporal VQ-VAE weights from {timae_checkpoint}")
            else:
                # 旧格式：Ti-MAE 或 旧版 VQ-VAE
                timae_state = {}
                for k, v in model_state.items():
                    if k.startswith('model.') or k.startswith('timae.'):
                        timae_state[k] = v
                    else:
                        timae_state[k] = v
                
                self.model.timae.load_state_dict(timae_state, strict=False)
                print(f"Loaded Ti-MAE/VQ-VAE pretrained weights from {timae_checkpoint}")
    
    def train_epoch(
        self, 
        dataloader: DataLoader,
        epoch: int,
        verbose_batches: int = 0  # 前 N 个 batch 打印详细信息，0 表示不打印
    ) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch
            verbose_batches: 前 N 个 batch 打印详细信息
            
        Returns:
            训练指标
        """
        self.model.train()
        self.metrics.reset()
        self.current_epoch = epoch
        
        # 更新门控调度
        self.model.update_epoch(epoch, self.total_epochs)
        
        pbar = tqdm(dataloader, desc=f'Joint Train (Epoch {epoch+1})')
        for batch_idx, batch in enumerate(pbar):
            # 检查是否为 SequentialMultiModalDataset 格式
            is_sequential = 'current' in batch
            
            if is_sequential:
                # SequentialMultiModalDataset 格式
                csd_matrix = batch['current']['csd_real'].to(self.device)
                pressure = batch['current']['pressure'].to(self.device)
            else:
                # 普通 MultiModalDataset 格式
                csd_matrix = batch['csd_real'].to(self.device)
                pressure = batch['pressure'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(csd_matrix, pressure, training=True)
            
            # 计算各项损失（联合阶段专注于跨模态融合，不包含对比学习）
            # v3.3 更新：移除 contrast 占位（对比学习在阶段一完成，联合阶段不需要）
            # v3.4 修复：保留 per-sample 维度 (B,)，让门控网络获得逐样本梯度信号
            losses = {
                'consistency': output['l_consistency'],   # (B,) per-sample
                'synergy': output['d_synergy'],           # (B,) per-sample，不再经过 SynergyLoss.mean()
            }
            
            # Ti-MAE 损失
            if 'l_recon' in output:
                losses['recon'] = output['l_recon']
                losses['smooth'] = output['l_smooth']
            
            # 注意：对比学习已移至阶段一（SPD预训练）
            # 联合阶段的 SPDNet 已经具备流形敏感性，无需重复训练
            
            # 使用门控加权
            q_context = output['q_context']
            gated_result = self.model.gated_loss(q_context, losses)
            
            total_loss = gated_result['total']
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录当前 batch 的损失（用于详细日志）
            batch_loss = total_loss.item()
            batch_consistency = losses['consistency'].mean().item()  # per-sample → scalar
            batch_synergy = losses['synergy'].mean().item()          # per-sample → scalar
            
            # 记录指标
            batch_size = csd_matrix.size(0)
            self.metrics.update('loss', batch_loss, batch_size)
            self.metrics.update('consistency', batch_consistency, batch_size)
            self.metrics.update('synergy', batch_synergy, batch_size)
            
            # 记录权重（v3.3 更新：2个权重 [consistency, synergy]）
            if 'weights' in gated_result:
                weights = gated_result['weights'].mean(dim=0)
                for i, name in enumerate(['w_consistency', 'w_synergy']):
                    if i < weights.size(0):
                        self.metrics.update(name, weights[i].item(), batch_size)
            
            # 详细的 batch 日志（前 verbose_batches 个 batch，或每隔 100 个 batch）
            if verbose_batches > 0 and (batch_idx < verbose_batches or batch_idx % 100 == 0):
                print(f"\n  [Batch {batch_idx+1}/{len(dataloader)}] "
                      f"loss={batch_loss:.6f}, consist={batch_consistency:.6f}, "
                      f"synergy={batch_synergy:.6f}, avg_loss={self.metrics.get('loss'):.6f}")
            
            pbar.set_postfix({
                'loss': f"{batch_loss:.4g}",  # 显示当前 batch 损失
                'avg': f"{self.metrics.get('loss'):.4g}",  # 显示累计平均
                'consist': f"{batch_consistency:.4g}",
                'synergy': f"{batch_synergy:.4g}"
            })
        
        return self.metrics.get_all()
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            验证指标
        """
        self.model.eval()
        self.metrics.reset()
        
        for batch in dataloader:
            csd_matrix = batch['csd_real'].to(self.device)
            pressure = batch['pressure'].to(self.device)
            
            output = self.model(csd_matrix, pressure, training=False)
            
            # 计算异常分数
            anomaly_score = self.model.get_anomaly_score(csd_matrix, pressure)
            
            batch_size = csd_matrix.size(0)
            self.metrics.update('val_loss', output['l_consistency'].mean().item(), batch_size)
            self.metrics.update('val_synergy', output['d_synergy'].mean().item(), batch_size)
            self.metrics.update('anomaly_score', anomaly_score.mean().item(), batch_size)
        
        return self.metrics.get_all()
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        checkpoint_dir: str = None,
        patience: int = 20,
        unfreeze_after: int = 10
    ) -> Dict[str, list]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            checkpoint_dir: 检查点保存目录
            patience: 早停耐心值
            unfreeze_after: 多少个 epoch 后解冻预训练层
            
        Returns:
            训练历史
        """
        self.total_epochs = epochs
        # v3.3 更新：移除 contrast，仅保留 consistency 和 synergy
        history = {
            'train_loss': [], 'val_loss': [],
            'consistency': [], 'synergy': [],
            'w_consistency': [], 'w_synergy': []
        }
        early_stopping = EarlyStopping(patience=patience, mode='min')
        best_loss = float('inf')
        
        # 在训练开始前初始化参考状态
        if self.current_epoch == 0:
            print("Initializing reference states from data...")
            self._initialize_reference_states(train_loader)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 解冻预训练层
            if self.freeze_pretrained and epoch >= unfreeze_after:
                self._unfreeze_all()
                self.freeze_pretrained = False
            
            # 训练（第一个 epoch 打印前 20 个 batch 的详细信息）
            verbose = 20 if epoch == 0 else 0
            train_metrics = self.train_epoch(train_loader, epoch, verbose_batches=verbose)
            history['train_loss'].append(train_metrics['loss'])
            history['consistency'].append(train_metrics['consistency'])
            history['synergy'].append(train_metrics['synergy'])
            
            # 记录权重（v3.3 更新：[consistency, synergy]）
            if 'w_consistency' in train_metrics:
                history['w_consistency'].append(train_metrics['w_consistency'])
            if 'w_synergy' in train_metrics:
                history['w_synergy'].append(train_metrics['w_synergy'])
            
            # 更新学习率
            self.scheduler.step()
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                
                current_loss = val_metrics['val_loss']
                print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"Consistency: {train_metrics['consistency']:.4f}, Synergy: {train_metrics['synergy']:.4f}")
            else:
                current_loss = train_metrics['loss']
                print(f"Train Loss: {train_metrics['loss']:.4f}")
            
            # 保存最佳模型
            if current_loss < best_loss:
                best_loss = current_loss
                if checkpoint_dir:
                    save_checkpoint(
                        f"{checkpoint_dir}/joint_best.pt",
                        self.model,
                        self.optimizer,
                        epoch=epoch,
                        best_loss=best_loss
                    )
            
            # 定期保存
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                save_checkpoint(
                    f"{checkpoint_dir}/joint_epoch{epoch+1}.pt",
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    best_loss=best_loss
                )
            
            # 早停检查
            if early_stopping(current_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def visualize_gating_weights(
        self, 
        dataloader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        可视化门控权重随工况变化
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            权重历史
        """
        self.model.eval()
        # v3.3 更新：2 个权重 [consistency, synergy]
        weights_history = {f'w_{i}': [] for i in range(2)}
        
        with torch.no_grad():
            for batch in dataloader:
                # 兼容两种数据格式
                if 'current' in batch:
                    pressure = batch['current']['pressure'].to(self.device)
                else:
                    pressure = batch['pressure'].to(self.device)
                q_context = self.model.encode_pressure(pressure)
                weights = self.model.gated_loss.get_weights(q_context)
                
                for i in range(min(2, weights.size(1))):
                    weights_history[f'w_{i}'].extend(weights[:, i].cpu().tolist())
        
        return weights_history
    
    def _initialize_reference_states(self, dataloader: DataLoader, max_samples: int = 5000):
        """
        收集数据并初始化参考状态
        
        使用 K-Means 从训练数据中聚类得到 4 种典型工况状态，
        然后用各簇的平均 CSD 特征初始化参考流形嵌入。
        
        v3.1 更新：使用 CSD Transformer 编码特征
        
        Args:
            dataloader: 训练数据加载器
            max_samples: 最大采样数量（避免内存溢出）
        """
        q_contexts, z_csds = [], []
        n_collected = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # 兼容两种数据格式
                if 'current' in batch:
                    csd = batch['current']['csd_real'].to(self.device)
                    pressure = batch['current']['pressure'].to(self.device)
                else:
                    csd = batch['csd_real'].to(self.device)
                    pressure = batch['pressure'].to(self.device)
                
                # 使用 CSD Transformer 编码
                z_csd = self.model.encode_csd(csd)
                q_context = self.model.encode_pressure(pressure)
                
                q_contexts.append(q_context.cpu())
                z_csds.append(z_csd.cpu())
                
                n_collected += csd.size(0)
                if n_collected >= max_samples:
                    break
        
        if len(q_contexts) == 0:
            print("Warning: No data collected for reference initialization")
            return
        
        q_contexts = torch.cat(q_contexts, dim=0)
        z_csds = torch.cat(z_csds, dim=0)
        
        # 调用融合模块的初始化方法
        self.model.fusion.attention.initialize_references_from_data(q_contexts, z_csds)

