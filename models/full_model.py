"""
完整的多模态异常检测模型

整合 CSD Transformer、VQ-VAE/Ti-MAE 和跨模态融合模块。

v3.2 更新：新增 VQ-VAE 双通道工况编码器
- VQ-VAE 替代 Ti-MAE 作为压力分支
- 码本直接对应研究报告 5.1 节的"参考状态嵌入"
- 统计量注入保留绝对能级信息

v3.1 更新：将 SPDNet 替换为 CSD Pair-Token Transformer
- 原方案：SPDNet (BiMap/ReEig/LogEig) - 基于黎曼流形
- 新方案：CSD Transformer - 基于 Self-Attention
- 优点：训练速度提升 50-100 倍，数值稳定，物理可解释
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

# 新方案：CSD Transformer
from .csd_transformer import CSDTransformerEncoder

# 新方案：VQ-VAE 工况编码器
from .vqvae import DualChannelVQVAE, VQVAEWithPhysicsLoss

# v3.2 时序 VQ-VAE（保留时序结构，增强统计聚合）
from .vqvae import TemporalVQVAE, TemporalVQVAEWithPhysicsLoss

# 保留旧模块以支持历史权重加载（可选）
from .spd.spd_encoder import SPDEncoder, SPDEncoderWithProjection
from .spd.spd_decoder import SPDDecoder, SPDAutoEncoder

# Ti-MAE 和融合模块
from .timae.timae import PhysicsAwareTiMAE, TiMAEWithPhysicsLoss
from .fusion.context_attention import ContextConditionedAttention, CrossModalFusion
from .fusion.synergy_module import RiemannianSynergyModule
from .fusion.gating import LossGatingNetwork, GatedLossComputer


class MultiModalAnomalyDetector(nn.Module):
    """
    多模态异常检测器
    
    架构：
    1. 振动分支：16×16 CSD矩阵 → CSD Transformer → Z_CSD
    2. 压力分支：压力信号 → Ti-MAE → Q_context
    3. 跨模态融合：条件注意力 + 协同感知 + 门控加权
    
    v3.1 更新：
    - 使用 CSD Pair-Token Transformer 替代 SPDNet
    - 上三角 136 个元素作为 Token，学习全局耦合关系
    - 物理位置编码保留块结构信息（P-V耦合/跨臂等）
    
    Args:
        # CSD Transformer 配置
        csd_matrix_size: CSD 矩阵大小（默认 16）
        csd_token_dim: Token 特征维度（默认 4: Re/Im/Mag/Phase）
        csd_d_model: Transformer 模型维度
        csd_n_heads: 注意力头数
        csd_n_layers: Encoder 层数
        csd_projection_dim: 输出投影维度
        
        # Ti-MAE 配置
        seq_len: 压力信号序列长度
        pressure_channels: 压力通道数
        patch_size: Ti-MAE patch 大小
        timae_d_model: Ti-MAE 模型维度
        timae_n_heads: Ti-MAE 注意力头数
        timae_n_layers: Ti-MAE 层数
        point_ratio: 点状遮蔽比例
        block_ratio: 块状遮蔽比例
        block_size: 块大小
        lambda_smooth: 平滑约束权重
        
        # 融合配置
        n_reference_states: 参考状态数量
        fusion_d_model: 融合层维度
        
        # 门控配置
        warmup_epochs: 门控预热期
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        # CSD Transformer 配置
        csd_matrix_size: int = 16,
        csd_token_dim: int = 4,
        csd_d_model: int = 128,
        csd_n_heads: int = 4,
        csd_n_layers: int = 3,
        csd_projection_dim: int = 128,
        # Ti-MAE 配置
        seq_len: int = 1024,
        pressure_channels: int = 2,
        patch_size: int = 16,
        timae_d_model: int = 128,
        timae_n_heads: int = 4,
        timae_n_layers: int = 4,
        timae_d_ff: int = 512,
        point_ratio: float = 0.5,
        block_ratio: float = 0.2,
        block_size: int = 4,
        lambda_smooth: float = 5.0,
        # 融合配置
        n_reference_states: int = 4,
        fusion_d_model: int = 128,
        # 门控配置
        warmup_epochs: int = 5,
        dropout: float = 0.1,
        # 压力编码器选择
        pressure_encoder: str = 'timae',  # 'timae' 或 'vqvae'
        vqvae_config: dict = None,  # VQ-VAE 配置
        # 兼容性参数（旧接口，将被忽略）
        spd_input_size: int = None,
        spd_hidden_sizes: list = None,
        spd_projection_dim: int = None
    ):
        super(MultiModalAnomalyDetector, self).__init__()
        
        self.csd_matrix_size = csd_matrix_size
        self.lambda_smooth = lambda_smooth
        self.pressure_encoder_type = pressure_encoder
        
        # ==================== 振动分支：CSD Transformer ====================
        self.csd_encoder = CSDTransformerEncoder(
            matrix_size=csd_matrix_size,
            token_dim=csd_token_dim,
            d_model=csd_d_model,
            n_heads=csd_n_heads,
            n_layers=csd_n_layers,
            dropout=dropout,
            use_cls_token=True,
            projection_dim=csd_projection_dim
        )
        
        # 为了兼容性，保留 spd_encoder 别名
        self.spd_encoder = self.csd_encoder
        
        # ==================== 压力分支 ====================
        if pressure_encoder == 'temporal_vqvae':
            # v3.2 时序 VQ-VAE（推荐）- 保留时序结构，增强统计聚合
            vqvae_cfg = vqvae_config or {}
            self.timae = TemporalVQVAEWithPhysicsLoss(
                seq_len=seq_len,
                in_channels=pressure_channels,
                d_model=timae_d_model,
                n_embeddings=vqvae_cfg.get('n_embeddings', 8),
                n_temporal=vqvae_cfg.get('n_temporal', 16),
                commitment_cost=vqvae_cfg.get('commitment_cost', 0.5),
                lambda_recon=vqvae_cfg.get('lambda_recon', 0.1),
                dropout=dropout,
                median_kernel=vqvae_cfg.get('median_kernel', 5)
            )
        elif pressure_encoder == 'vqvae':
            # 旧版 VQ-VAE（全局池化）
            vqvae_cfg = vqvae_config or {}
            self.timae = VQVAEWithPhysicsLoss(
                seq_len=seq_len,
                in_channels=pressure_channels,
                d_model=timae_d_model,
                n_embeddings=vqvae_cfg.get('n_embeddings', 16),
                encoder_channels=vqvae_cfg.get('encoder_channels', [32, 64, 128]),
                commitment_cost=vqvae_cfg.get('commitment_cost', 0.25),
                decay=vqvae_cfg.get('decay', 0.99),
                lambda_recon=vqvae_cfg.get('lambda_recon', 0.1),
                dropout=dropout
            )
        else:
            # Ti-MAE（默认）
            self.timae = TiMAEWithPhysicsLoss(
                seq_len=seq_len,
                in_channels=pressure_channels,
                patch_size=patch_size,
                d_model=timae_d_model,
                n_heads=timae_n_heads,
                n_layers=timae_n_layers,
                d_ff=timae_d_ff,
                dropout=dropout,
                point_ratio=point_ratio,
                block_ratio=block_ratio,
                block_size=block_size,
                lambda_smooth=lambda_smooth
            )
        
        # ==================== 跨模态融合 ====================
        self.fusion = CrossModalFusion(
            d_spd=csd_projection_dim,  # 使用 CSD encoder 输出维度
            d_context=timae_d_model,
            d_model=fusion_d_model,
            n_heads=4,
            n_reference_states=n_reference_states,
            dropout=dropout
        )
        
        # 协同感知模块：使用简化的向量距离
        # 注意：原 RiemannianSynergyModule 需要 SPD 矩阵输入
        # 这里改为基于 CSD 特征向量的距离
        self.synergy = SimplifiedSynergyModule(
            feature_dim=csd_projection_dim,
            csd_matrix_size=csd_matrix_size
        )
        
        # ==================== 门控网络 ====================
        # v3.3 更新：移除 contrast 占位损失（对比学习在阶段一完成，联合阶段不参与门控）
        # 新配置：[consistency, synergy]
        self.gated_loss = GatedLossComputer(
            d_context=timae_d_model,
            loss_names=['consistency', 'synergy'],
            hidden_dim=32,
            w_base=[1.0, 0.5],  # consistency 最重要，synergy 次之
            w_min=0.10,          # 安全下界，防止任何损失被完全忽略
            warmup_epochs=warmup_epochs,
            temperature=3.0      # 防止 softmax 赢者通吃坍缩
        )
        
        # 输出维度
        self.output_dim = fusion_d_model
    
    def forward(
        self,
        csd_matrix: torch.Tensor,
        pressure: torch.Tensor,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播
        
        Args:
            csd_matrix: CSD 矩阵
                - 复数格式: (B, 16, 16) complex
                - 实数格式: (B, 32, 32) real
            pressure: 压力信号，shape (B, T, C)
            training: 是否训练模式
            
        Returns:
            包含各种特征和损失的字典
        """
        result = {}
        
        # ==================== 振动分支 ====================
        # CSD Transformer 编码
        z_csd = self.csd_encoder(csd_matrix)  # (B, projection_dim)
        result['z_csd'] = z_csd
        result['z_spd'] = z_csd  # 兼容性别名
        
        # 协同距离
        d_synergy = self.synergy(csd_matrix, z_csd)  # (B,)
        result['d_synergy'] = d_synergy
        
        # ==================== 压力分支 ====================
        if training:
            timae_out = self.timae(pressure, return_losses=True)
            result['timae_recon'] = timae_out['recon']
            result['timae_mask'] = timae_out['mask']
            result['l_recon'] = timae_out['l_recon']
            result['l_smooth'] = timae_out['l_smooth']
        
        # 获取工况上下文
        q_context = self.timae.get_context(pressure)  # (B, d_model)
        result['q_context'] = q_context
        
        # ==================== 跨模态融合 ====================
        fused, z_expected = self.fusion(z_csd, q_context)
        result['fused'] = fused
        result['z_expected'] = z_expected
        
        # ==================== 计算条件一致性损失 ====================
        # 保留 per-sample 维度 (B,)，供门控网络逐样本加权
        consistency_loss = ((z_expected - z_csd) ** 2).sum(dim=-1)  # (B,)
        result['l_consistency'] = consistency_loss
        
        return result
    
    def get_anomaly_score(
        self,
        csd_matrix: torch.Tensor,
        pressure: torch.Tensor
    ) -> torch.Tensor:
        """
        计算异常分数
        
        Args:
            csd_matrix: CSD 矩阵
            pressure: 压力信号
            
        Returns:
            异常分数，shape (B,)
        """
        with torch.no_grad():
            result = self.forward(csd_matrix, pressure, training=False)
            
            z_csd = result['z_csd']
            z_expected = result['z_expected']
            d_synergy = result['d_synergy']
            q_context = result['q_context']
            
            # 获取门控权重
            # v3.3 更新：2个权重 [consistency, synergy]（移除无效的 contrast 占位）
            weights = self.gated_loss.get_weights(q_context)  # (B, 2)
            
            # 条件一致性偏差
            consistency_score = ((z_expected - z_csd) ** 2).sum(dim=-1)
            
            # 加权组合：w_consistency * consistency + w_synergy * synergy
            score = weights[:, 0] * consistency_score + weights[:, 1] * d_synergy
            
            return score
    
    def encode_csd(self, csd_matrix: torch.Tensor) -> torch.Tensor:
        """编码 CSD 矩阵"""
        return self.csd_encoder(csd_matrix)
    
    def encode_spd(self, csd_matrix: torch.Tensor) -> torch.Tensor:
        """编码 CSD 矩阵（兼容性别名）"""
        return self.encode_csd(csd_matrix)
    
    def encode_pressure(self, pressure: torch.Tensor) -> torch.Tensor:
        """编码压力信号"""
        return self.timae.get_context(pressure)
    
    def update_epoch(self, epoch: int, total_epochs: int):
        """更新 epoch（用于门控预热）"""
        self.gated_loss.update_epoch(epoch, total_epochs)


class SimplifiedSynergyModule(nn.Module):
    """
    简化的协同感知模块
    
    计算 Arm1 和 Arm2 的特征距离，替代原有的黎曼距离。
    
    方法：
    1. 从 CSD 矩阵提取 Arm1 和 Arm2 的子块
    2. 分别编码为向量
    3. 计算欧氏距离或余弦距离
    
    Args:
        feature_dim: 特征维度
        csd_matrix_size: CSD 矩阵大小
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        csd_matrix_size: int = 16
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.csd_matrix_size = csd_matrix_size
        
        # Arm1: 通道 0-7 (P1 + V1)
        # Arm2: 通道 8-15 (P2 + V2)
        self.arm1_size = csd_matrix_size // 2
        self.arm2_size = csd_matrix_size // 2
        
        # 子块特征提取
        # Arm 子块是 8x8 复数矩阵，上三角有 36 个元素
        n_arm_pairs = (self.arm1_size * (self.arm1_size + 1)) // 2
        
        self.arm1_proj = nn.Sequential(
            nn.Linear(n_arm_pairs * 2, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4)
        )
        
        self.arm2_proj = nn.Sequential(
            nn.Linear(n_arm_pairs * 2, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 4)
        )
    
    def _extract_arm_features(
        self,
        csd_matrix: torch.Tensor,
        arm_start: int,
        arm_end: int
    ) -> torch.Tensor:
        """
        提取臂的子块特征
        
        Args:
            csd_matrix: CSD 矩阵
            arm_start: 臂起始通道
            arm_end: 臂结束通道
            
        Returns:
            臂特征向量
        """
        B = csd_matrix.size(0)
        
        # 处理实数化矩阵
        if csd_matrix.size(-1) == 32 and not csd_matrix.is_complex():
            n = self.csd_matrix_size
            real_part = csd_matrix[:, :n, :n]
            imag_part = csd_matrix[:, n:, :n]
            csd_complex = torch.complex(real_part, imag_part)
        elif csd_matrix.is_complex():
            csd_complex = csd_matrix
        else:
            csd_complex = torch.complex(csd_matrix, torch.zeros_like(csd_matrix))
        
        # 提取子块
        sub_block = csd_complex[:, arm_start:arm_end, arm_start:arm_end]
        
        # 提取上三角
        size = arm_end - arm_start
        rows, cols = torch.triu_indices(size, size, offset=0, device=csd_matrix.device)
        upper_tri = sub_block[:, rows, cols]  # (B, n_pairs)
        
        # 实部虚部拼接
        features = torch.cat([upper_tri.real, upper_tri.imag], dim=-1)  # (B, n_pairs * 2)
        
        return features
    
    def forward(
        self,
        csd_matrix: torch.Tensor,
        z_csd: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算协同距离
        
        Args:
            csd_matrix: CSD 矩阵
            z_csd: CSD 特征（可选，用于增强）
            
        Returns:
            协同距离，shape (B,)
        """
        # 提取 Arm1 和 Arm2 特征
        arm1_feat = self._extract_arm_features(csd_matrix, 0, self.arm1_size)
        arm2_feat = self._extract_arm_features(csd_matrix, self.arm1_size, self.csd_matrix_size)
        
        # 投影
        arm1_proj = self.arm1_proj(arm1_feat)
        arm2_proj = self.arm2_proj(arm2_feat)
        
        # 计算距离（欧氏距离）
        d_synergy = torch.norm(arm1_proj - arm2_proj, p=2, dim=-1)
        
        return d_synergy


class CSDPretrainModel(nn.Module):
    """
    CSD Transformer 预训练模型（阶段一）
    
    用于对比学习预训练。
    """
    
    def __init__(
        self,
        matrix_size: int = 16,
        token_dim: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        projection_dim: int = 128
    ):
        super(CSDPretrainModel, self).__init__()
        
        self.encoder = CSDTransformerEncoder(
            matrix_size=matrix_size,
            token_dim=token_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_cls_token=True,
            projection_dim=projection_dim
        )
    
    def forward(self, csd_matrix: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.encoder(csd_matrix)
    
    def encode(self, csd_matrix: torch.Tensor) -> torch.Tensor:
        """编码"""
        return self.encoder(csd_matrix)


# 保留旧类名以兼容历史代码
class SPDPretrainModel(nn.Module):
    """
    SPD 预训练模型（阶段一）- 已弃用
    
    保留以兼容历史权重加载。
    新代码请使用 CSDPretrainModel。
    """
    
    def __init__(
        self,
        input_size: int = 16,
        hidden_sizes: list = [12, 8],
        epsilon: float = 1e-4
    ):
        super(SPDPretrainModel, self).__init__()
        
        self.autoencoder = SPDAutoEncoder(
            input_size=input_size,
            encoder_hidden_sizes=hidden_sizes,
            epsilon=epsilon
        )
    
    def forward(self, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.autoencoder(M)
    
    def encode(self, M: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(M)


class TiMAEPretrainModel(nn.Module):
    """
    Ti-MAE 预训练模型（阶段1.5）- 已弃用
    
    用于工况特征学习。
    新代码请使用 VQVAEPretrainModel。
    """
    
    def __init__(
        self,
        seq_len: int = 1024,
        in_channels: int = 2,
        patch_size: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        point_ratio: float = 0.5,
        block_ratio: float = 0.2,
        block_size: int = 4,
        lambda_smooth: float = 5.0
    ):
        super(TiMAEPretrainModel, self).__init__()
        
        self.model = TiMAEWithPhysicsLoss(
            seq_len=seq_len,
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            point_ratio=point_ratio,
            block_ratio=block_ratio,
            block_size=block_size,
            lambda_smooth=lambda_smooth
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return self.model(x, return_losses=True)
    
    def get_context(self, x: torch.Tensor) -> torch.Tensor:
        """获取工况上下文"""
        return self.model.get_context(x)


class VQVAEPretrainModel(nn.Module):
    """
    VQ-VAE 预训练模型（阶段1.5）- 新方案
    
    用于离散化的工况状态编码，替代 Ti-MAE。
    
    核心优势：
    - 码本直接对应研究报告 5.1 节的"参考状态嵌入 R"
    - 早期卷积融合学习 P1/P2 的共模+差模特征
    - 统计量注入保留绝对能级信息
    
    Args:
        seq_len: 序列长度
        in_channels: 输入通道数（默认 2：P1, P2）
        d_model: 输出维度
        n_embeddings: 码本大小（工况类别数）
        encoder_channels: 编码器各层通道数
        commitment_cost: Commitment Loss 权重
        decay: EMA 衰减率
        use_decoder: 是否使用解码器
        lambda_recon: 重构损失权重
    """
    
    def __init__(
        self,
        seq_len: int = 1024,
        in_channels: int = 2,
        d_model: int = 128,
        n_embeddings: int = 16,
        encoder_channels: List[int] = [32, 64, 128],
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        use_decoder: bool = True,
        lambda_recon: float = 0.1,
        dropout: float = 0.1,
        # 以下参数是为了兼容 Ti-MAE 接口而存在，实际不使用
        patch_size: int = None,
        n_heads: int = None,
        n_layers: int = None,
        d_ff: int = None,
        point_ratio: float = None,
        block_ratio: float = None,
        block_size: int = None,
        lambda_smooth: float = None
    ):
        super(VQVAEPretrainModel, self).__init__()
        
        self.model = DualChannelVQVAE(
            seq_len=seq_len,
            in_channels=in_channels,
            d_model=d_model,
            n_embeddings=n_embeddings,
            encoder_channels=encoder_channels,
            commitment_cost=commitment_cost,
            decay=decay,
            use_decoder=use_decoder,
            lambda_recon=lambda_recon,
            dropout=dropout
        )
        
        self.output_dim = d_model
        self.n_embeddings = n_embeddings
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return self.model(x, return_losses=True)
    
    def get_context(self, x: torch.Tensor) -> torch.Tensor:
        """获取工况上下文"""
        return self.model.get_context(x)
    
    def get_codebook_usage(self) -> torch.Tensor:
        """获取码本使用率"""
        return self.model.get_codebook_usage()
    
    def get_codebook_vectors(self) -> torch.Tensor:
        """获取码本向量"""
        return self.model.get_codebook_vectors()
