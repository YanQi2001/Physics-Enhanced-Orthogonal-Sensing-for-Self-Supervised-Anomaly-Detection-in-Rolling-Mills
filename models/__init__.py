"""
模型模块

v3.2 更新：新增 VQ-VAE 双通道工况编码器

模块列表：
- CSD Transformer: Pair-Token Transformer（新方案）
- VQ-VAE: 双通道工况编码器（新方案，替代 Ti-MAE）
- SPDNet: 流形神经网络（旧方案，保留以兼容）
- Ti-MAE: 多尺度物理感知时序掩码自编码器（旧方案）
- Fusion: 跨模态融合模块
"""

from .full_model import (
    MultiModalAnomalyDetector,
    CSDPretrainModel,
    SPDPretrainModel,  # 保留以兼容历史权重
    TiMAEPretrainModel,
    VQVAEPretrainModel,  # VQ-VAE 新方案
)

from .csd_transformer import CSDTransformerEncoder
from .vqvae import DualChannelVQVAE, VQVAEWithPhysicsLoss, VectorQuantizer

__all__ = [
    # 完整模型
    'MultiModalAnomalyDetector',
    # 预训练模型
    'CSDPretrainModel',
    'VQVAEPretrainModel',  # VQ-VAE 新方案
    'TiMAEPretrainModel',  # Ti-MAE 旧方案
    # 编码器
    'CSDTransformerEncoder',
    'DualChannelVQVAE',
    'VQVAEWithPhysicsLoss',
    'VectorQuantizer',
    # 旧模型（兼容）
    'SPDPretrainModel',
]
