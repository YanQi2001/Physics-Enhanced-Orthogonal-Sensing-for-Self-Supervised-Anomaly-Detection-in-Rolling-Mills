"""
VQ-VAE 模块

双通道 VQ-VAE 工况编码器，用于替代 Ti-MAE。

核心组件：
- VectorQuantizer: 向量量化层（STE + EMA）
- DualChannelEncoder: 双通道卷积编码器
- DualChannelVQVAE: 主模块（旧版，全局池化）
- VQVAEWithPhysicsLoss: 带物理约束的包装类（与 Ti-MAE 接口兼容）

v3.2 新增：
- TemporalVQVAE: 时序 VQ-VAE（保留时序结构，增强统计聚合）
- TemporalVQVAEWithPhysicsLoss: 带物理约束的时序 VQ-VAE 包装类
"""

from .quantizer import VectorQuantizer, VectorQuantizerEMA
from .encoder import DualChannelEncoder, DualChannelDecoder
from .vqvae import DualChannelVQVAE, VQVAEWithPhysicsLoss

# v3.2 时序 VQ-VAE
from .temporal_vqvae import (
    TemporalVQVAE,
    TemporalVQVAEWithPhysicsLoss,
    TemporalEncoder,
    TemporalDecoder,
    TemporalVectorQuantizer,
    apply_median_filter
)

__all__ = [
    # 旧版（全局池化）
    'VectorQuantizer',
    'VectorQuantizerEMA',
    'DualChannelEncoder',
    'DualChannelDecoder',
    'DualChannelVQVAE',
    'VQVAEWithPhysicsLoss',
    # v3.2 时序 VQ-VAE
    'TemporalVQVAE',
    'TemporalVQVAEWithPhysicsLoss',
    'TemporalEncoder',
    'TemporalDecoder',
    'TemporalVectorQuantizer',
    'apply_median_filter',
]

