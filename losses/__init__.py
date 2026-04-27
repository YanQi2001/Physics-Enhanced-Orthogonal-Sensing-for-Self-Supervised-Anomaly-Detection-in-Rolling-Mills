"""
损失函数模块

包含：
- 黎曼距离损失（LEM, AIRM）
- 物理约束损失（平滑度）
- 重构损失
"""

from .riemannian_loss import (
    LogEuclideanDistance,
    AffineInvariantRiemannianMetric,
    SPDFrobeniusDistance,
)
from .physics_loss import SmoothnessLoss, ContrastiveLoss, ConsistencyLoss, SynergyLoss, ManifoldCompactnessLoss
from .reconstruction_loss import MaskedReconstructionLoss

__all__ = [
    'LogEuclideanDistance',
    'AffineInvariantRiemannianMetric',
    'SPDFrobeniusDistance',
    'SmoothnessLoss',
    'ContrastiveLoss',
    'ConsistencyLoss',
    'SynergyLoss',
    'ManifoldCompactnessLoss',
    'MaskedReconstructionLoss',
]

