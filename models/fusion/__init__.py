"""
跨模态融合模块

包含：
- ContextConditionedAttention: 条件注意力融合
- RiemannianSynergyModule: 黎曼协同感知
- LossGatingNetwork: 物理上下文门控
"""

from .context_attention import ContextConditionedAttention
from .synergy_module import RiemannianSynergyModule
from .gating import LossGatingNetwork

__all__ = [
    'ContextConditionedAttention',
    'RiemannianSynergyModule',
    'LossGatingNetwork',
]

