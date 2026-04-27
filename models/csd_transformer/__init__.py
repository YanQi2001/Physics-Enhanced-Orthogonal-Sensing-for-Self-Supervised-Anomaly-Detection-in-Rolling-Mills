"""
CSD Pair-Token Transformer 模块

将 16×16 CSD 矩阵的上三角元素视为 136 个 Token，
使用 Transformer 学习传感器间的耦合关系。
"""

from .csd_encoder import (
    CSDTransformerEncoder,
    PairPositionEncoding,
    get_block_type
)

__all__ = [
    'CSDTransformerEncoder',
    'PairPositionEncoding',
    'get_block_type'
]


