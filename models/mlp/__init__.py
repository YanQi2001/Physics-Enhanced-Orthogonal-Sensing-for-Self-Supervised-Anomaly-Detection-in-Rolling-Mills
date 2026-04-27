"""
MLP 模块

用于 CSD 向量的特征学习，替代 SPDNet 流形方案。
"""

from .mlp_encoder import (
    MLPEncoder,
    MLPDecoder,
    MLPAutoEncoder,
    MLPEncoderWithProjection,
)

__all__ = [
    'MLPEncoder',
    'MLPDecoder',
    'MLPAutoEncoder',
    'MLPEncoderWithProjection',
]

