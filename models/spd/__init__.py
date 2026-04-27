"""
SPDNet 流形神经网络模块

基于黎曼几何的对称正定（SPD）矩阵学习网络。
参考: Huang, Z., & Van Gool, L. J. (2017). A Riemannian Network for SPD Matrix Learning.
"""

from .spd_layers import (
    SPDTransform,
    SPDRectified,
    SPDTangentSpace,
    SPDVectorize,
    SPDUnTangentSpace,
    SPDIncreaseDim,
    StiefelParameter,
)
from .spd_encoder import SPDEncoder
from .spd_decoder import SPDDecoder
from .optimizer import StiefelMetaOptimizer

__all__ = [
    'SPDTransform',
    'SPDRectified', 
    'SPDTangentSpace',
    'SPDVectorize',
    'SPDUnTangentSpace',
    'SPDIncreaseDim',
    'StiefelParameter',
    'SPDEncoder',
    'SPDDecoder',
    'StiefelMetaOptimizer',
]

