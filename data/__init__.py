"""
数据处理模块
- 全通道自适应虚拟扩展（4→16通道）
- 16×16 CSD矩阵构建
- PyTorch Dataset
- 数据增强（正交性扰动等）
"""

from .preprocessing import VirtualChannelExpander, CSDMatrixBuilder
from .dataset import (
    MultiModalDataset,
    SequentialMultiModalDataset,
    PrecomputedMultiModalDataset,
    ShaoGangDataset,
    create_dataloaders,
    create_dataloaders_from_preprocessed,
    create_shaogang_dataloaders,
)
from .augmentation import (
    OrthogonalityPerturbation,
    TimeWarpingAugmentation,
    GaussianNoiseAugmentation,
    CompositeAugmentation,
)
from .shaogang_loader import ShaoGangDataLoader, load_shaogang_csv

__all__ = [
    'VirtualChannelExpander',
    'CSDMatrixBuilder', 
    'MultiModalDataset',
    'SequentialMultiModalDataset',
    'PrecomputedMultiModalDataset',
    'ShaoGangDataset',
    'create_dataloaders',
    'create_dataloaders_from_preprocessed',
    'create_shaogang_dataloaders',
    'OrthogonalityPerturbation',
    'TimeWarpingAugmentation',
    'GaussianNoiseAugmentation',
    'CompositeAugmentation',
    'ShaoGangDataLoader',
    'load_shaogang_csv',
]

