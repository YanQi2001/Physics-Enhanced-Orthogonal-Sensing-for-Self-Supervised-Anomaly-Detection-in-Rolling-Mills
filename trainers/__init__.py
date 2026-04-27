"""
训练器模块

v3.2 更新：新增 VQ-VAE 工况预训练器

三阶段训练流程：
1. CSD Transformer 预训练（对比学习）- 新方案
   或 SPD 流形预训练（黎曼几何）- 旧方案
2. VQ-VAE 工况预训练（离散状态编码）- 新方案
   或 Ti-MAE 工况预训练（掩码重构）- 旧方案
3. 条件联合微调
"""

from .csd_pretrainer import CSDPretrainer  # 新方案
from .spd_pretrainer import SPDPretrainer  # 旧方案（保留以兼容）
from .vqvae_pretrainer import VQVAEPretrainer  # VQ-VAE 新方案
from .timae_pretrainer import TiMAEPretrainer  # Ti-MAE 旧方案
from .joint_trainer import JointTrainer
from .utils import (
    EarlyStopping,
    LRScheduler,
    TrainingLogger,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # 阶段一预训练
    'CSDPretrainer',   # 新方案
    'SPDPretrainer',   # 旧方案
    # 阶段 1.5 预训练
    'VQVAEPretrainer',  # VQ-VAE 新方案
    'TiMAEPretrainer',  # Ti-MAE 旧方案
    # 阶段二联合训练
    'JointTrainer',
    # 工具
    'EarlyStopping',
    'LRScheduler',
    'TrainingLogger',
    'save_checkpoint',
    'load_checkpoint',
]
