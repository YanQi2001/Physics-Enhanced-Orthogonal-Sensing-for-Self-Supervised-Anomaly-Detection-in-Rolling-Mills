"""
SPD 流形编码器

将 16×16 的 SPD 矩阵编码到低维切空间向量。
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .spd_layers import (
    SPDTransform,
    SPDRectified,
    SPDTangentSpace,
    SPDVectorize,
)


class SPDEncoder(nn.Module):
    """
    SPD 流形编码器
    
    架构: BiMap → ReEig → BiMap → ReEig → ... → LogEig → Vectorize
    
    将 SPD 矩阵逐层降维并映射到切空间。
    
    Args:
        input_size: 输入矩阵维度（默认16，对应16×16 CSD矩阵）
        hidden_sizes: 中间层维度列表
        epsilon: ReEig层的特征值阈值
        vectorize: 是否输出向量化结果
    """
    
    def __init__(
        self,
        input_size: int = 16,
        hidden_sizes: List[int] = [12, 8],
        epsilon: float = 1e-4,
        vectorize: bool = True
    ):
        super(SPDEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.vectorize = vectorize
        
        # 构建层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # BiMap层
            layers.append(SPDTransform(prev_size, hidden_size))
            # ReEig层
            layers.append(SPDRectified(epsilon=epsilon))
            prev_size = hidden_size
        
        # LogEig层（映射到切空间）
        layers.append(SPDTangentSpace(prev_size, vectorize=vectorize))
        
        self.layers = nn.ModuleList(layers)
        self.output_size = prev_size
        
        # 计算输出向量维度
        if vectorize:
            self.output_dim = prev_size * (prev_size + 1) // 2
        else:
            self.output_dim = prev_size * prev_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入SPD矩阵，shape (B, n, n)
            
        Returns:
            切空间特征，shape (B, output_dim) 如果 vectorize=True
            否则 shape (B, output_size, output_size)
        """
        # 确保输入是float64（SPDNet需要高精度）
        x = x.double()
        
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def get_intermediate_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        获取中间层特征（用于可视化/分析）
        
        Args:
            x: 输入SPD矩阵
            
        Returns:
            每层输出的列表
        """
        x = x.double()
        features = [x]
        
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        
        return features


class SPDEncoderWithProjection(nn.Module):
    """
    带额外投影层的 SPD 编码器
    
    在 SPDEncoder 输出后增加可学习的线性投影层，
    用于调整输出维度以匹配下游任务。
    
    Args:
        input_size: 输入矩阵维度
        hidden_sizes: SPD编码器中间层维度
        projection_dim: 最终投影维度
        epsilon: ReEig层阈值
    """
    
    def __init__(
        self,
        input_size: int = 16,
        hidden_sizes: List[int] = [12, 8],
        projection_dim: int = 128,
        epsilon: float = 1e-4
    ):
        super(SPDEncoderWithProjection, self).__init__()
        
        self.spd_encoder = SPDEncoder(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            epsilon=epsilon,
            vectorize=True
        )
        
        self.projection = nn.Sequential(
            nn.Linear(self.spd_encoder.output_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
        )
        
        self.output_dim = projection_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入SPD矩阵，shape (B, n, n)
            
        Returns:
            投影后的特征，shape (B, projection_dim)
        """
        z = self.spd_encoder(x)
        z = z.float()  # 投影层使用float32
        z = self.projection(z)
        return z
    
    def get_spd_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取SPD编码器的输出（不经过投影）
        """
        return self.spd_encoder(x)

