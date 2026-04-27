"""
SPD 流形解码器

将切空间向量解码回 SPD 矩阵。
用于自编码器的重构任务。
"""

import torch
import torch.nn as nn
from typing import List

from .spd_layers import (
    SPDTransform,
    SPDRectified,
    SPDUnTangentSpace,
    SPDUnVectorize,
    SPDIncreaseDim,
)


class SPDDecoder(nn.Module):
    """
    SPD 流形解码器
    
    架构: UnVectorize → ExpEig → BiMap → ReEig → ... → BiMap → ReEig
    
    将切空间向量逐层升维并映射回SPD流形。
    
    Args:
        output_size: 输出矩阵维度（默认16）
        hidden_sizes: 中间层维度列表（与编码器相反）
        epsilon: ReEig层的特征值阈值
    """
    
    def __init__(
        self,
        output_size: int = 16,
        hidden_sizes: List[int] = [8, 12],
        epsilon: float = 1e-4
    ):
        super(SPDDecoder, self).__init__()
        
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        # 输入维度（从向量还原）
        input_size = hidden_sizes[0]
        self.input_dim = input_size * (input_size + 1) // 2
        
        # 构建层
        layers = []
        
        # UnVectorize + ExpEig（从切空间映射回流形）
        layers.append(SPDUnTangentSpace(unvectorize=True))
        
        prev_size = input_size
        for hidden_size in hidden_sizes[1:]:
            # BiMap升维
            layers.append(SPDTransform(prev_size, hidden_size))
            # ReEig确保正定
            layers.append(SPDRectified(epsilon=epsilon))
            prev_size = hidden_size
        
        # 最后一层升维到输出尺寸
        if prev_size != output_size:
            layers.append(SPDTransform(prev_size, output_size))
            layers.append(SPDRectified(epsilon=epsilon))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 切空间向量，shape (B, input_dim)
            
        Returns:
            重构的SPD矩阵，shape (B, output_size, output_size)
        """
        # 确保输入是float64
        x = z.double()
        
        for layer in self.layers:
            x = layer(x)
        
        return x


class SPDAutoEncoder(nn.Module):
    """
    SPD 自编码器
    
    组合编码器和解码器，用于流形特征学习。
    
    Args:
        input_size: 输入/输出矩阵维度
        encoder_hidden_sizes: 编码器中间层维度
        decoder_hidden_sizes: 解码器中间层维度（默认为编码器的逆序）
        epsilon: ReEig层阈值
    """
    
    def __init__(
        self,
        input_size: int = 16,
        encoder_hidden_sizes: List[int] = [12, 8],
        decoder_hidden_sizes: List[int] = None,
        epsilon: float = 1e-4
    ):
        super(SPDAutoEncoder, self).__init__()
        
        from .spd_encoder import SPDEncoder
        
        # 编码器
        self.encoder = SPDEncoder(
            input_size=input_size,
            hidden_sizes=encoder_hidden_sizes,
            epsilon=epsilon,
            vectorize=True
        )
        
        # 解码器（默认为编码器的镜像结构）
        if decoder_hidden_sizes is None:
            decoder_hidden_sizes = encoder_hidden_sizes[::-1]
        
        # 确保解码器从正确的维度开始
        decoder_hidden_sizes = [encoder_hidden_sizes[-1]] + decoder_hidden_sizes
        
        self.decoder = SPDDecoder(
            output_size=input_size,
            hidden_sizes=decoder_hidden_sizes,
            epsilon=epsilon
        )
        
        self.latent_dim = self.encoder.output_dim
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入SPD矩阵，shape (B, n, n)
            
        Returns:
            (重构矩阵, 潜在表示)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """仅解码"""
        return self.decoder(z)

