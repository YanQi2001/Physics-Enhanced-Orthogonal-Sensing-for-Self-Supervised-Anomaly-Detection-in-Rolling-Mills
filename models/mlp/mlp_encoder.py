"""
MLP 自编码器模块

用于 CSD 向量的特征学习，替代 SPDNet 流形方案。
输入: 272 维 CSD 向量
输出: 低维潜在表示 + 重构向量
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MLPEncoder(nn.Module):
    """
    MLP 编码器
    
    将 CSD 向量编码为低维潜在表示。
    
    Args:
        input_dim: 输入维度 (默认 272 = 136 实部 + 136 虚部)
        hidden_dims: 隐藏层维度列表
        latent_dim: 潜在空间维度
        dropout: Dropout 比率
        use_batch_norm: 是否使用 BatchNorm
    """
    
    def __init__(
        self,
        input_dim: int = 272,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 64,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super(MLPEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 最后一层到潜在空间
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入 CSD 向量，shape (B, input_dim)
            
        Returns:
            潜在表示，shape (B, latent_dim)
        """
        return self.encoder(x)


class MLPDecoder(nn.Module):
    """
    MLP 解码器
    
    将潜在表示解码回 CSD 向量。
    
    Args:
        latent_dim: 潜在空间维度
        hidden_dims: 隐藏层维度列表 (与编码器相反)
        output_dim: 输出维度
        dropout: Dropout 比率
        use_batch_norm: 是否使用 BatchNorm
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: List[int] = [128, 256],
        output_dim: int = 272,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super(MLPDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 最后一层到输出空间
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 潜在表示，shape (B, latent_dim)
            
        Returns:
            重构向量，shape (B, output_dim)
        """
        return self.decoder(z)


class MLPAutoEncoder(nn.Module):
    """
    MLP 自编码器
    
    组合编码器和解码器，用于 CSD 向量的特征学习和重构。
    
    Args:
        input_dim: 输入/输出维度 (默认 272)
        hidden_dims: 编码器隐藏层维度
        latent_dim: 潜在空间维度
        dropout: Dropout 比率
        use_batch_norm: 是否使用 BatchNorm
    """
    
    def __init__(
        self,
        input_dim: int = 272,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 64,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super(MLPAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
        # 解码器 (镜像结构)
        decoder_hidden_dims = hidden_dims[::-1]
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            x: 输入 CSD 向量，shape (B, input_dim)
            
        Returns:
            (重构向量, 潜在表示)
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


class MLPEncoderWithProjection(nn.Module):
    """
    带投影层的 MLP 编码器
    
    在编码器输出后增加可学习的投影层，
    用于调整输出维度以匹配下游任务（如融合模块）。
    
    Args:
        input_dim: 输入维度
        hidden_dims: 编码器隐藏层维度
        latent_dim: 编码器潜在空间维度
        projection_dim: 最终投影维度
        dropout: Dropout 比率
    """
    
    def __init__(
        self,
        input_dim: int = 272,
        hidden_dims: List[int] = [256, 128],
        latent_dim: int = 64,
        projection_dim: int = 128,
        dropout: float = 0.1
    ):
        super(MLPEncoderWithProjection, self).__init__()
        
        self.mlp_encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
        )
        
        self.output_dim = projection_dim
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入 CSD 向量，shape (B, input_dim)
            
        Returns:
            投影后的特征，shape (B, projection_dim)
        """
        z = self.mlp_encoder(x)
        z = self.projection(z)
        return z
    
    def get_latent_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取编码器的输出（不经过投影）
        """
        return self.mlp_encoder(x)

