"""
双通道卷积编码器 (Dual-Channel Convolutional Encoder)

实现早期融合策略：
1. 第一层 Conv1d(in_channels=2) 直接接收 P1/P2 两路压力信号
2. 卷积核自动学习共模（总能级）和差模（双臂偏差）特征
3. 多层下采样提取时序抽象特征
"""

import torch
import torch.nn as nn
from typing import List, Optional


class DualChannelEncoder(nn.Module):
    """
    双通道早期融合编码器
    
    设计理念：
    - 第一层就融合 P1 和 P2，让网络学习它们的关系
    - 卷积核 w1*P1 + w2*P2：
      - 同号权重 → 共模特征（总压力能级）
      - 异号权重 → 差模特征（双臂偏载/不对称）
    
    Args:
        in_channels: 输入通道数（默认 2：P1, P2）
        hidden_channels: 隐藏层通道列表
        d_model: 输出特征维度
        kernel_size: 卷积核大小
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: List[int] = [32, 64, 128],
        d_model: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.d_model = d_model
        
        # 构建卷积层
        layers = []
        prev_ch = in_channels
        
        for i, ch in enumerate(hidden_channels):
            # 卷积 + 归一化 + 激活 + Dropout
            layers.append(
                nn.Conv1d(
                    prev_ch, ch,
                    kernel_size=kernel_size,
                    stride=2,  # 下采样
                    padding=kernel_size // 2
                )
            )
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.GELU())
            
            # 除了最后一层，都加 Dropout
            if i < len(hidden_channels) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_ch = ch
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 投影到 d_model（如果最后一层通道数不等于 d_model）
        if hidden_channels[-1] != d_model:
            self.projection = nn.Linear(hidden_channels[-1], d_model)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入信号 (B, C, T) - 已经是 (Batch, Channels, Time) 格式
            
        Returns:
            z: 编码特征 (B, d_model)
        """
        # 卷积编码
        z = self.conv_layers(x)  # (B, hidden_channels[-1], T')
        
        # 全局池化
        z = self.global_pool(z)  # (B, hidden_channels[-1], 1)
        z = z.squeeze(-1)  # (B, hidden_channels[-1])
        
        # 投影
        z = self.projection(z)  # (B, d_model)
        
        return z


class DualChannelDecoder(nn.Module):
    """
    双通道解码器（可选，用于重构监督）
    
    将压缩的特征向量重构回原始压力信号。
    
    Args:
        d_model: 输入特征维度
        hidden_channels: 隐藏层通道列表（从小到大）
        out_channels: 输出通道数（默认 2：P1, P2）
        seq_len: 输出序列长度
    """
    
    def __init__(
        self,
        d_model: int = 128,
        hidden_channels: List[int] = [128, 64, 32],
        out_channels: int = 2,
        seq_len: int = 1024
    ):
        super().__init__()
        
        self.d_model = d_model
        self.out_channels = out_channels
        self.seq_len = seq_len
        
        # 简化版：MLP 解码器
        # 直接映射到展平的输出
        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_channels[0] * 8),
            nn.GELU(),
            nn.Linear(hidden_channels[0] * 8, hidden_channels[1] * 16),
            nn.GELU(),
            nn.Linear(hidden_channels[1] * 16, seq_len * out_channels)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 特征向量 (B, d_model)
            
        Returns:
            x_recon: 重构信号 (B, T, C)
        """
        B = z.size(0)
        
        # 解码
        x_flat = self.decoder(z)  # (B, T * C)
        
        # 重塑为 (B, T, C)
        x_recon = x_flat.view(B, self.seq_len, self.out_channels)
        
        return x_recon


class ResidualBlock1D(nn.Module):
    """
    1D 残差块（可选，用于更深的编码器）
    """
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out

