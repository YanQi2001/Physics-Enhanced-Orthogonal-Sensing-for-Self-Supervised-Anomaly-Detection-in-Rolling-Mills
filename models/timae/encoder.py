"""
Ti-MAE 编码器组件

包含 Patch Embedding、位置编码和 Transformer Encoder。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PatchEmbedding(nn.Module):
    """
    Patch 嵌入层
    
    将输入信号分割成固定大小的 patch 并嵌入到高维空间。
    
    Args:
        patch_size: 每个 patch 的长度
        d_model: 嵌入维度
        in_channels: 输入通道数
        bias: 是否使用偏置
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        d_model: int = 128,
        in_channels: int = 1,
        bias: bool = True
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.in_channels = in_channels
        
        # 使用1D卷积实现patch embedding
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入信号，shape (B, T, C) 或 (B, T)
            
        Returns:
            Patch 嵌入，shape (B, num_patches, d_model)
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, T, 1)
        
        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        
        # Conv1d: (B, C, T) -> (B, d_model, num_patches)
        x = self.proj(x)
        
        # (B, d_model, num_patches) -> (B, num_patches, d_model)
        x = x.transpose(1, 2)
        
        # Layer norm
        x = self.norm(x)
        
        return x
    
    def get_num_patches(self, seq_len: int) -> int:
        """计算给定序列长度的patch数量"""
        return seq_len // self.patch_size


class PositionalEmbedding(nn.Module):
    """
    正弦位置编码
    
    使用正弦/余弦函数生成位置编码。
    
    Args:
        d_model: 嵌入维度
        max_len: 最大序列长度
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEmbedding, self).__init__()
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取位置编码
        
        Args:
            x: 输入张量，shape (B, L, d_model)
            
        Returns:
            位置编码，shape (1, L, d_model)
        """
        return self.pe[:, :x.size(1)]


class LearnablePositionalEmbedding(nn.Module):
    """
    可学习位置编码
    
    Args:
        d_model: 嵌入维度
        max_len: 最大序列长度
    """
    
    def __init__(self, d_model: int, max_len: int = 512):
        super(LearnablePositionalEmbedding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取位置编码
        
        Args:
            x: 输入张量，shape (B, L, d_model)
            
        Returns:
            位置编码，shape (1, L, d_model)
        """
        return self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层
    
    包含多头自注意力和前馈网络。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff: 前馈网络隐藏维度
        dropout: Dropout 比例
        activation: 激活函数
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super(TransformerEncoderLayer, self).__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入，shape (B, L, d_model)
            attn_mask: 注意力掩码
            key_padding_mask: 键填充掩码
            
        Returns:
            输出，shape (B, L, d_model)
        """
        # 自注意力 + 残差
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈网络 + 残差
        x = x + self.ff(x)
        x = self.norm2(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    
    堆叠多个 TransformerEncoderLayer。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: 编码器层数
        d_ff: 前馈网络隐藏维度
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入，shape (B, L, d_model)
            attn_mask: 注意力掩码
            key_padding_mask: 键填充掩码
            
        Returns:
            输出，shape (B, L, d_model)
        """
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)
        
        x = self.norm(x)
        return x
    
    def get_intermediate_outputs(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ):
        """
        获取每层的输出
        
        Args:
            x: 输入
            attn_mask: 注意力掩码
            
        Returns:
            每层输出的列表
        """
        outputs = [x]
        for layer in self.layers:
            x = layer(x, attn_mask)
            outputs.append(x)
        return outputs

