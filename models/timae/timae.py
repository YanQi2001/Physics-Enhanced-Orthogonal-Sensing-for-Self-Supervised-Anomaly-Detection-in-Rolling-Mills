"""
多尺度物理感知 Ti-MAE

将压力信号转化为干净的阶梯状工况特征，作为条件引导振动分支的分析。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .revin import RevIN
from .masking import MultiScaleMasker
from .encoder import (
    PatchEmbedding,
    PositionalEmbedding,
    LearnablePositionalEmbedding,
    TransformerEncoder,
)


class PhysicsAwareTiMAE(nn.Module):
    """
    多尺度物理感知 Ti-MAE（Time-series Masked AutoEncoder）
    
    核心功能：
    1. 多尺度遮蔽：点状（去噪）+ 块状（结构推理）
    2. 物理约束：平滑度损失强迫 Latent 变成阶梯状
    3. 工况提取：输出干净的工况上下文向量 Q_context
    
    Args:
        seq_len: 输入序列长度
        in_channels: 输入通道数
        patch_size: Patch 大小
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: Encoder 层数
        d_ff: 前馈网络维度
        dropout: Dropout 比例
        point_ratio: 点状遮蔽比例
        block_ratio: 块状遮蔽比例
        block_size: 块大小
        use_revin: 是否使用 RevIN
        learnable_pos: 是否使用可学习位置编码
    """
    
    def __init__(
        self,
        seq_len: int = 1024,
        in_channels: int = 2,  # 2个压力通道 P1, P2
        patch_size: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        point_ratio: float = 0.5,
        block_ratio: float = 0.2,
        block_size: int = 4,
        use_revin: bool = True,
        learnable_pos: bool = False
    ):
        super(PhysicsAwareTiMAE, self).__init__()
        
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = seq_len // patch_size
        
        # RevIN 归一化
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=in_channels)
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            in_channels=in_channels
        )
        
        # 位置编码
        if learnable_pos:
            self.pos_embed = LearnablePositionalEmbedding(d_model, max_len=self.num_patches)
        else:
            self.pos_embed = PositionalEmbedding(d_model, max_len=self.num_patches)
        
        # 可学习的 [MASK] token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 多尺度遮蔽器
        self.masker = MultiScaleMasker(
            point_ratio=point_ratio,
            block_ratio=block_ratio,
            block_size=block_size
        )
        
        # Transformer Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # 解码器（简单线性层）
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, patch_size * in_channels)
        )
        
        # 输出维度
        self.output_dim = d_model
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（训练模式）
        
        Args:
            x: 输入压力信号，shape (B, T, C) 或 (B, T)
            mask: 可选的外部遮蔽，shape (B, num_patches)
            return_latent: 是否返回潜在表示
            
        Returns:
            dict containing:
                - 'recon': 重构信号，shape (B, T, C)
                - 'mask': 使用的遮蔽，shape (B, num_patches)
                - 'latent': 潜在表示（如果 return_latent=True）
        """
        batch_size = x.size(0)
        
        # 处理输入维度
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, T, 1)
        
        # RevIN 归一化
        if self.use_revin:
            x = self.revin(x, 'norm')
        
        # Patch embedding
        x_patches = self.patch_embed(x)  # (B, num_patches, d_model)
        
        # 添加位置编码
        pos_embed = self.pos_embed(x_patches)
        x_patches = x_patches + pos_embed
        
        # 生成遮蔽（如果没有提供）
        if mask is None:
            mask = self.masker(self.num_patches, batch_size, x.device)
        
        # 用 [MASK] token 替换被遮蔽的 patches
        mask_tokens = self.mask_token.expand(batch_size, self.num_patches, -1)
        x_masked = torch.where(
            mask.unsqueeze(-1).expand_as(x_patches),
            mask_tokens,
            x_patches
        )
        
        # Transformer 编码
        latent = self.encoder(x_masked)  # (B, num_patches, d_model)
        
        # 解码（只解码被遮蔽的部分用于计算损失）
        decoded = self.decoder(latent)  # (B, num_patches, patch_size * in_channels)
        
        # 重组为原始形状
        recon = decoded.view(batch_size, -1, self.in_channels)  # (B, T, C)
        
        # RevIN 反归一化
        if self.use_revin:
            recon = self.revin(recon, 'denorm')
        
        result = {
            'recon': recon,
            'mask': mask,
        }
        
        if return_latent:
            result['latent'] = latent
        
        return result
    
    def get_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取工况上下文向量（推理模式，不遮蔽）
        
        Args:
            x: 输入压力信号，shape (B, T, C) 或 (B, T)
            
        Returns:
            工况上下文向量 Q_context，shape (B, d_model)
        """
        # 处理输入维度
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        
        # RevIN 归一化
        if self.use_revin:
            x = self.revin(x, 'norm')
        
        # Patch embedding + 位置编码
        x_patches = self.patch_embed(x)
        pos_embed = self.pos_embed(x_patches)
        x_patches = x_patches + pos_embed
        
        # Transformer 编码（不遮蔽）
        latent = self.encoder(x_patches)  # (B, num_patches, d_model)
        
        # 全局平均池化
        q_context = latent.mean(dim=1)  # (B, d_model)
        
        return q_context
    
    def get_latent_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取完整的潜在序列（用于平滑度约束）
        
        Args:
            x: 输入压力信号
            
        Returns:
            潜在序列，shape (B, num_patches, d_model)
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        
        if self.use_revin:
            x = self.revin(x, 'norm')
        
        x_patches = self.patch_embed(x)
        pos_embed = self.pos_embed(x_patches)
        x_patches = x_patches + pos_embed
        
        latent = self.encoder(x_patches)
        
        return latent


class TiMAEWithPhysicsLoss(nn.Module):
    """
    带物理约束损失的 Ti-MAE
    
    集成了重构损失和平滑度约束。
    
    Args:
        timae_config: Ti-MAE 配置参数
        lambda_smooth: 平滑度约束权重
    """
    
    def __init__(
        self,
        seq_len: int = 1024,
        in_channels: int = 2,
        patch_size: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        point_ratio: float = 0.5,
        block_ratio: float = 0.2,
        block_size: int = 4,
        lambda_smooth: float = 5.0
    ):
        super(TiMAEWithPhysicsLoss, self).__init__()
        
        self.timae = PhysicsAwareTiMAE(
            seq_len=seq_len,
            in_channels=in_channels,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            point_ratio=point_ratio,
            block_ratio=block_ratio,
            block_size=block_size
        )
        
        self.lambda_smooth = lambda_smooth
        self.patch_size = patch_size
        self.in_channels = in_channels
    
    def forward(
        self,
        x: torch.Tensor,
        return_losses: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入信号
            return_losses: 是否返回各项损失
            
        Returns:
            dict with 'recon', 'mask', 'latent', and optionally losses
        """
        result = self.timae(x, return_latent=True)
        
        if return_losses:
            # 重构损失（只计算被遮蔽部分）
            recon = result['recon']
            mask = result['mask']
            
            # 将 mask 扩展到原始时间维度
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
            mask_expanded = mask_expanded.view(mask.size(0), -1)  # (B, T)
            mask_expanded = mask_expanded.unsqueeze(-1).expand_as(recon)  # (B, T, C)
            
            # 处理输入维度
            target = x
            if target.ndim == 2:
                target = target.unsqueeze(-1)
            
            # 只计算被遮蔽部分的 MSE
            masked_recon = recon[mask_expanded]
            masked_target = target[mask_expanded]
            
            if len(masked_recon) > 0:
                l_recon = F.mse_loss(masked_recon, masked_target)
            else:
                l_recon = torch.tensor(0.0, device=x.device)
            
            # 平滑度约束
            latent = result['latent']  # (B, num_patches, d_model)
            latent_diff = latent[:, 1:] - latent[:, :-1]  # (B, num_patches-1, d_model)
            l_smooth = torch.norm(latent_diff, dim=-1).mean()
            
            # 总损失
            l_total = l_recon + self.lambda_smooth * l_smooth
            
            result['l_recon'] = l_recon
            result['l_smooth'] = l_smooth
            result['l_total'] = l_total
        
        return result
    
    def get_context(self, x: torch.Tensor) -> torch.Tensor:
        """获取工况上下文"""
        return self.timae.get_context(x)

