"""
双通道 VQ-VAE 工况编码器

替代 Ti-MAE，实现离散化的工况状态编码。

核心设计：
1. 早期卷积融合：第一层 Conv1d(in=2) 学习 P1/P2 的共模+差模
2. VQ 离散化：强制工况归类为 K 种原型状态
3. 统计量注入：保留 [μ1, σ1, μ2, σ2] 防止绝对能级丢失

接口与 Ti-MAE 完全兼容：
- forward(): 训练时返回损失字典
- get_context(): 推理时返回 Q_context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .quantizer import VectorQuantizer
from .encoder import DualChannelEncoder, DualChannelDecoder


class DualChannelVQVAE(nn.Module):
    """
    双通道 VQ-VAE 工况编码器
    
    设计亮点：
    1. 早期卷积融合：第一层 Conv1d(in=2) 学习 P1/P2 的共模+差模特征
    2. VQ 离散化：强制将连续工况特征离散化为 K 个原型状态
    3. 统计量注入：保留 [μ1, σ1, μ2, σ2] 防止绝对能级信息丢失
    
    物理语义：
    - Codebook 直接对应研究报告 5.1 节的"参考状态嵌入 R"
    - 每个码本向量代表一种典型工况：低压稳态、高压稳态、上升过渡、下降过渡等
    - 统计量注入使得模型既知道"当前是工况A"，又知道"左臂150bar、右臂50bar"
    
    接口设计（与 Ti-MAE 兼容）：
    - forward(x) → dict: 训练时返回损失
    - get_context(x) → Tensor: 推理时返回 Q_context
    
    Args:
        seq_len: 输入序列长度
        in_channels: 输入通道数（默认 2：P1, P2）
        d_model: 输出特征维度（与融合层兼容）
        n_embeddings: 码本大小（工况类别数，建议 8-16）
        encoder_channels: 编码器各层通道数
        commitment_cost: VQ Commitment Loss 权重
        decay: EMA 衰减率
        use_decoder: 是否使用解码器进行重构监督
        lambda_recon: 重构损失权重
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        seq_len: int = 1024,
        in_channels: int = 2,
        d_model: int = 128,
        n_embeddings: int = 16,
        encoder_channels: List[int] = [32, 64, 128],
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        use_decoder: bool = True,
        lambda_recon: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_embeddings = n_embeddings
        self.use_decoder = use_decoder
        self.lambda_recon = lambda_recon
        
        # 初始化 BatchNorm
        self.batch_norm = nn.BatchNorm1d(in_channels)
        
        # ==================== 编码器 ====================
        self.encoder = DualChannelEncoder(
            in_channels=in_channels,
            hidden_channels=encoder_channels,
            d_model=d_model,
            dropout=dropout
        )
        
        # ==================== VQ 层 ====================
        self.quantizer = VectorQuantizer(
            n_embeddings=n_embeddings,
            embedding_dim=d_model,
            commitment_cost=commitment_cost,
            decay=decay
        )
        
        # ==================== 统计量融合层 ====================
        # 输入: d_model (VQ特征) + 4 (P1/P2 的 μ 和 σ)
        # 输出: d_model（保持与 Ti-MAE 输出维度一致）
        self.stat_fusion = nn.Sequential(
            nn.Linear(d_model + 2 * in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # ==================== 解码器（可选）====================
        if use_decoder:
            self.decoder = DualChannelDecoder(
                d_model=d_model,
                hidden_channels=encoder_channels[::-1],
                out_channels=in_channels,
                seq_len=seq_len
            )
        else:
            self.decoder = None
        
        # 输出维度（与 Ti-MAE 兼容）
        self.output_dim = d_model
    
    def _compute_stats(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输入信号的统计量
        
        Args:
            x: 输入信号 (B, T, C)
            
        Returns:
            stats: [μ1, μ2, σ1, σ2] (B, 2*C)
        """
        # 计算每个通道的均值和标准差
        mean = x.mean(dim=1)  # (B, C)
        std = x.std(dim=1) + 1e-5  # (B, C)
        
        # 拼接: [μ1, μ2, σ1, σ2]
        stats = torch.cat([mean, std], dim=-1)  # (B, 2*C)
        
        return stats
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        [Deprecated] 旧版归一化方法，已废弃
        """
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        return_losses: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 压力信号 (B, T, C) 或 (B, T)
            return_losses: 是否返回损失
            
        Returns:
            dict: {
                # 与 Ti-MAE 兼容的键
                'recon': 重构信号 (B, T, C)，
                'mask': None（VQ-VAE 没有 mask）,
                'latent': 潜在表示 (B, num_patches, d_model)（模拟）,
                'l_recon': 重构损失,
                'l_smooth': VQ 损失（映射为 smooth，语义相近）,
                'l_total': 总损失,
                
                # VQ-VAE 特有的键
                'q_context': 工况上下文向量 (B, d_model),
                'indices': 码本索引 (B,),
                'stats': 原始统计量 (B, 2*C),
                'vq_loss': VQ 损失,
            }
        """
        B = x.size(0)
        
        # 处理输入维度
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, T) -> (B, T, 1)
        
        # 保存原始输入用于重构损失
        x_original = x
        
        # --- 1. 计算并保存原始统计量 ---
        stats = self._compute_stats(x)  # (B, 2*C)
        
        # --- 2. 归一化 (使用 BatchNorm1d 替代 RevIN，防止 magnitude 丢失) ---
        # x: (B, T, C) -> (B, C, T)
        x_norm = x.transpose(1, 2)
        
        # 动态创建 BatchNorm (如果未初始化)
        if not hasattr(self, 'batch_norm'):
            self.batch_norm = nn.BatchNorm1d(self.in_channels).to(x.device)
            
        x_norm = self.batch_norm(x_norm)
        # x_norm: (B, C, T)
        
        # --- 3. 编码（早期融合）---
        # x_norm 已经是 (B, C, T)，直接输入 Encoder
        z_e = self.encoder(x_norm)  # (B, d_model)
        
        # --- 4. VQ 离散化 ---
        z_q, vq_loss, indices = self.quantizer(z_e)
        
        # --- 5. 统计量注入 ---
        # 拼接: [离散工况语义] + [连续物理能级]
        z_combined = torch.cat([z_q, stats], dim=-1)  # (B, d_model + 2*C)
        q_context = self.stat_fusion(z_combined)  # (B, d_model)
        
        # --- 6. 构建输出字典 ---
        result = {
            # VQ-VAE 特有
            'q_context': q_context,
            'indices': indices,
            'stats': stats,
            'z_q': z_q,
            'z_e': z_e,
            'vq_loss': vq_loss,
            
            # 与 Ti-MAE 兼容的占位符
            'mask': None,
        }
        
        # --- 7. 可选：重构损失 ---
        if return_losses:
            if self.use_decoder and self.decoder is not None:
                x_recon = self.decoder(z_q)  # (B, T, C)
                # 使用归一化后的数据计算重构损失
                x_target = x_norm.transpose(1, 2)  # (B, C, T) -> (B, T, C)
                recon_loss = F.mse_loss(x_recon, x_target)
                result['recon'] = x_recon
                result['l_recon'] = recon_loss
            else:
                result['recon'] = x  # 占位
                result['l_recon'] = torch.tensor(0.0, device=x.device)
            
            # 映射 VQ Loss 为 l_smooth（语义相近：都是正则化约束）
            result['l_smooth'] = vq_loss
            
            # 总损失
            l_total = vq_loss
            if self.use_decoder:
                l_total = l_total + self.lambda_recon * result['l_recon']
            result['l_total'] = l_total
            
            # 模拟 latent 序列（用于兼容）
            # Ti-MAE 返回 (B, num_patches, d_model)，这里用单个向量扩展
            result['latent'] = q_context.unsqueeze(1)  # (B, 1, d_model)
        
        return result
    
    def get_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取工况上下文向量（推理模式）
        
        与 Ti-MAE 的 get_context() 接口完全一致。
        
        Args:
            x: 压力信号 (B, T, C)
            
        Returns:
            q_context: 工况上下文向量 (B, d_model)
        """
        with torch.no_grad():
            result = self.forward(x, return_losses=False)
            return result['q_context']
    
    def get_latent_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取潜在序列（与 Ti-MAE 接口兼容）
        
        注意：VQ-VAE 是全局编码，没有序列结构。
        这里返回单个向量的扩展版本以保持接口兼容。
        
        Args:
            x: 压力信号
            
        Returns:
            latent: (B, 1, d_model)
        """
        with torch.no_grad():
            result = self.forward(x, return_losses=False)
            return result['q_context'].unsqueeze(1)
    
    def get_codebook_usage(self) -> torch.Tensor:
        """获取码本使用率"""
        return self.quantizer.get_codebook_usage()
    
    def get_active_codes(self, threshold: float = 0.01) -> int:
        """获取活跃码本数量"""
        return self.quantizer.get_active_codes(threshold)
    
    def get_codebook_vectors(self) -> torch.Tensor:
        """
        获取码本向量（用于可视化）
        
        Returns:
            codebook: (K, d_model)
        """
        return self.quantizer.embedding.weight.data.clone()


class VQVAEWithPhysicsLoss(nn.Module):
    """
    带物理约束损失的 VQ-VAE（与 TiMAEWithPhysicsLoss 接口兼容）
    
    这是 DualChannelVQVAE 的包装类，提供与 TiMAEWithPhysicsLoss 完全一致的接口。
    """
    
    def __init__(
        self,
        seq_len: int = 1024,
        in_channels: int = 2,
        d_model: int = 128,
        n_embeddings: int = 16,
        encoder_channels: List[int] = [32, 64, 128],
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        lambda_recon: float = 0.1,
        # 以下参数是为了兼容 Ti-MAE 接口而存在，实际不使用
        patch_size: int = 16,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        point_ratio: float = 0.5,
        block_ratio: float = 0.2,
        block_size: int = 4,
        lambda_smooth: float = 5.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vqvae = DualChannelVQVAE(
            seq_len=seq_len,
            in_channels=in_channels,
            d_model=d_model,
            n_embeddings=n_embeddings,
            encoder_channels=encoder_channels,
            commitment_cost=commitment_cost,
            decay=decay,
            use_decoder=True,
            lambda_recon=lambda_recon,
            dropout=dropout
        )
        
        # 保存 lambda_smooth 以模拟 Ti-MAE 的接口
        # 在 VQ-VAE 中，这个值不起作用（VQ Loss 替代了 Smooth Loss）
        self.lambda_smooth = lambda_smooth
        self.output_dim = d_model
    
    def forward(
        self,
        x: torch.Tensor,
        return_losses: bool = True
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return self.vqvae(x, return_losses=return_losses)
    
    def get_context(self, x: torch.Tensor) -> torch.Tensor:
        """获取工况上下文"""
        return self.vqvae.get_context(x)
    
    def get_latent_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """获取潜在序列"""
        return self.vqvae.get_latent_sequence(x)
    
    def get_codebook_usage(self) -> torch.Tensor:
        """获取码本使用率"""
        return self.vqvae.get_codebook_usage()

