"""
向量量化层 (Vector Quantizer)

实现 VQ-VAE 的核心组件：
1. EMA 更新码本（比梯度更新更稳定）
2. Straight-Through Estimator（STE）解决梯度断裂
3. Commitment Loss 约束编码器输出接近码本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """
    向量量化层
    
    核心技术：
    1. EMA 更新码本（比梯度更新更稳定，防止码本坍塌）
    2. Straight-Through Estimator（STE）解决 argmin 不可导问题
    3. Commitment Loss 约束编码器输出接近码本向量
    
    物理意义：
    - n_embeddings 个码本向量代表 K 种典型工况状态
    - 每个输入被强制映射到最近的工况原型
    - 实现从连续特征到离散状态的"硬着陆"
    
    Args:
        n_embeddings: 码本大小（工况类别数，建议 8-16）
        embedding_dim: 码本向量维度
        commitment_cost: Commitment Loss 权重（默认 0.25）
        decay: EMA 衰减率（默认 0.99）
        epsilon: 数值稳定性常数
    """
    
    def __init__(
        self,
        n_embeddings: int = 16,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # 码本嵌入
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        # 初始化：均匀分布在 [-1/K, 1/K]
        self.embedding.weight.data.uniform_(-1.0 / n_embeddings, 1.0 / n_embeddings)
        
        # EMA 更新所需的缓冲区（不参与梯度计算）
        self.register_buffer('ema_cluster_size', torch.zeros(n_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        
        # 跟踪码本使用情况
        self.register_buffer('usage_count', torch.zeros(n_embeddings))
    
    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            z_e: 编码器输出 (B, D)
            
        Returns:
            z_q: 量化后的向量 (B, D)
            loss: VQ Loss + Commitment Loss
            encoding_indices: 码本索引 (B,)
        """
        # 确保输入是 2D
        input_shape = z_e.shape
        if z_e.ndim > 2:
            z_e = z_e.view(-1, self.embedding_dim)
        
        # 1. 计算到所有码本向量的 L2 距离
        # distances[i,j] = ||z_e[i] - embedding[j]||^2
        # 展开: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_e @ self.embedding.weight.T
        )  # (B, K)
        
        # 2. 找最近的码本向量
        encoding_indices = torch.argmin(distances, dim=1)  # (B,)
        z_q = self.embedding(encoding_indices)  # (B, D)
        
        # 3. 计算损失
        # VQ Loss: 让码本向量移动到编码器输出（通过 EMA 实现）
        # Commitment Loss: 让编码器输出接近码本向量（通过梯度实现）
        loss_vq = F.mse_loss(z_q, z_e.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_e)
        loss = loss_vq + self.commitment_cost * loss_commit
        
        # 4. EMA 更新码本（仅训练时）
        if self.training:
            self._ema_update(z_e, encoding_indices)
        
        # 5. Straight-Through Estimator (STE)
        # 前向传播：使用 z_q（离散的量化向量）
        # 反向传播：梯度直接传给 z_e（连续的编码器输出）
        z_q = z_e + (z_q - z_e).detach()
        
        # 恢复原始形状（如果需要）
        if len(input_shape) > 2:
            z_q = z_q.view(input_shape)
            encoding_indices = encoding_indices.view(input_shape[:-1])
        
        return z_q, loss, encoding_indices
    
    def _ema_update(self, z_e: torch.Tensor, encoding_indices: torch.Tensor):
        """
        EMA 更新码本
        
        使用指数移动平均而非梯度下降更新码本向量，
        这样更稳定，且能有效防止码本坍塌。
        
        Args:
            z_e: 编码器输出 (B, D)
            encoding_indices: 码本索引 (B,)
        """
        # 转换为 one-hot 编码
        encodings = F.one_hot(encoding_indices, self.n_embeddings).float()  # (B, K)
        
        # 更新使用计数
        self.usage_count = self.usage_count + encodings.sum(0)
        
        # 更新集群大小（指数移动平均）
        self.ema_cluster_size = (
            self.decay * self.ema_cluster_size
            + (1 - self.decay) * encodings.sum(0)
        )
        
        # 更新码本向量的加权和
        dw = encodings.T @ z_e  # (K, D): 每个码本被分配到的所有 z_e 的和
        self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
        
        # Laplace 平滑防止除零
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.n_embeddings * self.epsilon)
            * n
        )
        
        # 更新嵌入权重
        self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)
    
    def get_codebook_usage(self) -> torch.Tensor:
        """
        获取码本使用率
        
        Returns:
            usage: 每个码本向量的使用比例 (K,)
        """
        total = self.ema_cluster_size.sum()
        if total > 0:
            return self.ema_cluster_size / total
        else:
            return torch.zeros_like(self.ema_cluster_size)
    
    def get_active_codes(self, threshold: float = 0.01) -> int:
        """
        获取活跃码本数量
        
        Args:
            threshold: 使用率阈值
            
        Returns:
            活跃码本数量
        """
        usage = self.get_codebook_usage()
        return (usage > threshold).sum().item()
    
    def reset_unused_codes(self, z_e: torch.Tensor, threshold: float = 0.01):
        """
        重置未使用的码本向量
        
        将长期未使用的码本向量重新初始化为随机选择的编码器输出，
        防止码本坍塌。
        
        Args:
            z_e: 编码器输出 (B, D)
            threshold: 使用率阈值
        """
        usage = self.get_codebook_usage()
        unused_mask = usage < threshold
        
        if unused_mask.any():
            # 随机选择一些编码器输出来替换未使用的码本
            num_unused = unused_mask.sum().item()
            random_indices = torch.randperm(z_e.size(0))[:num_unused]
            
            # 更新码本
            self.embedding.weight.data[unused_mask] = z_e[random_indices].detach()
            self.ema_w[unused_mask] = z_e[random_indices].detach()
            self.ema_cluster_size[unused_mask] = 1.0  # 重置计数


class VectorQuantizerEMA(VectorQuantizer):
    """
    VectorQuantizer 的别名，保持向后兼容
    """
    pass

