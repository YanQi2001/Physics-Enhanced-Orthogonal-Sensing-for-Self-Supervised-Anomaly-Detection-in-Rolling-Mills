"""
条件注意力融合层

使用工况上下文作为 Query，从振动流形特征中检索预期模式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContextConditionedAttention(nn.Module):
    """
    条件注意力融合层
    
    核心思想：P(Vibration Manifold | Pressure Context)
    
    使用工况上下文（Q_context）作为 Query，从振动流形特征（Z_SPD）中
    检索"在该工况下的预期模式"。
    
    Args:
        d_spd: SPD 流形特征维度
        d_context: 工况上下文维度
        d_model: 注意力模型维度
        n_heads: 注意力头数
        n_reference_states: 可学习参考状态数量
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        d_spd: int = 136,  # 16×17/2 for 16×16 SPD matrix
        d_context: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_reference_states: int = 4,
        dropout: float = 0.1
    ):
        super(ContextConditionedAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # 投影层
        self.q_proj = nn.Linear(d_context, d_model)
        self.k_proj = nn.Linear(d_spd, d_model)
        self.v_proj = nn.Linear(d_spd, d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 可学习的参考流形嵌入
        # 代表各工况状态下的典型流形特征
        self.reference_embeddings = nn.Parameter(
            torch.randn(n_reference_states, d_model) * 0.02
        )
        
        # 参考状态的 Key 和 Value
        self.ref_k = nn.Linear(d_model, d_model)
        self.ref_v = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        z_spd: torch.Tensor,
        q_context: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            z_spd: 振动流形特征，shape (B, D_spd)
            q_context: 工况上下文，shape (B, D_context)
            return_attention: 是否返回注意力权重
            
        Returns:
            (z_expected, attention_weights) 
            z_expected: 预期的流形特征，shape (B, D_model)
        """
        batch_size = z_spd.size(0)
        
        # 投影 Query (来自工况上下文)
        Q = self.q_proj(q_context)  # (B, d_model)
        Q = self.norm_q(Q)
        Q = Q.view(batch_size, 1, self.n_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (B, n_heads, 1, head_dim)
        
        # 投影 Key 和 Value (来自振动特征)
        K_spd = self.k_proj(z_spd)  # (B, d_model)
        V_spd = self.v_proj(z_spd)  # (B, d_model)
        
        # 获取参考状态的 Key 和 Value
        ref_K = self.ref_k(self.reference_embeddings)  # (n_ref, d_model)
        ref_V = self.ref_v(self.reference_embeddings)  # (n_ref, d_model)
        
        # 扩展到 batch
        ref_K = ref_K.unsqueeze(0).expand(batch_size, -1, -1)  # (B, n_ref, d_model)
        ref_V = ref_V.unsqueeze(0).expand(batch_size, -1, -1)  # (B, n_ref, d_model)
        
        # 拼接实际特征和参考特征
        K = torch.cat([K_spd.unsqueeze(1), ref_K], dim=1)  # (B, 1+n_ref, d_model)
        V = torch.cat([V_spd.unsqueeze(1), ref_V], dim=1)  # (B, 1+n_ref, d_model)
        
        K = self.norm_k(K)
        
        # 重塑为多头
        K = K.view(batch_size, -1, self.n_heads, self.head_dim)
        K = K.transpose(1, 2)  # (B, n_heads, 1+n_ref, head_dim)
        
        V = V.view(batch_size, -1, self.n_heads, self.head_dim)
        V = V.transpose(1, 2)  # (B, n_heads, 1+n_ref, head_dim)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # (B, n_heads, 1, 1+n_ref)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # (B, n_heads, 1, head_dim)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, self.d_model)
        
        # 输出投影
        z_expected = self.out_proj(attn_output)
        
        if return_attention:
            return z_expected, attn_weights.squeeze(2)  # (B, n_heads, 1+n_ref)
        
        return z_expected, None
    
    def get_reference_similarities(self, q_context: torch.Tensor) -> torch.Tensor:
        """
        获取工况与各参考状态的相似度
        
        Args:
            q_context: 工况上下文，shape (B, D_context)
            
        Returns:
            相似度，shape (B, n_ref)
        """
        Q = self.q_proj(q_context)  # (B, d_model)
        ref_K = self.ref_k(self.reference_embeddings)  # (n_ref, d_model)
        
        # 计算余弦相似度
        Q_norm = F.normalize(Q, dim=-1)
        ref_K_norm = F.normalize(ref_K, dim=-1)
        
        similarities = torch.matmul(Q_norm, ref_K_norm.T)  # (B, n_ref)
        
        return similarities
    
    @torch.no_grad()
    def initialize_references_from_data(
        self,
        q_contexts: torch.Tensor,
        z_spds: torch.Tensor,
        n_clusters: int = None
    ):
        """
        使用 K-Means 从数据中初始化参考状态嵌入
        
        物理语义：
        - 簇 0: 低位稳态（低压平台）
        - 簇 1: 高位稳态（高压平台）
        - 簇 2: 上升过渡（咬钢）
        - 簇 3: 下降过渡（抛钢）
        
        Args:
            q_contexts: 工况上下文向量，shape (N, D_context)
            z_spds: SPD 特征向量，shape (N, D_spd)
            n_clusters: 聚类数（默认使用 n_reference_states）
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("Warning: sklearn not available, skipping reference initialization")
            return
        
        if n_clusters is None:
            n_clusters = self.reference_embeddings.size(0)
        
        # 确保数据在 CPU 上进行聚类
        q_contexts_np = q_contexts.cpu().numpy()
        z_spds_cpu = z_spds.cpu()
        
        # 对 q_context 进行 K-Means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(q_contexts_np)
        
        # 计算每个簇对应的 z_spd 均值，并投影到 d_model 维度
        device = self.reference_embeddings.device
        
        for i in range(n_clusters):
            cluster_mask = (labels == i)
            if cluster_mask.sum() == 0:
                # 如果某个簇没有样本，保持随机初始化
                continue
            
            # 计算该簇的 z_spd 均值
            cluster_z_spd = z_spds_cpu[cluster_mask].mean(dim=0)
            
            # 投影到 d_model 维度
            cluster_z_spd = cluster_z_spd.to(device)
            ref_embedding = self.k_proj(cluster_z_spd.unsqueeze(0)).squeeze(0)
            
            # 赋值给 reference_embeddings
            self.reference_embeddings.data[i] = ref_embedding
        
        # 统计各簇样本数
        cluster_counts = [int((labels == i).sum()) for i in range(n_clusters)]
        print(f"Initialized {n_clusters} reference states from {len(q_contexts)} samples")
        print(f"Cluster distribution: {cluster_counts}")


class CrossModalFusion(nn.Module):
    """
    完整的跨模态融合模块
    
    结合条件注意力和残差连接。
    """
    
    def __init__(
        self,
        d_spd: int = 136,
        d_context: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_reference_states: int = 4,
        dropout: float = 0.1
    ):
        super(CrossModalFusion, self).__init__()
        
        # 条件注意力
        self.attention = ContextConditionedAttention(
            d_spd=d_spd,
            d_context=d_context,
            d_model=d_model,
            n_heads=n_heads,
            n_reference_states=n_reference_states,
            dropout=dropout
        )
        
        # 投影层（将 SPD 特征投影到与 attention 输出相同的维度）
        self.spd_proj = nn.Linear(d_spd, d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        z_spd: torch.Tensor,
        q_context: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            z_spd: 振动流形特征
            q_context: 工况上下文
            return_attention: 是否返回注意力权重
            
        Returns:
            (fused_features, z_expected) 或 (fused_features, z_expected, attn_weights)
        """
        # 条件注意力
        z_expected, attn_weights = self.attention(z_spd, q_context, return_attention=return_attention)
        
        # 投影 SPD 特征
        z_spd_proj = self.spd_proj(z_spd.float())
        
        # 残差连接 + Layer Norm
        fused = self.norm1(z_spd_proj + z_expected)
        
        # FFN + 残差
        fused = fused + self.ffn(fused)
        fused = self.norm2(fused)
        
        if return_attention:
            return fused, z_expected, attn_weights
        return fused, z_expected

