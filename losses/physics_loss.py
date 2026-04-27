"""
物理约束损失函数

包含平滑度约束和工况对比损失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SmoothnessLoss(nn.Module):
    """
    平滑度约束损失
    
    强迫潜在表示序列在稳态区域保持平滑，只在跳变处产生变化。
    效果：将噪声频繁的原始信号转化为干净的阶梯状工况特征。
    
    L_smooth = (1/(N-1)) * Σ ||z_t - z_{t-1}||_2
    
    Args:
        norm: 使用的范数类型（'l1', 'l2', 'huber'）
        reduction: 归约方式
    """
    
    def __init__(
        self,
        norm: str = 'l2',
        reduction: str = 'mean',
        huber_delta: float = 1.0
    ):
        super(SmoothnessLoss, self).__init__()
        self.norm = norm
        self.reduction = reduction
        self.huber_delta = huber_delta
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算平滑度损失
        
        Args:
            z: 潜在序列，shape (B, T, D) 或 (B, T)
            
        Returns:
            平滑度损失
        """
        # 计算相邻时间步的差分
        z_diff = z[:, 1:] - z[:, :-1]  # (B, T-1, D)
        
        if self.norm == 'l1':
            # L1 范数
            distances = torch.abs(z_diff).sum(dim=-1)  # (B, T-1)
        elif self.norm == 'l2':
            # L2 范数
            distances = torch.norm(z_diff, dim=-1)  # (B, T-1)
        elif self.norm == 'huber':
            # Huber 损失（对异常值更鲁棒）
            distances = F.huber_loss(
                z_diff, 
                torch.zeros_like(z_diff),
                reduction='none',
                delta=self.huber_delta
            ).sum(dim=-1)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")
        
        if self.reduction == 'mean':
            return distances.mean()
        elif self.reduction == 'sum':
            return distances.sum()
        else:
            return distances


class TotalVariationLoss(nn.Module):
    """
    全变分损失（Total Variation Loss）
    
    类似于平滑度约束，但更强调保留边缘（跳变）。
    
    L_TV = Σ |z_{t+1} - z_t|
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(TotalVariationLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算全变分损失
        
        Args:
            z: 潜在序列，shape (B, T, D)
            
        Returns:
            全变分损失
        """
        diff = z[:, 1:] - z[:, :-1]
        tv = torch.abs(diff).sum(dim=-1)
        
        if self.reduction == 'mean':
            return tv.mean()
        elif self.reduction == 'sum':
            return tv.sum()
        else:
            return tv


class ContrastiveLoss(nn.Module):
    """
    工况对比损失
    
    用于区分不同工况状态的潜在表示。
    
    Args:
        temperature: 温度参数
        margin: 负样本边界
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5
    ):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            anchor: 锚点特征，shape (B, D)
            positive: 正样本特征，shape (B, D)
            negative: 负样本特征，shape (B, D) 或 (B, N, D)
            
        Returns:
            对比损失
        """
        # 归一化
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        
        # 正样本相似度
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature
        
        if negative is not None:
            negative = F.normalize(negative, dim=-1)
            
            if negative.ndim == 2:
                negative = negative.unsqueeze(1)  # (B, 1, D)
            
            # 负样本相似度
            neg_sim = torch.bmm(
                anchor.unsqueeze(1),  # (B, 1, D)
                negative.transpose(1, 2)  # (B, D, N)
            ).squeeze(1) / self.temperature  # (B, N)
            
            # InfoNCE 损失
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+N)
            labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # 只有正样本对的情况
            loss = -pos_sim.mean()
        
        return loss


class ConsistencyLoss(nn.Module):
    """
    条件一致性损失
    
    在给定工况下，振动特征应符合预期模式。
    
    L_consistency = ||z_expected - z_actual||^2
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(ConsistencyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(
        self,
        z_expected: torch.Tensor,
        z_actual: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算一致性损失
        
        Args:
            z_expected: 预期特征
            z_actual: 实际特征
            weights: 可选的样本权重
            
        Returns:
            一致性损失
        """
        diff_sq = (z_expected - z_actual) ** 2
        
        if diff_sq.ndim > 1:
            diff_sq = diff_sq.sum(dim=-1)  # 对特征维度求和
        
        if weights is not None:
            diff_sq = diff_sq * weights
        
        if self.reduction == 'mean':
            return diff_sq.mean()
        elif self.reduction == 'sum':
            return diff_sq.sum()
        else:
            return diff_sq


class SynergyLoss(nn.Module):
    """
    双臂协同性损失
    
    惩罚双臂之间的不一致性。
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(SynergyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(
        self,
        synergy_distance: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算协同性损失
        
        Args:
            synergy_distance: 协同距离，shape (B,)
            weights: 可选的样本权重
            
        Returns:
            协同性损失
        """
        if weights is not None:
            synergy_distance = synergy_distance * weights
        
        if self.reduction == 'mean':
            return synergy_distance.mean()
        elif self.reduction == 'sum':
            return synergy_distance.sum()
        else:
            return synergy_distance


class ManifoldCompactnessLoss(nn.Module):
    """
    流形紧致性损失 - 惩罚同一 batch 内特征的方差
    
    强迫同一工况下的流形特征分布更紧凑。
    
    L_manifold = mean(||z - centroid||_2)
    
    Args:
        margin: 可选的边界阈值，距离小于 margin 时不惩罚
        reduction: 归约方式
    """
    
    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        super(ManifoldCompactnessLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        z_spd: torch.Tensor,
        q_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算流形紧致性损失
        
        Args:
            z_spd: SPD 流形特征，shape (B, D)
            q_context: 可选的工况上下文（暂未使用，预留分组功能）
            
        Returns:
            紧致性损失
        """
        # 计算 batch 内的特征质心
        centroid = z_spd.mean(dim=0, keepdim=True)  # (1, D)
        
        # 计算每个样本到质心的 L2 距离
        dist = torch.norm(z_spd - centroid, p=2, dim=1)  # (B,)
        
        # 应用边界阈值
        if self.margin > 0:
            dist = F.relu(dist - self.margin)
        
        if self.reduction == 'mean':
            return dist.mean()
        elif self.reduction == 'sum':
            return dist.sum()
        else:
            return dist

