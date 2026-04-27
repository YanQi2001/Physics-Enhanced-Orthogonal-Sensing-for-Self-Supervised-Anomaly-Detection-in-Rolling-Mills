"""
重构损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MaskedReconstructionLoss(nn.Module):
    """
    遮蔽重构损失
    
    只计算被遮蔽部分的重构误差（MAE 风格）。
    
    Args:
        loss_type: 损失类型 ('mse', 'l1', 'huber')
        reduction: 归约方式
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        reduction: str = 'mean'
    ):
        super(MaskedReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算遮蔽重构损失
        
        Args:
            pred: 预测值，shape (B, T, C) 或 (B, T)
            target: 目标值，shape (B, T, C) 或 (B, T)
            mask: 遮蔽，True 表示被遮蔽，shape (B, T)
            
        Returns:
            重构损失
        """
        # 扩展 mask 到与 pred 相同的形状
        if pred.ndim > mask.ndim:
            mask = mask.unsqueeze(-1).expand_as(pred)
        
        # 提取被遮蔽的元素
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        if len(pred_masked) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 计算损失
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred_masked, target_masked, reduction=self.reduction)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(pred_masked, target_masked, reduction=self.reduction)
        elif self.loss_type == 'huber':
            loss = F.huber_loss(pred_masked, target_masked, reduction=self.reduction)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class SPDReconstructionLoss(nn.Module):
    """
    SPD 矩阵重构损失
    
    用于 SPD 自编码器的重构损失。
    
    Args:
        metric: 距离度量 ('lem', 'frobenius')
    """
    
    def __init__(
        self,
        metric: str = 'lem',
        reduction: str = 'mean'
    ):
        super(SPDReconstructionLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction
        
        if metric == 'lem':
            from .riemannian_loss import LogEuclideanDistance
            self.distance = LogEuclideanDistance(reduction=reduction)
        else:
            from .riemannian_loss import SPDFrobeniusDistance
            self.distance = SPDFrobeniusDistance(reduction=reduction)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 SPD 重构损失
        
        Args:
            pred: 重构的 SPD 矩阵
            target: 原始 SPD 矩阵
            
        Returns:
            重构损失
        """
        return self.distance(pred, target)


class CombinedLoss(nn.Module):
    """
    组合损失
    
    将多个损失函数加权组合。
    """
    
    def __init__(
        self,
        losses: dict,
        weights: Optional[dict] = None
    ):
        """
        Args:
            losses: 损失函数字典 {'name': loss_fn}
            weights: 权重字典 {'name': weight}
        """
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses}
    
    def forward(self, **kwargs) -> dict:
        """
        计算组合损失
        
        Args:
            **kwargs: 传递给各损失函数的参数
            
        Returns:
            包含各项损失和总损失的字典
        """
        total_loss = 0.0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            # 尝试获取该损失函数需要的参数
            loss_value = loss_fn(**kwargs)
            loss_dict[name] = loss_value
            total_loss = total_loss + self.weights.get(name, 1.0) * loss_value
        
        loss_dict['total'] = total_loss
        return loss_dict
    
    def update_weights(self, weights: dict):
        """更新权重"""
        self.weights.update(weights)

