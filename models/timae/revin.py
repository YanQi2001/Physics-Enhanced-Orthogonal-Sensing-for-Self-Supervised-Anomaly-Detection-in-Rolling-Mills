"""
RevIN: 可逆实例归一化

用于时序数据的分布对齐，支持训练时归一化、推理时反归一化。
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    可逆实例归一化（Reversible Instance Normalization）
    
    在训练时对输入进行归一化，在推理时可以反归一化恢复原始尺度。
    
    Args:
        num_features: 特征/通道数
        eps: 数值稳定性参数
        affine: 是否使用可学习的仿射参数
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        # 运行时统计量（非参数）
        self.mean = None
        self.stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，shape (B, T, C) 或 (B, T)
            mode: 'norm' 归一化 或 'denorm' 反归一化
            
        Returns:
            处理后的张量
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")
        return x

    def _get_statistics(self, x: torch.Tensor):
        """计算统计量（均值和标准差）"""
        # 对时间维度计算统计量
        dim2reduce = tuple(range(1, x.ndim - 1)) if x.ndim > 2 else (1,)
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """归一化"""
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化"""
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """便捷方法：归一化"""
        return self.forward(x, 'norm')
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """便捷方法：反归一化"""
        return self.forward(x, 'denorm')

