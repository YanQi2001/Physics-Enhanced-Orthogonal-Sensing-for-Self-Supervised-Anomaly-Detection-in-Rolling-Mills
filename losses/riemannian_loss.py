"""
黎曼距离损失函数

用于 SPD（对称正定）矩阵的距离度量。
"""

import torch
import torch.nn as nn
from typing import Optional


def matrix_log(M: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    计算矩阵对数
    
    对于 SPD 矩阵 M = U Σ U^T，log(M) = U log(Σ) U^T
    
    注意：CSD 矩阵的特征值高度简并，需要较大的正则化参数。
    
    Args:
        M: SPD 矩阵，shape (..., n, n)
        eps: 数值稳定性参数（默认 1e-4，CSD 矩阵需要更大值）
        
    Returns:
        矩阵对数，shape (..., n, n)
    """
    # 添加较大的正则化确保正定性和特征值间距
    # CSD 矩阵的最小特征值间距接近 0，需要强正则化
    n = M.shape[-1]
    reg = eps * torch.eye(n, device=M.device, dtype=M.dtype)
    M_reg = M + reg
    
    # 确保对称性
    M_reg = (M_reg + M_reg.transpose(-2, -1)) / 2
    
    try:
        # 特征值分解
        eigenvalues, eigenvectors = torch.linalg.eigh(M_reg)
    except RuntimeError:
        # 如果仍然失败，使用 SVD 作为后备
        U, S, Vh = torch.linalg.svd(M_reg)
        eigenvalues = S
        eigenvectors = U
    
    # 确保特征值为正且有足够间距（数值稳定性）
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    
    # 计算 log(Σ)
    log_eigenvalues = torch.log(eigenvalues)
    
    # 限制 log 值范围，防止梯度爆炸
    log_eigenvalues = torch.clamp(log_eigenvalues, min=-20, max=20)
    
    # 重构 log(M) = U log(Σ) U^T
    log_M = eigenvectors @ torch.diag_embed(log_eigenvalues) @ eigenvectors.transpose(-2, -1)
    
    return log_M


def _safe_eigh(M: torch.Tensor, eps: float = 1e-7):
    """
    安全的特征值分解，处理病态矩阵
    
    Args:
        M: 对称矩阵
        eps: 正则化参数
        
    Returns:
        eigenvalues, eigenvectors
    """
    n = M.shape[-1]
    reg = eps * torch.eye(n, device=M.device, dtype=M.dtype)
    M_reg = M + reg
    
    # 确保对称性
    M_reg = (M_reg + M_reg.transpose(-2, -1)) / 2
    
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(M_reg)
    except RuntimeError:
        # 使用 SVD 作为后备
        U, S, Vh = torch.linalg.svd(M_reg)
        eigenvalues = S
        eigenvectors = U
    
    return eigenvalues, eigenvectors


def matrix_exp(M: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    计算矩阵指数
    
    对于对称矩阵 M = U Σ U^T，exp(M) = U exp(Σ) U^T
    
    Args:
        M: 对称矩阵，shape (..., n, n)
        eps: 数值稳定性参数
        
    Returns:
        矩阵指数，shape (..., n, n)
    """
    eigenvalues, eigenvectors = _safe_eigh(M, eps)
    # 限制指数值防止溢出
    eigenvalues = torch.clamp(eigenvalues, min=-20, max=20)
    exp_eigenvalues = torch.exp(eigenvalues)
    exp_M = eigenvectors @ torch.diag_embed(exp_eigenvalues) @ eigenvectors.transpose(-2, -1)
    return exp_M


def matrix_sqrt(M: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    计算矩阵平方根
    
    Args:
        M: SPD 矩阵
        eps: 数值稳定性参数
        
    Returns:
        矩阵平方根
    """
    eigenvalues, eigenvectors = _safe_eigh(M, eps)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    sqrt_M = eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-2, -1)
    return sqrt_M


def matrix_inv_sqrt(M: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    计算矩阵逆平方根 M^{-1/2}
    
    Args:
        M: SPD 矩阵
        eps: 数值稳定性参数
        
    Returns:
        矩阵逆平方根
    """
    eigenvalues, eigenvectors = _safe_eigh(M, eps)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues)
    inv_sqrt_M = eigenvectors @ torch.diag_embed(inv_sqrt_eigenvalues) @ eigenvectors.transpose(-2, -1)
    return inv_sqrt_M


class LogEuclideanDistance(nn.Module):
    """
    对数欧氏距离（Log-Euclidean Metric, LEM）
    
    d_LEM(M1, M2) = ||log(M1) - log(M2)||_F
    
    这是 SPD 流形上最简单的距离度量之一，计算效率高。
    """
    
    def __init__(self, eps: float = 1e-4, reduction: str = 'mean'):
        super(LogEuclideanDistance, self).__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(
        self, 
        M1: torch.Tensor, 
        M2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对数欧氏距离
        
        Args:
            M1: SPD 矩阵，shape (B, n, n) 或 (n, n)
            M2: SPD 矩阵，shape (B, n, n) 或 (n, n)
            
        Returns:
            距离，标量或 shape (B,)
        """
        # 计算矩阵对数
        log_M1 = matrix_log(M1, self.eps)
        log_M2 = matrix_log(M2, self.eps)
        
        # 计算 Frobenius 范数
        diff = log_M1 - log_M2
        distance = torch.norm(diff, p='fro', dim=(-2, -1))
        
        if self.reduction == 'mean':
            return distance.mean()
        elif self.reduction == 'sum':
            return distance.sum()
        else:
            return distance


class AffineInvariantRiemannianMetric(nn.Module):
    """
    仿射不变黎曼度量（Affine-Invariant Riemannian Metric, AIRM）
    
    d_AIRM(M1, M2) = ||log(M1^{-1/2} M2 M1^{-1/2})||_F
    
    这是 SPD 流形上的自然度量，具有仿射不变性：
    d(A M1 A^T, A M2 A^T) = d(M1, M2)
    
    对于工业故障检测非常有用：即使传感器增益变化，只要耦合关系不变，距离就不变。
    """
    
    def __init__(self, eps: float = 1e-7, reduction: str = 'mean'):
        super(AffineInvariantRiemannianMetric, self).__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(
        self,
        M1: torch.Tensor,
        M2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算仿射不变黎曼距离
        
        Args:
            M1: SPD 矩阵，shape (B, n, n)
            M2: SPD 矩阵，shape (B, n, n)
            
        Returns:
            距离
        """
        # 计算 M1^{-1/2}
        M1_inv_sqrt = matrix_inv_sqrt(M1, self.eps)
        
        # 计算 M1^{-1/2} M2 M1^{-1/2}
        middle = M1_inv_sqrt @ M2 @ M1_inv_sqrt
        
        # 计算 log
        log_middle = matrix_log(middle, self.eps)
        
        # Frobenius 范数
        distance = torch.norm(log_middle, p='fro', dim=(-2, -1))
        
        if self.reduction == 'mean':
            return distance.mean()
        elif self.reduction == 'sum':
            return distance.sum()
        else:
            return distance


class SPDFrobeniusDistance(nn.Module):
    """
    SPD 矩阵的 Frobenius 距离
    
    简单但不具有仿射不变性，仅用于快速近似。
    
    d_F(M1, M2) = ||M1 - M2||_F
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(SPDFrobeniusDistance, self).__init__()
        self.reduction = reduction
    
    def forward(
        self,
        M1: torch.Tensor,
        M2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Frobenius 距离
        
        Args:
            M1, M2: SPD 矩阵
            
        Returns:
            距离
        """
        diff = M1 - M2
        distance = torch.norm(diff, p='fro', dim=(-2, -1))
        
        if self.reduction == 'mean':
            return distance.mean()
        elif self.reduction == 'sum':
            return distance.sum()
        else:
            return distance


class GeodesicDistance(nn.Module):
    """
    SPD 流形上的测地线距离
    
    综合了 LEM 和 AIRM 的优点，可选择使用哪种度量。
    """
    
    def __init__(
        self,
        metric: str = 'lem',  # 'lem', 'airm', 'frobenius'
        eps: float = 1e-7,
        reduction: str = 'mean'
    ):
        super(GeodesicDistance, self).__init__()
        self.metric = metric
        self.eps = eps
        self.reduction = reduction
        
        if metric == 'lem':
            self.distance_fn = LogEuclideanDistance(eps, reduction)
        elif metric == 'airm':
            self.distance_fn = AffineInvariantRiemannianMetric(eps, reduction)
        elif metric == 'frobenius':
            self.distance_fn = SPDFrobeniusDistance(reduction)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def forward(
        self,
        M1: torch.Tensor,
        M2: torch.Tensor
    ) -> torch.Tensor:
        """计算距离"""
        return self.distance_fn(M1, M2)


class SPDContrastiveLoss(nn.Module):
    """
    SPD 流形上的对比学习损失
    
    用于自监督学习：拉近正样本对，推远负样本对。
    
    Args:
        metric: 距离度量类型
        margin: 负样本对的边界
        temperature: 温度参数
    """
    
    def __init__(
        self,
        metric: str = 'lem',
        margin: float = 1.0,
        temperature: float = 0.1,
        eps: float = 1e-7
    ):
        super(SPDContrastiveLoss, self).__init__()
        self.distance = GeodesicDistance(metric=metric, eps=eps, reduction='none')
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            anchor: 锚点样本
            positive: 正样本
            negative: 负样本
            
        Returns:
            对比损失
        """
        # 计算距离
        d_pos = self.distance(anchor, positive)
        d_neg = self.distance(anchor, negative)
        
        # Triplet loss with margin
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0)
        
        return loss.mean()

