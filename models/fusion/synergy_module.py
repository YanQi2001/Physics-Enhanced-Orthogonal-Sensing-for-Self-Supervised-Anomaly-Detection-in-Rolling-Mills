"""
黎曼协同感知模块

监控双臂动作一致性，使用黎曼几何距离。
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from models.spd.spd_layers import symmetric


def matrix_log(M: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """计算矩阵对数"""
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    log_eigenvalues = torch.log(eigenvalues)
    log_M = eigenvectors @ torch.diag_embed(log_eigenvalues) @ eigenvectors.transpose(-2, -1)
    return log_M


def matrix_inv_sqrt(M: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """计算矩阵逆平方根"""
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    inv_sqrt_eigenvalues = 1.0 / torch.sqrt(eigenvalues)
    inv_sqrt_M = eigenvectors @ torch.diag_embed(inv_sqrt_eigenvalues) @ eigenvectors.transpose(-2, -1)
    return inv_sqrt_M


class RiemannianSynergyModule(nn.Module):
    """
    黎曼协同感知模块
    
    从 16×16 CSD 矩阵中提取 Arm1 和 Arm2 的子矩阵，
    计算它们在 SPD 流形上的测地线距离。
    
    在正常协同状态下，两个导卫臂的动力学特性应相似，该距离较小。
    如果一侧发生偏载或磨损，该黎曼距离会急剧增大。
    
    Args:
        matrix_size: 输入矩阵大小（默认16）
        arm1_indices: Arm1 的通道索引
        arm2_indices: Arm2 的通道索引
        metric: 距离度量类型 ('airm', 'lem')
        eps: 数值稳定性参数
    """
    
    def __init__(
        self,
        matrix_size: int = 16,
        arm1_indices: Optional[Tuple[int, ...]] = None,
        arm2_indices: Optional[Tuple[int, ...]] = None,
        metric: str = 'airm',
        eps: float = 1e-7
    ):
        super(RiemannianSynergyModule, self).__init__()
        
        self.matrix_size = matrix_size
        self.metric = metric
        self.eps = eps
        
        # 默认索引：前8个通道为Arm1，后8个为Arm2
        if arm1_indices is None:
            arm1_indices = tuple(range(8))
        if arm2_indices is None:
            arm2_indices = tuple(range(8, 16))
        
        self.register_buffer('arm1_idx', torch.tensor(arm1_indices))
        self.register_buffer('arm2_idx', torch.tensor(arm2_indices))
        
        self.arm_size = len(arm1_indices)
    
    def extract_arm_submatrices(
        self, 
        M: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取 Arm1 和 Arm2 的子矩阵
        
        Args:
            M: 16×16 CSD 矩阵，shape (B, 16, 16) 或 (16, 16)
            
        Returns:
            (M1, M2): Arm1 和 Arm2 的子矩阵，各为 8×8
        """
        squeeze = False
        if M.ndim == 2:
            M = M.unsqueeze(0)
            squeeze = True
        
        # 提取子矩阵
        # M1[i,j] = M[arm1_idx[i], arm1_idx[j]]
        M1 = M[:, self.arm1_idx][:, :, self.arm1_idx]  # (B, 8, 8)
        M2 = M[:, self.arm2_idx][:, :, self.arm2_idx]  # (B, 8, 8)
        
        if squeeze:
            M1 = M1.squeeze(0)
            M2 = M2.squeeze(0)
        
        return M1, M2
    
    def geodesic_distance_airm(
        self, 
        M1: torch.Tensor, 
        M2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算仿射不变黎曼距离（AIRM）
        
        d_AIRM(M1, M2) = ||log(M1^{-1/2} M2 M1^{-1/2})||_F
        
        Args:
            M1, M2: SPD 矩阵，shape (B, n, n)
            
        Returns:
            距离，shape (B,)
        """
        # 计算 M1^{-1/2}
        M1_inv_sqrt = matrix_inv_sqrt(M1, self.eps)
        
        # 计算 M1^{-1/2} M2 M1^{-1/2}
        middle = M1_inv_sqrt @ M2 @ M1_inv_sqrt
        
        # 对称化（处理数值误差）
        middle = (middle + middle.transpose(-2, -1)) / 2
        
        # 计算 log
        log_middle = matrix_log(middle, self.eps)
        
        # Frobenius 范数
        distance = torch.norm(log_middle, p='fro', dim=(-2, -1))
        
        return distance
    
    def geodesic_distance_lem(
        self, 
        M1: torch.Tensor, 
        M2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对数欧氏距离（LEM）
        
        d_LEM(M1, M2) = ||log(M1) - log(M2)||_F
        
        Args:
            M1, M2: SPD 矩阵
            
        Returns:
            距离
        """
        log_M1 = matrix_log(M1, self.eps)
        log_M2 = matrix_log(M2, self.eps)
        
        diff = log_M1 - log_M2
        distance = torch.norm(diff, p='fro', dim=(-2, -1))
        
        return distance
    
    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        计算双臂协同距离
        
        Args:
            M: 16×16 CSD 矩阵，shape (B, 16, 16)
            
        Returns:
            协同距离 D_synergy，shape (B,)
        """
        # 处理复数矩阵（取实部或模）
        if M.is_complex():
            # 使用 Hermitian 矩阵的实部（对于正定 Hermitian 矩阵）
            M = M.real
        
        # 提取子矩阵
        M1, M2 = self.extract_arm_submatrices(M)
        
        # 计算距离
        if self.metric == 'airm':
            distance = self.geodesic_distance_airm(M1, M2)
        elif self.metric == 'lem':
            distance = self.geodesic_distance_lem(M1, M2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return distance
    
    def get_arm_features(
        self, 
        M: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取两个臂的切空间特征
        
        Args:
            M: CSD 矩阵
            
        Returns:
            (log_M1, log_M2): 切空间表示
        """
        if M.is_complex():
            M = M.real
        
        M1, M2 = self.extract_arm_submatrices(M)
        
        log_M1 = matrix_log(M1, self.eps)
        log_M2 = matrix_log(M2, self.eps)
        
        return log_M1, log_M2


class CrossArmCorrelationModule(nn.Module):
    """
    跨臂相关性模块
    
    除了计算协同距离，还分析两臂之间的交叉相关性。
    """
    
    def __init__(
        self,
        matrix_size: int = 16,
        eps: float = 1e-7
    ):
        super(CrossArmCorrelationModule, self).__init__()
        
        self.synergy = RiemannianSynergyModule(matrix_size=matrix_size, eps=eps)
        
        # 跨臂子矩阵索引
        # 提取 M[0:8, 8:16] 区域（Arm1 到 Arm2 的耦合）
        self.arm1_idx = torch.arange(8)
        self.arm2_idx = torch.arange(8, 16)
    
    def extract_cross_coupling(self, M: torch.Tensor) -> torch.Tensor:
        """
        提取跨臂耦合子矩阵
        
        Args:
            M: 16×16 矩阵
            
        Returns:
            跨臂耦合矩阵，shape (B, 8, 8)
        """
        if M.ndim == 2:
            M = M.unsqueeze(0)
        
        # M[arm1, arm2] 区域
        cross = M[:, :8, 8:]
        
        return cross
    
    def forward(self, M: torch.Tensor) -> dict:
        """
        前向传播
        
        Returns:
            dict with 'synergy_distance', 'cross_coupling_norm'
        """
        # 协同距离
        d_synergy = self.synergy(M)
        
        # 跨臂耦合强度
        if M.is_complex():
            cross = self.extract_cross_coupling(M)
            cross_norm = torch.norm(torch.abs(cross), p='fro', dim=(-2, -1))
        else:
            cross = self.extract_cross_coupling(M)
            cross_norm = torch.norm(cross, p='fro', dim=(-2, -1))
        
        return {
            'synergy_distance': d_synergy,
            'cross_coupling_norm': cross_norm
        }

