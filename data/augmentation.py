"""
数据增强模块

包含用于对比学习的正交性扰动增强。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class OrthogonalityPerturbation:
    """
    正交性扰动增强 - 用于生成对比学习的负样本
    
    对 CSD 矩阵的非对角块（P-V 耦合项）添加噪声，
    破坏 P⊥V 的统计解耦关系。
    
    16×16 CSD 矩阵结构：
    - 对角块 (4个 4×4)：各物理通道内部的跨频带调制关系
    - 非对角块：跨通道、跨模态、跨臂的联合耦合关系
    
    Args:
        perturbation_scale: 扰动强度（相对于矩阵 Frobenius 范数的比例）
        mode: 扰动模式
            - 'full': 扰动所有非对角元素
            - 'pv_coupling': 仅扰动 P-V 耦合块
            - 'cross_arm': 仅扰动跨臂块
        preserve_hermitian: 是否保持埃尔米特对称性
    """
    
    def __init__(
        self,
        perturbation_scale: float = 0.1,
        mode: str = 'full',
        preserve_hermitian: bool = True,
        regularization_eps: float = 1e-6
    ):
        self.scale = perturbation_scale
        self.mode = mode
        self.preserve_hermitian = preserve_hermitian
        self.regularization_eps = regularization_eps
    
    def _create_perturbation_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        创建扰动掩码
        
        Args:
            size: 矩阵大小（默认16）
            device: 设备
            
        Returns:
            扰动掩码，shape (size, size)
        """
        mask = torch.zeros(size, size, device=device)
        
        if self.mode == 'full':
            # 扰动所有非对角元素
            mask = 1.0 - torch.eye(size, device=device)
            
        elif self.mode == 'pv_coupling':
            # 仅扰动 P-V 耦合块
            # 结构：每个臂有 4 通道（原始 + 3 子频带）
            # Arm1: [P1(0:4), V1(4:8)], Arm2: [P2(8:12), V2(12:16)]
            # P-V 耦合块位于：
            # - Arm1: (0:4, 4:8) 和 (4:8, 0:4)
            # - Arm2: (8:12, 12:16) 和 (12:16, 8:12)
            mask[0:4, 4:8] = 1.0
            mask[4:8, 0:4] = 1.0
            mask[8:12, 12:16] = 1.0
            mask[12:16, 8:12] = 1.0
            
        elif self.mode == 'cross_arm':
            # 仅扰动跨臂块（Arm1 和 Arm2 之间的耦合）
            # Arm1: 0:8, Arm2: 8:16
            mask[0:8, 8:16] = 1.0
            mask[8:16, 0:8] = 1.0
        
        else:
            raise ValueError(f"Unknown perturbation mode: {self.mode}")
        
        return mask
    
    def _project_to_spd(self, M: torch.Tensor) -> torch.Tensor:
        """
        将扰动后的矩阵重新投影回 SPD 流形
        
        使用特征值分解 + 正值约束 + 正则化
        
        Args:
            M: 扰动后的矩阵，shape (..., n, n)
            
        Returns:
            SPD 矩阵
        """
        # 对复数矩阵使用 SVD
        if M.is_complex():
            U, S, Vh = torch.linalg.svd(M)
            # 确保特征值为正
            S_clamp = torch.clamp(S.real, min=self.regularization_eps)
            # 重构
            reconstructed = U @ torch.diag_embed(S_clamp.to(U.dtype)) @ Vh
            # 强制埃尔米特对称
            if self.preserve_hermitian:
                reconstructed = (reconstructed + reconstructed.mH) / 2
        else:
            # 对实数矩阵使用特征值分解
            eigenvalues, eigenvectors = torch.linalg.eigh(M)
            # 确保特征值为正
            eigenvalues_clamp = torch.clamp(eigenvalues, min=self.regularization_eps)
            # 重构
            reconstructed = eigenvectors @ torch.diag_embed(eigenvalues_clamp) @ eigenvectors.mH
        
        return reconstructed
    
    def __call__(self, csd_matrix: torch.Tensor) -> torch.Tensor:
        """
        对 CSD 矩阵应用正交性扰动
        
        Args:
            csd_matrix: CSD 矩阵，shape (n, n) 或 (B, n, n)，可以是复数或实数
            
        Returns:
            扰动后的矩阵（重新投影回 SPD 流形）
        """
        squeeze = False
        if csd_matrix.ndim == 2:
            csd_matrix = csd_matrix.unsqueeze(0)
            squeeze = True
        
        B, n, _ = csd_matrix.shape
        device = csd_matrix.device
        
        # 计算扰动强度（相对于矩阵范数）
        matrix_norm = torch.norm(csd_matrix, p='fro', dim=(-2, -1), keepdim=True)
        noise_scale = self.scale * matrix_norm / n
        
        # 生成噪声
        if csd_matrix.is_complex():
            # 复数噪声
            noise_real = torch.randn(B, n, n, device=device)
            noise_imag = torch.randn(B, n, n, device=device)
            noise = (noise_real + 1j * noise_imag) * noise_scale
            
            # 确保噪声是埃尔米特的（保持矩阵性质）
            if self.preserve_hermitian:
                noise = (noise + noise.mH) / 2
        else:
            # 实数噪声（对称）
            noise = torch.randn(B, n, n, device=device) * noise_scale
            if self.preserve_hermitian:
                noise = (noise + noise.transpose(-2, -1)) / 2
        
        # 创建掩码
        mask = self._create_perturbation_mask(n, device)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        
        # 应用扰动
        perturbed = csd_matrix + noise * mask
        
        # 重新投影回 SPD 流形
        result = self._project_to_spd(perturbed)
        
        if squeeze:
            result = result.squeeze(0)
        
        return result


class TimeWarpingAugmentation:
    """
    时间扭曲增强
    
    对时间序列进行非线性时间扭曲，用于增强模型对时间变形的鲁棒性。
    
    Args:
        warp_strength: 扭曲强度
        n_knots: 控制点数量
    """
    
    def __init__(self, warp_strength: float = 0.1, n_knots: int = 4):
        self.warp_strength = warp_strength
        self.n_knots = n_knots
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        对信号应用时间扭曲
        
        Args:
            x: 输入信号，shape (T, C) 或 (B, T, C)
            
        Returns:
            扭曲后的信号
        """
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            squeeze = True
        
        B, T, C = x.shape
        device = x.device
        
        # 生成扭曲函数的控制点
        orig_steps = torch.linspace(0, 1, self.n_knots + 2, device=device)
        random_warps = torch.randn(B, self.n_knots, device=device) * self.warp_strength
        warp_steps = torch.cat([
            torch.zeros(B, 1, device=device),
            orig_steps[1:-1].unsqueeze(0).expand(B, -1) + random_warps,
            torch.ones(B, 1, device=device)
        ], dim=1)
        
        # 确保单调性
        warp_steps = torch.cumsum(torch.softmax(warp_steps, dim=1), dim=1)
        warp_steps = warp_steps / warp_steps[:, -1:] * (T - 1)
        
        # 使用线性插值进行扭曲
        time_steps = torch.arange(T, device=device).float()
        result = torch.zeros_like(x)
        
        for b in range(B):
            for c in range(C):
                result[b, :, c] = torch.from_numpy(
                    np.interp(
                        time_steps.cpu().numpy(),
                        warp_steps[b].cpu().numpy(),
                        orig_steps.cpu().numpy() * (T - 1)
                    )
                ).to(device)
                # 从原始信号中采样
                indices = result[b, :, c].long().clamp(0, T - 1)
                result[b, :, c] = x[b, indices, c]
        
        if squeeze:
            result = result.squeeze(0)
        
        return result


class GaussianNoiseAugmentation:
    """
    高斯噪声增强
    
    Args:
        noise_std: 噪声标准差（相对于信号标准差）
    """
    
    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加高斯噪声
        
        Args:
            x: 输入信号
            
        Returns:
            添加噪声后的信号
        """
        std = x.std() * self.noise_std
        noise = torch.randn_like(x) * std
        return x + noise


class CompositeAugmentation:
    """
    组合增强
    
    按顺序应用多个增强
    """
    
    def __init__(self, augmentations: list, p: list = None):
        """
        Args:
            augmentations: 增强列表
            p: 各增强的应用概率（None 表示全部应用）
        """
        self.augmentations = augmentations
        self.p = p if p is not None else [1.0] * len(augmentations)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for aug, prob in zip(self.augmentations, self.p):
            if torch.rand(1).item() < prob:
                x = aug(x)
        return x

