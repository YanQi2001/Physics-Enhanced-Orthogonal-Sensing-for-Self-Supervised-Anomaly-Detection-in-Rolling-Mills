"""
SPDNet 的数值稳定函数库

核心功能：提供数值稳定的 SVD 分解及其反向传播
参考实现：https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
"""

import torch
from torch.autograd import Function

def symmetric(A):
    return 0.5 * (A + A.transpose(-2, -1))

class SafeSVD(Function):
    """
    数值稳定的 SVD 分解
    
    在反向传播中，当特征值过于接近时（lambda_i - lambda_j < eps），
    梯度的分母会趋于 0。本实现通过掩码机制避免这种情况。
    """
    @staticmethod
    def forward(ctx, input):
        # 确保输入是对称的
        input = symmetric(input)
        
        # SVD 分解
        # 注意：对于对称矩阵，SVD 和 特征值分解是等价的（特征值非负时）
        # 使用 eigh 通常比 svd 更快且更稳定
        try:
            U, S, Vh = torch.linalg.svd(input)
        except RuntimeError:
            # 极少数情况 SVD 失败，添加微小扰动重试
            input = input + 1e-4 * input.mean() * torch.randn_like(input)
            U, S, Vh = torch.linalg.svd(input)
            
        V = Vh.transpose(-2, -1)
        
        ctx.save_for_backward(U, S)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S = ctx.saved_tensors
        
        # 梯度裁剪：防止梯度爆炸
        grad_U = torch.clamp(grad_U, min=-100, max=100)
        grad_S = torch.clamp(grad_S, min=-100, max=100)
        grad_V = torch.clamp(grad_V, min=-100, max=100)
        
        # 重构 grad_input
        # 具体的 SVD 导数推导较为复杂，这里使用简化且稳定的形式
        
        # 构造 K 矩阵: K_ij = 1 / (s_i - s_j)
        S_view = S.unsqueeze(-1)
        # 避免除以零：当 |s_i - s_j| < eps 时，设 K_ij = 0
        diff = S_view - S_view.transpose(-2, -1)
        
        # 核心稳定性处理：
        # 对于非常接近的特征值，梯度贡献设为 0 (视为简并子空间)
        # 
        # 重要：CSD 矩阵的特征值高度简并，100% 样本的最小间距 < 1e-5
        # 因此需要使用更大的阈值，并用平滑近似替代硬阈值
        #
        # 使用 Lorentzian 平滑: K_ij = diff_ij / (diff_ij^2 + eps^2)
        # 这在 diff → 0 时趋于 0，在 diff 较大时趋于 1/diff
        eps_threshold = 1e-2  # 较大阈值处理简并特征值
        K = diff / (diff.pow(2) + eps_threshold ** 2)
        
        # 额外的梯度保护
        K = torch.clamp(K, min=-100, max=100)
        
        # 对角线设为 0
        K.diagonal(dim1=-2, dim2=-1).fill_(0)
        
        # 计算梯度
        # dL/dX = U * ( Symmetric(K * (U^T * dL/dU)) + diag(dL/dS) ) * U^T
        
        # 1. U^T * dL/dU
        Ut_grad_U = U.transpose(-2, -1).matmul(grad_U)
        
        # 2. Symmetric part
        # 这里的 symmetric 是 (A + A^T)/2
        term1 = 0.5 * (Ut_grad_U - Ut_grad_U.transpose(-2, -1))
        term1 = K * term1
        
        # 3. diag(dL/dS)
        term2 = torch.diag_embed(grad_S)
        
        # 组合
        mid = term1 + term2
        
        # 4. U * mid * U^T
        grad_input = U.matmul(mid).matmul(U.transpose(-2, -1))
        
        # 最终梯度裁剪
        grad_input = torch.clamp(grad_input, min=-100, max=100)
        
        # 检查 NaN
        if torch.isnan(grad_input).any():
            print("Warning: NaN gradient in SafeSVD backward, resetting to 0")
            grad_input = torch.zeros_like(grad_input)
            
        return symmetric(grad_input)

def safe_svd(x):
    """SafeSVD 的包装函数"""
    return SafeSVD.apply(x)

