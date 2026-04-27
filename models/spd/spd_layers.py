"""
SPDNet 核心层实现

包含：
- StiefelParameter: Stiefel流形参数
- SPDTransform (BiMap): 双线性映射层
- SPDRectified (ReEig): 特征值校正层  
- SPDTangentSpace (LogEig): 对数欧氏映射层
- SPDUnTangentSpace (ExpEig): 指数映射层

参考: Huang, Z., & Van Gool, L. J. (2017). A Riemannian Network for SPD Matrix Learning.
"""

import torch
from torch import nn
from torch.autograd import Function
import numpy as np


# ==================== 工具函数 ====================

def symmetric(A):
    """对称化矩阵"""
    return 0.5 * (A + A.t())


def is_nan_or_inf(A):
    """检查是否包含NaN或Inf"""
    C1 = torch.nonzero(A == float('inf'))
    C2 = torch.nonzero(A != A)
    if len(C1.size()) > 0 or len(C2.size()) > 0:
        return True
    return False


def is_pos_def(x):
    """检查矩阵是否正定"""
    x = x.cpu().numpy()
    return np.all(np.linalg.eigvals(x) > 0)


def orthogonal_projection(A, B):
    """正交投影"""
    out = A - B.mm(symmetric(B.transpose(0, 1).mm(A)))
    return out


def retraction(A, ref):
    """Retraction映射，将切空间投影回Stiefel流形"""
    data = A + ref
    Q, R = torch.linalg.qr(data)
    # 处理R对角线上的负值
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out


# ==================== Stiefel参数 ====================

class StiefelParameter(nn.Parameter):
    """
    Stiefel流形上的参数
    
    Stiefel流形 St(n,p) = {W ∈ R^{n×p} : W^T W = I_p}
    即所有列正交矩阵构成的流形
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'StiefelParameter containing:' + self.data.__repr__()


# ==================== SPD维度增加层 ====================

class SPDIncreaseDim(nn.Module):
    """
    SPD矩阵维度增加层
    
    将 n×n 的SPD矩阵扩展为 m×m (m > n)
    通过块对角扩展：[[M, 0], [0, I]]
    """
    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        self.register_buffer('eye', torch.eye(output_size, input_size, dtype=torch.float64))
        add = np.asarray([0] * input_size + [1] * (output_size - input_size), dtype=np.float64)
        self.register_buffer('add', torch.from_numpy(np.diag(add)))

    def forward(self, input):
        eye = self.eye.unsqueeze(0).expand(input.size(0), -1, -1)
        add = self.add.unsqueeze(0).expand(input.size(0), -1, -1)
        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1, 2)))
        return output


# ==================== BiMap层（双线性映射） ====================

class SPDTransform(nn.Module):
    """
    双线性映射层（BiMap Layer）
    
    Y = W X W^T
    
    其中 W 是 Stiefel 流形上的正交矩阵参数。
    这相当于在SPD流形上的"旋转"操作。
    
    Args:
        input_size: 输入矩阵维度
        output_size: 输出矩阵维度
    """
    def __init__(self, input_size, output_size):
        super(SPDTransform, self).__init__()
        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size

        self.weight = StiefelParameter(
            torch.empty(input_size, output_size, dtype=torch.float64), 
            requires_grad=True
        )
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        output = input
        if self.increase_dim:
            output = self.increase_dim(output)
            
        # 强制正交化权重 (Gram-Schmidt 或 QR分解)
        if self.training:
            with torch.no_grad():
                q, r = torch.linalg.qr(self.weight)
                self.weight.data = q
                
        weight = self.weight.unsqueeze(0).expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1, 2), torch.bmm(output, weight))
        return output


# ==================== ReEig层（特征值校正） ====================

from .functional import safe_svd

# ==================== ReEig层（特征值校正） ====================

class SPDRectifiedFunction(Function):
    """特征值校正的前向/反向传播"""
    
    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)
        output = input.new(input.size(0), input.size(1), input.size(2))

        for k, x in enumerate(input):
            # 使用 SafeSVD 替代原始 SVD
            u, s, v = safe_svd(x)
            
            s = torch.clamp(s, min=epsilon[0].item())
            output[k] = u.mm(torch.diag(s).mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_tensors
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            eye = torch.eye(input.size(1), dtype=input.dtype, device=input.device)
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            
            for k, g in enumerate(grad_output):
                if len(g.shape) == 1:
                    continue

                g = symmetric(g)
                x = input[k]
                
                # 使用 SafeSVD
                u, s, v = safe_svd(x)
                
                max_mask = s > epsilon[0]
                s_max_diag = s.clone()
                s_max_diag[~max_mask] = epsilon[0]
                s_max_diag = torch.diag(s_max_diag)
                Q = max_mask.double().diag()
                
                dLdV = 2 * (g.mm(u.mm(s_max_diag)))
                dLdS = eye * (Q.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1).expand(-1, s.size(0))
                P = P - P.t()
                
                # 使用 Lorentzian 平滑处理简并特征值
                # CSD 矩阵特征值高度简并，需要较大阈值
                eps_threshold = 1e-2
                P = P / (P.pow(2) + eps_threshold ** 2)

                grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV)) + dLdS).mm(u.t())
            
        return grad_input, None


class SPDRectified(nn.Module):
    """
    特征值校正层（ReEig Layer）
    """
    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.tensor([epsilon], dtype=torch.float64))

    def forward(self, input):
        # 简化实现，直接使用 forward 计算，让 PyTorch 自动求导处理 safe_svd 的梯度
        # 注意：这里我们不再调用 SPDRectifiedFunction，而是直接写 forward 逻辑
        # 因为 safe_svd 是可微的
        
        output = []
        for x in input:
            u, s, v = safe_svd(x)
            s = torch.clamp(s, min=self.epsilon.item())
            output.append(u.mm(torch.diag(s).mm(u.t())))
            
        return torch.stack(output)


# ==================== LogEig层（切空间映射） ====================

class SPDTangentSpace(nn.Module):
    """
    对数欧氏映射层（LogEig Layer）
    """
    def __init__(self, input_size, vectorize=True):
        super(SPDTangentSpace, self).__init__()
        self.vectorize = vectorize
        if vectorize:
            self.vec = SPDVectorize(input_size)

    def forward(self, input):
        # 直接使用 safe_svd 实现，自动求导
        output = []
        for x in input:
            u, s, v = safe_svd(x)
            s = torch.clamp(s, min=1e-6) # 确保正定
            s_log = torch.log(s)
            output.append(u.mm(torch.diag(s_log).mm(u.t())))
        
        output = torch.stack(output)
        
        if self.vectorize:
            output = self.vec(output)
        return output


# ==================== ExpEig层（逆切空间映射） ====================

class SPDUnTangentSpace(nn.Module):
    """
    指数映射层（ExpEig Layer）
    """
    def __init__(self, unvectorize=True):
        super(SPDUnTangentSpace, self).__init__()
        self.unvectorize = unvectorize
        if unvectorize:
            self.unvec = SPDUnVectorize()

    def forward(self, input):
        if self.unvectorize:
            input = self.unvec(input)
            
        # 直接使用 safe_svd 实现，自动求导
        output = []
        for x in input:
            u, s, v = safe_svd(x)
            # 限制指数范围
            s = torch.clamp(s, min=-10, max=10)
            s_exp = torch.exp(s)
            output.append(u.mm(torch.diag(s_exp).mm(u.t())))
            
        return torch.stack(output)


# ==================== 向量化层 ====================

class SPDVectorize(nn.Module):
    """
    SPD矩阵向量化
    
    将对称矩阵的上三角部分提取为向量。
    对于 n×n 矩阵，输出维度为 n(n+1)/2。
    """
    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = np.triu_indices(input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output


class SPDUnVectorizeFunction(Function):
    """向量还原为矩阵的前向/反向传播"""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        n = int(-0.5 + 0.5 * np.sqrt(1 + 8 * input.size(1)))
        output = input.new(len(input), n, n)
        output.fill_(0)
        mask_upper = np.triu_indices(n)
        mask_diag = np.diag_indices(n)
        
        for k, x in enumerate(input):
            output[k][mask_upper] = x
            output[k] = output[k] + output[k].t()
            output[k][mask_diag] /= 2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            n = int(-0.5 + 0.5 * np.sqrt(1 + 8 * input.size(1)))
            grad_input = input.new(len(input), input.size(1))
            mask = np.triu_indices(n)
            for k, g in enumerate(grad_output):
                grad_input[k] = g[mask]

        return grad_input


class SPDUnVectorize(nn.Module):
    """
    向量还原为SPD矩阵
    
    将向量还原为对称矩阵（填充上下三角）。
    """
    def __init__(self):
        super(SPDUnVectorize, self).__init__()

    def forward(self, input):
        return SPDUnVectorizeFunction.apply(input)

