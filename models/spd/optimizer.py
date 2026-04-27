"""
Stiefel 流形优化器

在Stiefel流形 St(n,p) = {W ∈ R^{n×p} : W^T W = I_p} 上进行优化。
"""

import torch
from .spd_layers import StiefelParameter, symmetric, orthogonal_projection, retraction


class StiefelMetaOptimizer:
    """
    Stiefel 流形元优化器
    
    包裹标准优化器（如Adam、SGD），在每次更新后将 StiefelParameter 
    投影回 Stiefel 流形，确保正交约束。
    
    Args:
        optimizer: 标准PyTorch优化器
    """
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self, closure=None):
        """
        执行一步优化
        
        对于普通参数，直接使用标准优化器更新。
        对于 StiefelParameter，先投影梯度到切空间，更新后再 retract 回流形。
        """
        # 保存当前参数并投影梯度
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    if id(p) not in self.state:
                        self.state[id(p)] = p.data.clone()
                    else:
                        self.state[id(p)].fill_(0).add_(p.data)
                    
                    # 暂存参数，投影梯度到切空间
                    p.data.fill_(0)
                    trans = orthogonal_projection(p.grad.data, p.data)
                    p.grad.data.fill_(0).add_(trans)
                    
        # 执行标准优化步骤
        loss = self.optimizer.step(closure)

        # 将 Stiefel 参数 retract 回流形
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    trans = retraction(p.data, self.state[id(p)])
                    p.data.fill_(0).add_(trans)

        return loss
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

