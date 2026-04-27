"""
物理上下文门控加权

让工况上下文自动决定各项 Loss 的相对重要性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class LossGatingNetwork(nn.Module):
    """
    Loss 门控网络
    
    根据工况上下文动态生成各项 Loss 的权重。
    解决非平稳工况下的 Loss 冲突问题。
    
    例如：
    - 高压稳态：提高 w_synergy 和 w_consistency
    - 咬钢冲击：降低 w_consistency（容忍正交性暂时破坏）
    
    v3.4 更新：
    - 3 层 MLP + LayerNorm，提升对 q_context 的拟合能力
    - 移除零初始化，使用默认 Xavier 初始化，避免梯度陷入初始点
    
    Args:
        d_context: 工况上下文维度
        num_losses: Loss 项数量
        hidden_dim: MLP 隐藏层维度
        w_base: 各 Loss 的基础权重
        w_min: 权重下界（防止塌缩）
        temperature: softmax 温度参数（>1 使分布更平坦，防止赢者通吃坍缩）
        use_entropy_reg: 是否使用熵正则化
    """
    
    def __init__(
        self,
        d_context: int = 128,
        num_losses: int = 2,
        hidden_dim: int = 32,
        w_base: Optional[List[float]] = None,
        w_min: float = 0.10,
        temperature: float = 3.0,
        use_entropy_reg: bool = False
    ):
        super(LossGatingNetwork, self).__init__()
        
        self.num_losses = num_losses
        self.w_min = w_min
        self.temperature = temperature
        self.use_entropy_reg = use_entropy_reg
        
        # v3.3 默认基础权重 [consistency, synergy]
        if w_base is None:
            w_base = [1.0, 0.5]
        
        self.register_buffer('w_base', torch.tensor(w_base[:num_losses]))
        
        # v3.4 更新：3 层 MLP + LayerNorm，增强门控网络拟合能力
        self.mlp = nn.Sequential(
            nn.Linear(d_context, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_losses)
        )
        # v3.4 更新：不再零初始化，使用默认 Xavier 初始化
        # 让网络从一开始就有非零梯度信号
    
    def forward(
        self, 
        q_context: torch.Tensor
    ) -> torch.Tensor:
        """
        生成动态权重（v3.3 修复版）
        
        使用 softmax 归一化，确保权重总和恒定。
        这样门控网络只能调整权重**分配**，不能通过降低所有权重来减少总损失。
        
        w(Q) = softmax(g(Q)) * sum(w_base) + w_min
        
        Args:
            q_context: 工况上下文，shape (B, d_context)
            
        Returns:
            动态权重，shape (B, num_losses)
        """
        logits = self.mlp(q_context)  # (B, num_losses)
        
        # 使用 softmax 归一化，权重总和恒定为 sum(w_base)
        # 这防止了门控网络通过"降低所有权重"来最小化总损失的作弊行为
        # temperature > 1 使 softmax 更平坦，防止赢者通吃坍缩
        total_weight = self.w_base.sum()
        weights = F.softmax(logits / self.temperature, dim=-1) * total_weight
        
        # 加上下界保护，确保每个损失项都有最小权重
        weights = weights + self.w_min
        
        return weights
    
    def get_entropy_regularization(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算熵正则化项
        
        鼓励权重分布均匀，防止某些 Loss 被完全忽略。
        
        Args:
            weights: 动态权重
            
        Returns:
            熵正则化损失（负熵，需要最小化）
        """
        # 归一化权重
        w_norm = weights / weights.sum(dim=-1, keepdim=True)
        
        # 计算熵
        entropy = -(w_norm * torch.log(w_norm + 1e-10)).sum(dim=-1)
        
        # 返回负熵（最大化熵 = 最小化负熵）
        return -entropy.mean()


class GatedLossComputer(nn.Module):
    """
    门控损失计算器
    
    整合门控网络和损失计算。
    
    v3.4 更新：
    - Loss EMA 动态归一化，防止量级失衡
    - warmup alpha 从 0.1 起步，不完全切断门控梯度
    - 可选的门控多样性正则项
    
    Args:
        d_context: 上下文维度
        loss_names: 各 Loss 的名称
        hidden_dim: MLP 隐藏维度
        w_base: 基础权重
        w_min: 权重下界
        warmup_epochs: 预热期数
        temperature: softmax 温度
        diversity_weight: 多样性正则系数（0 表示关闭）
    """
    
    def __init__(
        self,
        d_context: int = 128,
        loss_names: List[str] = ['consistency', 'synergy'],
        hidden_dim: int = 32,
        w_base: Optional[List[float]] = None,
        w_min: float = 0.10,
        warmup_epochs: int = 5,
        temperature: float = 3.0,
        diversity_weight: float = 0.01
    ):
        super(GatedLossComputer, self).__init__()
        
        self.loss_names = loss_names
        self.num_losses = len(loss_names)
        self.warmup_epochs = warmup_epochs
        self.diversity_weight = diversity_weight
        
        # 门控网络
        self.gating = LossGatingNetwork(
            d_context=d_context,
            num_losses=self.num_losses,
            hidden_dim=hidden_dim,
            w_base=w_base,
            w_min=w_min,
            temperature=temperature
        )
        
        # v3.3 默认固定权重 [consistency, synergy]
        if w_base is None:
            w_base = [1.0, 0.5]
        self.register_buffer('fixed_weights', torch.tensor(w_base[:self.num_losses]))
        
        # v3.4 新增：Loss EMA 动态归一化
        # 为每个 loss 项维护一个指数移动平均值，用于归一化 loss 量级
        self.register_buffer('loss_ema', torch.ones(self.num_losses))
        self.ema_decay = 0.95
        self.ema_initialized = False
        
        # 当前 epoch（用于判断是否在预热期）
        self.current_epoch = 0
        
        # v3.4 更新：渐进启用系数从 0.1 起步，确保门控 MLP 从一开始就有梯度
        self.alpha = 0.1
    
    def update_epoch(self, epoch: int, total_epochs: int):
        """
        更新当前 epoch 和渐进系数
        
        v3.4 更新：warmup 期间 alpha = 0.1（非零），确保门控网络始终有梯度。
        
        Args:
            epoch: 当前 epoch
            total_epochs: 总 epoch 数
        """
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # v3.4 修复：warmup 期间保持 alpha=0.1，不完全切断门控梯度
            self.alpha = 0.1
        else:
            # 线性增加：从 0.1 到 1.0
            progress = (epoch - self.warmup_epochs) / max(1, total_epochs - self.warmup_epochs)
            self.alpha = min(1.0, 0.1 + 0.9 * min(1.0, progress * 2))
    
    def get_weights(self, q_context: torch.Tensor) -> torch.Tensor:
        """
        获取当前权重（考虑预热和渐进）
        
        Args:
            q_context: 工况上下文
            
        Returns:
            权重
        """
        if self.alpha <= 0.0:
            # 完全固定权重（不应该发生，v3.4 保底）
            batch_size = q_context.size(0)
            return self.fixed_weights.unsqueeze(0).expand(batch_size, -1)
        elif self.alpha >= 1.0:
            # 完全启用门控
            return self.gating(q_context)
        else:
            # 渐进混合
            gated_weights = self.gating(q_context)
            fixed_weights = self.fixed_weights.unsqueeze(0).expand_as(gated_weights)
            return self.alpha * gated_weights + (1 - self.alpha) * fixed_weights
    
    def _update_loss_ema(self, losses: Dict[str, torch.Tensor]):
        """
        更新 Loss EMA（仅在训练模式下调用）
        
        Args:
            losses: 各项 per-sample 损失
        """
        for i, name in enumerate(self.loss_names):
            if name in losses:
                loss_mean = losses[name].detach().mean()
                if not self.ema_initialized:
                    self.loss_ema[i] = loss_mean
                else:
                    self.loss_ema[i] = self.ema_decay * self.loss_ema[i] + (1 - self.ema_decay) * loss_mean
        self.ema_initialized = True
    
    def forward(
        self,
        q_context: torch.Tensor,
        losses: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算门控加权总损失
        
        v3.4 更新：
        - Loss EMA 归一化：各项 loss 除以其 EMA 均值，消除量级差异
        - per-sample 加权：每个样本的 gating weight 乘以该样本自己的 loss
        - 多样性正则：鼓励不同样本的权重有差异
        
        Args:
            q_context: 工况上下文
            losses: 各项损失字典 {'name': loss_value}，推荐 per-sample (B,)
            
        Returns:
            包含各项加权损失和总损失的字典
        """
        # 更新 Loss EMA（用于归一化）
        if self.training:
            self._update_loss_ema(losses)
        
        # 获取权重
        weights = self.get_weights(q_context)  # (B, num_losses)
        
        # 计算加权损失
        result = {}
        total_loss = 0.0
        
        for i, name in enumerate(self.loss_names):
            if name in losses:
                loss_value = losses[name]
                
                # v3.4 新增：Loss EMA 归一化，消除量级差异
                ema_scale = self.loss_ema[i].clamp(min=1e-6)
                loss_normalized = loss_value / ema_scale
                
                # 如果 loss 是标量（兼容旧代码），使用平均权重
                if loss_normalized.ndim == 0:
                    w = weights[:, i].mean()
                    weighted_loss = w * loss_normalized
                else:
                    # v3.4 核心：per-sample 加权
                    w = weights[:, i]  # (B,)
                    weighted_loss = (w * loss_normalized).mean()
                
                result[f'weighted_{name}'] = weighted_loss
                result[f'weight_{name}'] = weights[:, i].mean()
                total_loss = total_loss + weighted_loss
        
        # v3.4 新增：门控多样性正则
        # 鼓励不同样本的权重有差异，防止门控网络输出常数
        if self.diversity_weight > 0 and self.training:
            diversity_loss = -weights.var(dim=0).mean()  # 最大化 batch 内权重方差
            total_loss = total_loss + self.diversity_weight * diversity_loss
            result['diversity_loss'] = diversity_loss
        
        result['total'] = total_loss
        result['weights'] = weights
        
        return result


class ContextAwareGating(nn.Module):
    """
    更复杂的上下文感知门控
    
    不仅基于工况，还考虑历史状态。
    """
    
    def __init__(
        self,
        d_context: int = 128,
        num_losses: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_lstm: bool = False
    ):
        super(ContextAwareGating, self).__init__()
        
        self.num_losses = num_losses
        self.use_lstm = use_lstm
        
        if use_lstm:
            # 使用 LSTM 考虑历史
            self.lstm = nn.LSTM(
                input_size=d_context,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, num_losses)
        else:
            # 简单 MLP
            layers = []
            prev_dim = d_context
            for _ in range(num_layers - 1):
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, num_losses))
            self.mlp = nn.Sequential(*layers)
        
        # 可学习的基础权重
        self.w_base = nn.Parameter(torch.ones(num_losses))
        self.w_min = 0.05
    
    def forward(
        self,
        q_context: torch.Tensor,
        hidden: Optional[tuple] = None
    ) -> tuple:
        """
        前向传播
        
        Args:
            q_context: 工况上下文，shape (B, T, d_context) 或 (B, d_context)
            hidden: LSTM 隐藏状态
            
        Returns:
            (weights, hidden)
        """
        if self.use_lstm:
            if q_context.ndim == 2:
                q_context = q_context.unsqueeze(1)
            
            output, hidden = self.lstm(q_context, hidden)
            logits = self.fc(output[:, -1])  # 取最后一个时间步
        else:
            if q_context.ndim == 3:
                q_context = q_context[:, -1]  # 取最后一个时间步
            
            logits = self.mlp(q_context)
            hidden = None
        
        weights = torch.sigmoid(logits) * F.softplus(self.w_base) + self.w_min
        
        return weights, hidden

