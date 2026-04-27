"""
训练工具函数
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime


class EarlyStopping:
    """
    早停机制
    
    Args:
        patience: 容忍的 epoch 数
        min_delta: 最小改善量
        mode: 'min' 或 'max'
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该停止
        
        Args:
            score: 当前分数
            
        Returns:
            是否应该停止
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """重置状态"""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class LRScheduler:
    """
    学习率调度器包装器
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'cosine',
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # 预热：线性增加
            scale = (self.current_epoch + 1) / self.warmup_epochs
        else:
            # 调度
            progress = (self.current_epoch - self.warmup_epochs) / \
                      max(1, self.total_epochs - self.warmup_epochs)
            
            if self.scheduler_type == 'cosine':
                scale = 0.5 * (1 + np.cos(np.pi * progress))
            elif self.scheduler_type == 'linear':
                scale = 1 - progress
            else:
                scale = 1.0
        
        for i, pg in enumerate(self.optimizer.param_groups):
            new_lr = max(self.min_lr, self.base_lrs[i] * scale)
            pg['lr'] = new_lr
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return [pg['lr'] for pg in self.optimizer.param_groups]


class TrainingLogger:
    """
    训练日志记录器
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str = None
    ):
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.history: Dict[str, List] = {}
        self.current_epoch = 0
    
    def log(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            epoch: epoch 编号
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.history[name].append(value)
        
        self._save_history()
    
    def _save_history(self):
        """保存历史到文件"""
        path = os.path.join(self.log_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_best(self, metric: str, mode: str = 'min') -> tuple:
        """
        获取最佳值
        
        Returns:
            (best_value, best_epoch)
        """
        if metric not in self.history:
            return None, None
        
        values = self.history[metric]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return values[best_idx], best_idx
    
    def print_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        """打印 epoch 摘要"""
        msg = f"Epoch {epoch}: "
        msg += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(msg)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_loss: float = float('inf'),
    **kwargs
):
    """
    保存检查点
    
    Args:
        path: 保存路径
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        best_loss: 最佳损失
        **kwargs: 其他需要保存的信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_loss': best_loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict:
    """
    加载检查点
    
    Args:
        path: 检查点路径
        model: 模型
        optimizer: 优化器
        device: 设备
        
    Returns:
        检查点字典
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    return checkpoint


class MetricTracker:
    """
    指标追踪器
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置"""
        self.values = {}
        self.counts = {}
    
    def update(self, name: str, value: float, count: int = 1):
        """更新指标"""
        if name not in self.values:
            self.values[name] = 0
            self.counts[name] = 0
        
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        self.values[name] += value * count
        self.counts[name] += count
    
    def get(self, name: str) -> float:
        """获取平均值"""
        if name not in self.values or self.counts[name] == 0:
            return 0.0
        return self.values[name] / self.counts[name]
    
    def get_all(self) -> Dict[str, float]:
        """获取所有平均值"""
        return {name: self.get(name) for name in self.values}

