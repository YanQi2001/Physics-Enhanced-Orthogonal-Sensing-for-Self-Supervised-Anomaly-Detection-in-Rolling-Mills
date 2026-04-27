"""
阶段一：SPD 流形预训练

基于研究报告第 3 章：正交/振动分支架构 - 基于互谱密度矩阵与SPD网络的流形几何感知

使用 SPDNet 自编码器学习 CSD 矩阵的流形表示。
包含黎曼距离重构损失和流形上的对比学习损失。

对比学习设计（来自研究报告 3.3 节）：
- 正样本对：同一工况下，相邻时间窗口的 CSD 矩阵
- 负样本对：经过"正交性扰动"的 CSD 矩阵
- 损失函数：对数欧氏度量（LEM）或仿射不变黎曼度量（AIRM）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union
from tqdm import tqdm

from models.spd.spd_decoder import SPDAutoEncoder
from models.spd.optimizer import StiefelMetaOptimizer
from losses.riemannian_loss import (
    LogEuclideanDistance,
    SPDContrastiveLoss,
    SPDFrobeniusDistance
)
from .utils import MetricTracker, EarlyStopping, save_checkpoint


class SPDPretrainer:
    """
    SPD 流形预训练器
    
    使用 SPDNet 自编码器在黎曼流形上学习 CSD 矩阵的特征表示。
    支持黎曼距离重构损失 + 流形对比学习损失。
    
    参考研究报告：
    - 3.2 节：SPDNet 流形神经网络（BiMap/ReEig/LogEig）
    - 3.3 节：流形距离度量学习
    
    Args:
        model: SPD 自编码器模型
        device: 训练设备
        learning_rate: 学习率
        weight_decay: 权重衰减
        use_stiefel_optimizer: 是否使用 Stiefel 流形优化器
        distance_metric: 距离度量类型 ('lem', 'airm', 'frobenius')
        lambda_contrast: 对比学习损失权重
        margin: 对比学习边界
        warmup_epochs: 使用 Frobenius 损失的预热轮数（提高数值稳定性）
        eps: 数值稳定性参数
    """
    
    def __init__(
        self,
        model: SPDAutoEncoder,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_stiefel_optimizer: bool = False,
        distance_metric: str = 'lem',
        lambda_contrast: float = 0.5,
        margin: float = 1.0,
        warmup_epochs: int = 5,
        eps: float = 1e-4
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_contrast = lambda_contrast
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.eps = eps
        
        # 创建标准优化器
        base_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 可选：使用 Stiefel 流形优化器
        if use_stiefel_optimizer:
            self.optimizer = StiefelMetaOptimizer(base_optimizer)
        else:
            self.optimizer = base_optimizer
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            base_optimizer if use_stiefel_optimizer else self.optimizer,
            mode='min', factor=0.5, patience=5
        )
        
        # 损失函数
        self.distance_metric = distance_metric
        
        # 预热阶段使用 Frobenius 距离（更稳定）
        self.frobenius_loss = SPDFrobeniusDistance(reduction='mean')
        
        # 正式阶段使用黎曼距离
        if distance_metric == 'lem':
            self.riemannian_loss = LogEuclideanDistance(eps=eps, reduction='mean')
        else:
            # 默认使用 LEM，AIRM 计算量大且可能不稳定
            self.riemannian_loss = LogEuclideanDistance(eps=eps, reduction='mean')
        
        # 对比学习损失（流形上的）
        self.contrastive_loss = SPDContrastiveLoss(
            metric='lem',
            margin=margin,
            eps=eps
        )
        
        # 指标追踪
        self.metrics = MetricTracker()
    
    def _get_recon_loss(self, M_input: torch.Tensor, M_recon: torch.Tensor) -> torch.Tensor:
        """
        计算重构损失
        
        预热阶段使用 Frobenius 距离，之后使用黎曼距离。
        
        Args:
            M_input: 输入 SPD 矩阵
            M_recon: 重构 SPD 矩阵
            
        Returns:
            重构损失
        """
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：使用 Frobenius 距离（更稳定）
            return self.frobenius_loss(M_input, M_recon)
        else:
            # 正式阶段：使用黎曼距离
            try:
                return self.riemannian_loss(M_input, M_recon)
            except RuntimeError as e:
                # 如果黎曼距离计算失败，回退到 Frobenius
                print(f"Warning: Riemannian loss failed, falling back to Frobenius: {e}")
                return self.frobenius_loss(M_input, M_recon)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch
        
        支持两种数据格式：
        1. 普通格式：dict with 'csd_real' (32x32 实数 SPD 矩阵)
        2. 顺序格式（对比学习）：dict with 'current', 'prev', 'negative_csd_real'
        
        Args:
            dataloader: 数据加载器，提供 CSD 实数矩阵
            
        Returns:
            训练指标
        """
        self.model.train()
        self.metrics.reset()
        
        pbar = tqdm(dataloader, desc=f'SPD Pretrain (Epoch {self.current_epoch + 1})')
        for batch in pbar:
            # 检测数据格式
            is_sequential = isinstance(batch, dict) and 'current' in batch
            
            if is_sequential:
                # 顺序采样格式（支持对比学习）
                M = batch['current']['csd_real'].to(self.device).double()
            elif isinstance(batch, dict):
                M = batch['csd_real'].to(self.device).double()
            else:
                M = batch.to(self.device).double()
            
            # 确保矩阵是对称正定的（添加正则化）
            M = self._regularize_spd(M)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 只使用编码器提取特征（研究报告 3.2 节的核心目标）
            # 不使用解码器重构，因为 ExpEig 层会导致巨大的尺度变化
            z_latent = self.model.encoder(M)
            
            # 主损失：特征空间的正则化（防止特征塌缩）
            # 使用特征的方差作为正则化项，鼓励学习有区分度的特征
            feature_var = z_latent.var(dim=0).mean()  # 跨样本的特征方差
            feature_loss = -torch.log(feature_var + 1e-6)  # 最大化方差
            
            # 对比学习损失（研究报告 3.3 节核心任务）
            contrast_loss = torch.tensor(0.0, device=self.device)
            if is_sequential and self.lambda_contrast > 0:
                try:
                    # 获取正样本（前一时刻）
                    M_prev = batch['prev']['csd_real'].to(self.device).double()
                    M_prev = self._regularize_spd(M_prev)
                    z_prev = self.model.encoder(M_prev)
                    
                    # 获取负样本（正交性扰动后）
                    if 'negative_csd_real' in batch:
                        M_negative = batch['negative_csd_real'].to(self.device).double()
                        M_negative = self._regularize_spd(M_negative)
                        z_negative = self.model.encoder(M_negative)
                        
                        # 欧氏空间中的三元组损失（在切空间中）
                        # 正样本对应该更近，负样本对应该更远
                        d_pos = torch.norm(z_latent - z_prev, dim=1)
                        d_neg = torch.norm(z_latent - z_negative, dim=1)
                        contrast_loss = torch.clamp(d_pos - d_neg + self.contrastive_loss.margin, min=0).mean()
                    else:
                        # 只有正样本：最小化正样本对距离
                        contrast_loss = torch.norm(z_latent - z_prev, dim=1).mean()
                    
                    if torch.isnan(contrast_loss) or torch.isinf(contrast_loss):
                        contrast_loss = torch.tensor(0.0, device=self.device)
                except Exception as e:
                    print(f"Warning: Contrastive loss computation failed: {e}")
                    contrast_loss = torch.tensor(0.0, device=self.device)
            
            # 总损失 = 特征正则化 + 对比学习
            recon_loss = feature_loss  # 为了兼容性保留变量名
            loss = feature_loss + self.lambda_contrast * contrast_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录指标
            self.metrics.update('loss', loss.item(), M.size(0))
            self.metrics.update('recon_loss', recon_loss.item(), M.size(0))
            if is_sequential and self.lambda_contrast > 0:
                self.metrics.update('contrast_loss', contrast_loss.item(), M.size(0))
            
            # 显示详细指标
            postfix = {
                'loss': f"{self.metrics.get('loss'):.4f}",
                'recon': f"{self.metrics.get('recon_loss'):.4f}",
            }
            if is_sequential and self.lambda_contrast > 0:
                contrast_val = self.metrics.get('contrast_loss')
                postfix['contrast'] = f"{contrast_val:.4f}"
            postfix['metric'] = 'Frobenius' if self.current_epoch < self.warmup_epochs else 'LEM'
            pbar.set_postfix(postfix)
        
        return self.metrics.get_all()
    
    def _regularize_spd(self, M: torch.Tensor) -> torch.Tensor:
        """
        正则化 SPD 矩阵，确保正定性和数值稳定性
        
        Args:
            M: 输入矩阵，shape (B, n, n)
            
        Returns:
            正则化后的 SPD 矩阵
        """
        # 确保对称性
        M = (M + M.transpose(-2, -1)) / 2
        
        # 添加正则化项确保正定性
        n = M.shape[-1]
        reg = self.eps * torch.eye(n, device=M.device, dtype=M.dtype)
        M = M + reg
        
        return M
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            验证指标
        """
        self.model.eval()
        self.metrics.reset()
        
        for batch in dataloader:
            # 检测数据格式
            is_sequential = isinstance(batch, dict) and 'current' in batch
            
            if is_sequential:
                M = batch['current']['csd_real'].to(self.device).double()
            elif isinstance(batch, dict):
                M = batch['csd_real'].to(self.device).double()
            else:
                M = batch.to(self.device).double()
            
            # 正则化
            M = self._regularize_spd(M)
            
            M_recon, z_latent = self.model(M)
            
            # 使用 Frobenius 距离进行验证（更稳定）
            loss = self.frobenius_loss(M, M_recon)
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                self.metrics.update('val_loss', loss.item(), M.size(0))
        
        return self.metrics.get_all()
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        checkpoint_dir: str = None,
        patience: int = 20
    ) -> Dict[str, list]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            checkpoint_dir: 检查点保存目录
            patience: 早停耐心值
            
        Returns:
            训练历史
        """
        history = {'train_loss': [], 'val_loss': []}
        early_stopping = EarlyStopping(patience=patience, mode='min')
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 显示训练阶段信息
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                phase = f'Warmup ({epoch+1}/{self.warmup_epochs}) - Frobenius Loss'
            else:
                phase = 'Riemannian (LEM) + Contrastive'
            print(f"\nEpoch {epoch+1}/{epochs} - {phase}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics.get('loss', float('nan')))
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics.get('val_loss', float('nan')))
                
                current_loss = val_metrics.get('val_loss', float('inf'))
                print(f"Train Loss: {train_metrics.get('loss', 0):.6f}, "
                      f"Val Loss: {val_metrics.get('val_loss', 0):.6f}")
                
                # 学习率调度
                self.scheduler.step(current_loss)
            else:
                current_loss = train_metrics.get('loss', float('inf'))
                print(f"Train Loss: {train_metrics.get('loss', 0):.6f}")
                self.scheduler.step(current_loss)
            
            # 保存最佳模型
            if current_loss < best_loss and not torch.isnan(torch.tensor(current_loss)):
                best_loss = current_loss
                if checkpoint_dir:
                    save_checkpoint(
                        f"{checkpoint_dir}/spd_pretrain_best.pt",
                        self.model,
                        self.optimizer,
                        epoch=epoch,
                        best_loss=best_loss
                    )
                    print(f"Checkpoint saved to {checkpoint_dir}/spd_pretrain_best.pt")
            
            # 早停检查
            if early_stopping(current_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def get_encoder(self) -> nn.Module:
        """获取训练好的编码器"""
        return self.model.encoder


# ============================================================
# MLP 版本（作为备选方案）
# ============================================================

class FeaturePretrainer:
    """
    CSD 特征预训练器（MLP 版本）
    
    使用 MLP 自编码器学习 CSD 向量的特征表示。
    作为 SPDNet 的简化替代方案。
    
    Args:
        model: MLP 自编码器模型
        device: 训练设备
        learning_rate: 学习率
        weight_decay: 权重衰减
        lambda_contrast: 对比学习损失权重
        temperature: 对比学习温度参数
    """
    
    def __init__(
        self,
        model,  # MLPAutoEncoder
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_contrast: float = 0.5,
        temperature: float = 0.1
    ):
        from models.mlp.mlp_encoder import MLPAutoEncoder
        from losses.physics_loss import ContrastiveLoss
        
        self.model = model.to(device)
        self.device = device
        self.lambda_contrast = lambda_contrast
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 损失函数
        self.recon_criterion = nn.MSELoss()
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        
        # 指标追踪
        self.metrics = MetricTracker()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        self.metrics.reset()
        
        pbar = tqdm(dataloader, desc='Feature Pretrain')
        for batch in pbar:
            is_sequential = isinstance(batch, dict) and 'current' in batch
            
            if is_sequential:
                v = batch['current']['csd_vector'].to(self.device).float()
            elif isinstance(batch, dict):
                v = batch['csd_vector'].to(self.device).float()
            else:
                v = batch.to(self.device).float()
            
            self.optimizer.zero_grad()
            v_recon, z_anchor = self.model(v)
            
            recon_loss = self.recon_criterion(v_recon, v)
            
            contrast_loss = torch.tensor(0.0, device=self.device)
            if is_sequential and self.lambda_contrast > 0:
                v_prev = batch['prev']['csd_vector'].to(self.device).float()
                _, z_positive = self.model(v_prev)
                
                if 'negative_csd_vector' in batch and batch['negative_csd_vector'] is not None:
                    v_negative = batch['negative_csd_vector'].to(self.device).float()
                    _, z_negative = self.model(v_negative)
                    contrast_loss = self.contrastive_loss(z_anchor, z_positive, z_negative)
                else:
                    contrast_loss = self.contrastive_loss(z_anchor, z_positive)
            
            loss = recon_loss + self.lambda_contrast * contrast_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            self.metrics.update('loss', loss.item(), v.size(0))
            self.metrics.update('recon_loss', recon_loss.item(), v.size(0))
            if is_sequential:
                self.metrics.update('contrast_loss', contrast_loss.item(), v.size(0))
            
            pbar.set_postfix({
                'loss': f"{self.metrics.get('loss'):.6f}",
                'recon': f"{self.metrics.get('recon_loss'):.6f}"
            })
        
        return self.metrics.get_all()
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        self.metrics.reset()
        
        for batch in dataloader:
            is_sequential = isinstance(batch, dict) and 'current' in batch
            
            if is_sequential:
                v = batch['current']['csd_vector'].to(self.device).float()
            elif isinstance(batch, dict):
                v = batch['csd_vector'].to(self.device).float()
            else:
                v = batch.to(self.device).float()
            
            v_recon, z_latent = self.model(v)
            loss = self.recon_criterion(v_recon, v)
            
            self.metrics.update('val_loss', loss.item(), v.size(0))
        
        return self.metrics.get_all()
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        checkpoint_dir: str = None,
        patience: int = 20
    ) -> Dict[str, list]:
        """完整训练流程"""
        history = {'train_loss': [], 'val_loss': []}
        early_stopping = EarlyStopping(patience=patience, mode='min')
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                
                current_loss = val_metrics['val_loss']
                print(f"Train Loss: {train_metrics['loss']:.6f}, Val Loss: {val_metrics['val_loss']:.6f}")
                
                self.scheduler.step(current_loss)
            else:
                current_loss = train_metrics['loss']
                print(f"Train Loss: {train_metrics['loss']:.6f}")
                self.scheduler.step(current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                if checkpoint_dir:
                    save_checkpoint(
                        f"{checkpoint_dir}/feature_pretrain_best.pt",
                        self.model,
                        self.optimizer,
                        epoch=epoch,
                        best_loss=best_loss
                    )
                    print(f"Checkpoint saved to {checkpoint_dir}/feature_pretrain_best.pt")
            
            if early_stopping(current_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def get_encoder(self) -> nn.Module:
        """获取训练好的编码器"""
        return self.model.encoder
