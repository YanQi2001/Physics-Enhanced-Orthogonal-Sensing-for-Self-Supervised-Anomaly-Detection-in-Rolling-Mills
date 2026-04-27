"""
阶段 1.5：Ti-MAE 工况预训练

使用多尺度遮蔽和平滑约束训练压力信号编码器。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from models.timae.timae import TiMAEWithPhysicsLoss
from .utils import MetricTracker, EarlyStopping, save_checkpoint


class TiMAEPretrainer:
    """
    Ti-MAE 工况预训练器
    
    训练目标：从噪声压力信号中提取干净的阶梯状工况特征。
    
    损失函数：L_total = L_recon + λ_smooth * L_smooth
    
    Args:
        model: Ti-MAE 模型
        device: 训练设备
        learning_rate: 学习率
        weight_decay: 权重衰减
        lambda_smooth: 平滑约束权重
    """
    
    def __init__(
        self,
        model: TiMAEWithPhysicsLoss,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_smooth: float = 5.0
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_smooth = lambda_smooth
        
        # 更新模型中的 lambda_smooth
        self.model.lambda_smooth = lambda_smooth
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # 指标追踪
        self.metrics = MetricTracker()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            dataloader: 数据加载器，提供压力信号
            
        Returns:
            训练指标
        """
        self.model.train()
        self.metrics.reset()
        
        pbar = tqdm(dataloader, desc='Ti-MAE Pretrain')
        for batch in pbar:
            # 获取压力数据
            if isinstance(batch, dict):
                pressure = batch['pressure'].to(self.device)
            else:
                pressure = batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(pressure, return_losses=True)
            
            # 获取损失
            loss_total = output['l_total']
            loss_recon = output['l_recon']
            loss_smooth = output['l_smooth']
            
            # 反向传播
            loss_total.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录指标
            batch_size = pressure.size(0)
            self.metrics.update('loss', loss_total.item(), batch_size)
            self.metrics.update('recon_loss', loss_recon.item(), batch_size)
            self.metrics.update('smooth_loss', loss_smooth.item(), batch_size)
            
            pbar.set_postfix({
                'loss': self.metrics.get('loss'),
                'recon': self.metrics.get('recon_loss'),
                'smooth': self.metrics.get('smooth_loss')
            })
        
        return self.metrics.get_all()
    
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
            if isinstance(batch, dict):
                pressure = batch['pressure'].to(self.device)
            else:
                pressure = batch.to(self.device)
            
            output = self.model(pressure, return_losses=True)
            
            batch_size = pressure.size(0)
            self.metrics.update('val_loss', output['l_total'].item(), batch_size)
            self.metrics.update('val_recon', output['l_recon'].item(), batch_size)
            self.metrics.update('val_smooth', output['l_smooth'].item(), batch_size)
        
        return self.metrics.get_all()
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        checkpoint_dir: str = None,
        patience: int = 15
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
        history = {
            'train_loss': [], 'train_recon': [], 'train_smooth': [],
            'val_loss': [], 'val_recon': [], 'val_smooth': []
        }
        early_stopping = EarlyStopping(patience=patience, mode='min')
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_recon'].append(train_metrics['recon_loss'])
            history['train_smooth'].append(train_metrics['smooth_loss'])
            
            # 更新学习率
            self.scheduler.step()
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_recon'].append(val_metrics['val_recon'])
                history['val_smooth'].append(val_metrics['val_smooth'])
                
                current_loss = val_metrics['val_loss']
                print(f"Train Loss: {train_metrics['loss']:.4f} "
                      f"(recon: {train_metrics['recon_loss']:.4f}, smooth: {train_metrics['smooth_loss']:.4f})")
                print(f"Val Loss: {val_metrics['val_loss']:.4f} "
                      f"(recon: {val_metrics['val_recon']:.4f}, smooth: {val_metrics['val_smooth']:.4f})")
            else:
                current_loss = train_metrics['loss']
                print(f"Train Loss: {train_metrics['loss']:.4f} "
                      f"(recon: {train_metrics['recon_loss']:.4f}, smooth: {train_metrics['smooth_loss']:.4f})")
            
            # 保存最佳模型
            if current_loss < best_loss:
                best_loss = current_loss
                if checkpoint_dir:
                    save_checkpoint(
                        f"{checkpoint_dir}/timae_pretrain_best.pt",
                        self.model,
                        self.optimizer,
                        epoch=epoch,
                        best_loss=best_loss
                    )
            
            # 早停检查
            if early_stopping(current_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def get_context_encoder(self):
        """获取训练好的工况编码器"""
        return self.model.timae
    
    @torch.no_grad()
    def visualize_reconstruction(
        self, 
        dataloader: DataLoader, 
        num_samples: int = 5
    ):
        """
        可视化重构效果
        
        Args:
            dataloader: 数据加载器
            num_samples: 展示样本数
        """
        self.model.eval()
        
        batch = next(iter(dataloader))
        if isinstance(batch, dict):
            pressure = batch['pressure'][:num_samples].to(self.device)
        else:
            pressure = batch[:num_samples].to(self.device)
        
        output = self.model(pressure, return_losses=False)
        recon = output['recon']
        
        # 返回原始和重构数据用于绘图
        return {
            'original': pressure.cpu().numpy(),
            'reconstructed': recon.cpu().numpy(),
            'mask': output['mask'].cpu().numpy()
        }

