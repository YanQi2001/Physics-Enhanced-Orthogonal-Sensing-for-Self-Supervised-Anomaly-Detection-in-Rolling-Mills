"""
阶段 1.5：VQ-VAE 工况预训练

使用 VQ-VAE 训练压力信号编码器，学习离散的工况状态表示。

替代原有的 Ti-MAE 预训练，提供更清晰的工况分类能力。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import os

from models.vqvae import DualChannelVQVAE, VQVAEWithPhysicsLoss
from .utils import MetricTracker, EarlyStopping, save_checkpoint


class VQVAEPretrainer:
    """
    VQ-VAE 工况预训练器
    
    训练目标：
    1. 学习 K 种典型工况的码本表示（Codebook）
    2. 让编码器输出接近最近的码本向量
    3. 可选：辅助重构监督
    
    损失函数：
    L = VQ_Loss + λ_recon * Recon_Loss
    
    其中 VQ_Loss = VQ_Loss + β * Commitment_Loss
    
    Args:
        model: VQ-VAE 模型 (DualChannelVQVAE 或 VQVAEWithPhysicsLoss)
        device: 训练设备
        learning_rate: 学习率
        weight_decay: 权重衰减
        lambda_recon: 重构损失权重
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_recon: float = 0.1
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_recon = lambda_recon
        
        # 优化器
        # 注意：EMA 更新的码本不参与梯度优化
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
            训练指标字典
        """
        self.model.train()
        self.metrics.reset()
        
        pbar = tqdm(dataloader, desc='VQ-VAE Pretrain')
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
            vq_loss = output.get('vq_loss', output.get('l_smooth', torch.tensor(0.0)))
            recon_loss = output.get('l_recon', torch.tensor(0.0))
            
            # 反向传播
            loss_total.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录指标
            batch_size = pressure.size(0)
            self.metrics.update('loss', loss_total.item(), batch_size)
            self.metrics.update('vq_loss', vq_loss.item() if torch.is_tensor(vq_loss) else vq_loss, batch_size)
            self.metrics.update('recon_loss', recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss, batch_size)
            
            # 记录码本使用情况
            if hasattr(self.model, 'get_codebook_usage'):
                usage = self.model.get_codebook_usage()
                active_codes = (usage > 0.01).sum().item()
                self.metrics.update('active_codes', active_codes, 1)
            elif hasattr(self.model, 'vqvae') and hasattr(self.model.vqvae, 'get_codebook_usage'):
                usage = self.model.vqvae.get_codebook_usage()
                active_codes = (usage > 0.01).sum().item()
                self.metrics.update('active_codes', active_codes, 1)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': self.metrics.get('loss'),
                'vq': self.metrics.get('vq_loss'),
                'recon': self.metrics.get('recon_loss'),
                'codes': int(self.metrics.get('active_codes')) if 'active_codes' in self.metrics.values else 0
            })
        
        return self.metrics.get_all()
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            验证指标字典
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
            
            vq_loss = output.get('vq_loss', output.get('l_smooth', torch.tensor(0.0)))
            recon_loss = output.get('l_recon', torch.tensor(0.0))
            self.metrics.update('val_vq', vq_loss.item() if torch.is_tensor(vq_loss) else vq_loss, batch_size)
            self.metrics.update('val_recon', recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss, batch_size)
        
        return self.metrics.get_all()
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
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
            'train_loss': [], 'train_vq': [], 'train_recon': [],
            'val_loss': [], 'val_vq': [], 'val_recon': [],
            'active_codes': []
        }
        early_stopping = EarlyStopping(patience=patience, mode='min')
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_vq'].append(train_metrics['vq_loss'])
            history['train_recon'].append(train_metrics['recon_loss'])
            
            if 'active_codes' in train_metrics:
                history['active_codes'].append(train_metrics['active_codes'])
            
            # 更新学习率
            self.scheduler.step()
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_vq'].append(val_metrics['val_vq'])
                history['val_recon'].append(val_metrics['val_recon'])
                
                current_loss = val_metrics['val_loss']
                print(f"Train Loss: {train_metrics['loss']:.4f} "
                      f"(vq: {train_metrics['vq_loss']:.4f}, recon: {train_metrics['recon_loss']:.4f})")
                print(f"Val Loss: {val_metrics['val_loss']:.4f} "
                      f"(vq: {val_metrics['val_vq']:.4f}, recon: {val_metrics['val_recon']:.4f})")
                
                if 'active_codes' in train_metrics:
                    print(f"Active Codes: {int(train_metrics['active_codes'])}")
            else:
                current_loss = train_metrics['loss']
                print(f"Train Loss: {train_metrics['loss']:.4f} "
                      f"(vq: {train_metrics['vq_loss']:.4f}, recon: {train_metrics['recon_loss']:.4f})")
            
            # 保存最佳模型
            if current_loss < best_loss:
                best_loss = current_loss
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    save_checkpoint(
                        f"{checkpoint_dir}/vqvae_pretrain_best.pt",
                        self.model,
                        self.optimizer,
                        epoch=epoch,
                        best_loss=best_loss
                    )
            
            # 早停检查
            if early_stopping(current_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 保存最终模型
        if checkpoint_dir:
            save_checkpoint(
                f"{checkpoint_dir}/vqvae_pretrain_final.pt",
                self.model,
                self.optimizer,
                epoch=epochs-1,
                best_loss=best_loss
            )
        
        return history
    
    def get_context_encoder(self):
        """
        获取训练好的工况编码器
        
        Returns:
            VQ-VAE 模型
        """
        if hasattr(self.model, 'vqvae'):
            return self.model.vqvae
        return self.model
    
    @torch.no_grad()
    def visualize_codebook_usage(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        可视化码本使用情况
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            包含码本使用统计的字典
        """
        self.model.eval()
        
        all_indices = []
        all_stats = []
        
        for batch in dataloader:
            if isinstance(batch, dict):
                pressure = batch['pressure'].to(self.device)
            else:
                pressure = batch.to(self.device)
            
            output = self.model(pressure, return_losses=False)
            all_indices.append(output['indices'].cpu())
            if 'stats' in output:
                all_stats.append(output['stats'].cpu())
        
        all_indices = torch.cat(all_indices, dim=0)
        
        # 统计每个码本的使用频率
        if hasattr(self.model, 'vqvae'):
            n_embeddings = self.model.vqvae.n_embeddings
        elif hasattr(self.model, 'n_embeddings'):
            n_embeddings = self.model.n_embeddings
        else:
            n_embeddings = 16
        
        usage_counts = torch.bincount(all_indices, minlength=n_embeddings)
        usage_ratio = usage_counts.float() / usage_counts.sum()
        
        result = {
            'indices': all_indices,
            'usage_counts': usage_counts,
            'usage_ratio': usage_ratio,
        }
        
        if all_stats:
            result['stats'] = torch.cat(all_stats, dim=0)
        
        return result

