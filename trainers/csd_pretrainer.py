"""
阶段一：CSD Pair-Token Transformer 预训练

使用对比学习训练 CSD Transformer 编码器，
学习传感器间的耦合关系特征。

对比学习设计（对应研究报告 3.3 节）：
- 正样本对：同一工况下，相邻时间窗口的 CSD 矩阵
- 负样本对：经过"正交性扰动"的 CSD 矩阵
- 损失函数：InfoNCE（余弦相似度 + 温度参数）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from models.csd_transformer import CSDTransformerEncoder
from .utils import MetricTracker, EarlyStopping, save_checkpoint


class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比学习损失
    
    使用余弦相似度和温度参数控制分布的集中程度。
    
    Args:
        temperature: 温度参数（越小越集中）
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 InfoNCE 损失
        
        Args:
            anchor: 锚点特征 (B, D)
            positive: 正样本特征 (B, D)
            negatives: 负样本特征 (B, D) 或 (B, N, D)
            
        Returns:
            损失值
        """
        # 归一化
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        
        # 正样本相似度
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # (B,)
        
        if negatives is not None:
            negatives = F.normalize(negatives, dim=-1)
            
            if negatives.dim() == 2:
                # 单个负样本 (B, D)
                neg_sim = torch.sum(anchor * negatives, dim=-1) / self.temperature  # (B,)
                # 三元组风格：max(0, neg_sim - pos_sim + margin)
                # 或者 softmax 风格
                logits = torch.stack([pos_sim, neg_sim], dim=-1)  # (B, 2)
                labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
                loss = F.cross_entropy(logits, labels)
            else:
                # 多个负样本 (B, N, D)
                # anchor: (B, D) -> (B, 1, D)
                neg_sim = torch.sum(
                    anchor.unsqueeze(1) * negatives,
                    dim=-1
                ) / self.temperature  # (B, N)
                
                # 拼接正负样本
                logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=-1)  # (B, 1+N)
                labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
                loss = F.cross_entropy(logits, labels)
        else:
            # 仅正样本：最大化相似度
            loss = -pos_sim.mean()
        
        return loss


class TripletLoss(nn.Module):
    """
    三元组损失
    
    d(anchor, positive) < d(anchor, negative) + margin
    
    Args:
        margin: 边界值
        distance: 距离类型 ('cosine' 或 'euclidean')
    """
    
    def __init__(self, margin: float = 1.0, distance: str = 'cosine'):
        super().__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        计算三元组损失
        
        Args:
            anchor: 锚点特征 (B, D)
            positive: 正样本特征 (B, D)
            negative: 负样本特征 (B, D)
            
        Returns:
            损失值
        """
        if self.distance == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            anchor = F.normalize(anchor, dim=-1)
            positive = F.normalize(positive, dim=-1)
            negative = F.normalize(negative, dim=-1)
            
            d_pos = 1 - torch.sum(anchor * positive, dim=-1)
            d_neg = 1 - torch.sum(anchor * negative, dim=-1)
        else:
            # 欧氏距离
            d_pos = torch.norm(anchor - positive, p=2, dim=-1)
            d_neg = torch.norm(anchor - negative, p=2, dim=-1)
        
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0)
        return loss.mean()


class CSDPretrainer:
    """
    CSD Pair-Token Transformer 预训练器
    
    使用对比学习训练编码器，让模型学会：
    1. 拉近相邻时间窗口的 CSD 特征（时间一致性）
    2. 推远正交性扰动后的 CSD 特征（正交性敏感度）
    
    Args:
        model: CSD Transformer 编码器
        device: 训练设备
        learning_rate: 学习率
        weight_decay: 权重衰减
        temperature: InfoNCE 温度参数
        loss_type: 损失类型 ('infonce' 或 'triplet')
        margin: 三元组损失边界（仅 triplet 使用）
    """
    
    def __init__(
        self,
        model: CSDTransformerEncoder,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.1,
        loss_type: str = 'infonce',
        margin: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature
        
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
        
        # 损失函数
        if loss_type == 'infonce':
            self.criterion = InfoNCELoss(temperature=temperature)
        else:
            self.criterion = TripletLoss(margin=margin, distance='cosine')
        
        self.loss_type = loss_type
        
        # 指标追踪
        self.metrics = MetricTracker()
        
        # 当前 epoch
        self.current_epoch = 0
    
    def _get_csd_matrix(self, batch: Dict, key: str = 'csd_matrix') -> torch.Tensor:
        """
        从 batch 中获取 CSD 矩阵
        
        支持多种数据格式：
        1. 普通格式：batch['csd_matrix'] 或 batch['csd_real']
        2. 顺序格式：batch['current']['csd_matrix']
        
        Args:
            batch: 数据批次
            key: 优先使用的键名
            
        Returns:
            CSD 矩阵
        """
        if isinstance(batch, dict):
            if 'current' in batch:
                # SequentialMultiModalDataset 格式
                sample = batch['current']
            else:
                sample = batch
            
            # 优先使用复数矩阵，其次使用实数化矩阵
            if key in sample:
                return sample[key].to(self.device)
            elif 'csd_matrix' in sample:
                return sample['csd_matrix'].to(self.device)
            elif 'csd_real' in sample:
                return sample['csd_real'].to(self.device)
            else:
                raise KeyError(f"Cannot find CSD matrix in batch. Available keys: {sample.keys()}")
        else:
            return batch.to(self.device)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch
        
        支持两种数据格式：
        1. 普通格式：仅使用 batch 内样本作为对比
        2. 顺序格式：使用相邻时间窗口和正交性扰动
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            训练指标
        """
        self.model.train()
        self.metrics.reset()
        
        pbar = tqdm(dataloader, desc=f'CSD Pretrain (Epoch {self.current_epoch + 1})')
        
        for batch in pbar:
            # 检测数据格式
            is_sequential = isinstance(batch, dict) and 'current' in batch
            
            self.optimizer.zero_grad()
            
            if is_sequential:
                # 顺序采样格式：有明确的正负样本
                # 锚点：当前样本
                anchor_csd = self._get_csd_matrix(batch['current'])
                z_anchor = self.model(anchor_csd)
                
                # 正样本：前一时刻
                positive_csd = self._get_csd_matrix(batch['prev'])
                z_positive = self.model(positive_csd)
                
                # 负样本：正交性扰动后
                if 'negative_csd_real' in batch:
                    negative_csd = batch['negative_csd_real'].to(self.device)
                    z_negative = self.model(negative_csd)
                    loss = self.criterion(z_anchor, z_positive, z_negative)
                elif 'negative_csd' in batch:
                    negative_csd = batch['negative_csd'].to(self.device)
                    z_negative = self.model(negative_csd)
                    loss = self.criterion(z_anchor, z_positive, z_negative)
                else:
                    # 只有正样本
                    loss = self.criterion(z_anchor, z_positive)
            else:
                # 普通格式：使用 batch 内对比
                csd = self._get_csd_matrix(batch)
                z = self.model(csd)
                
                # Batch 内对比：相邻样本作为正样本
                B = z.size(0)
                if B > 1:
                    # 循环移位创建正样本对
                    z_anchor = z
                    z_positive = torch.roll(z, shifts=1, dims=0)
                    
                    # 负样本：随机打乱
                    perm = torch.randperm(B, device=self.device)
                    z_negative = z[perm]
                    
                    # 确保负样本不是自己
                    same_mask = (perm == torch.arange(B, device=self.device))
                    if same_mask.any():
                        # 对相同的位置再次打乱
                        z_negative[same_mask] = torch.roll(z_negative[same_mask], shifts=1, dims=0)
                    
                    loss = self.criterion(z_anchor, z_positive, z_negative)
                else:
                    # Batch size = 1，跳过
                    continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录指标
            batch_size = z_anchor.size(0) if is_sequential else z.size(0)
            self.metrics.update('loss', loss.item(), batch_size)
            
            # 计算额外指标
            with torch.no_grad():
                if is_sequential:
                    # 正负样本距离差
                    pos_sim = F.cosine_similarity(z_anchor, z_positive).mean()
                    if 'z_negative' in dir():
                        neg_sim = F.cosine_similarity(z_anchor, z_negative).mean()
                        self.metrics.update('pos_sim', pos_sim.item(), batch_size)
                        self.metrics.update('neg_sim', neg_sim.item(), batch_size)
            
            # 更新进度条
            postfix = {'loss': f"{self.metrics.get('loss'):.4f}"}
            if 'pos_sim' in self.metrics.values:
                postfix['pos'] = f"{self.metrics.get('pos_sim'):.3f}"
                postfix['neg'] = f"{self.metrics.get('neg_sim'):.3f}"
            pbar.set_postfix(postfix)
        
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
        
        all_features = []
        
        for batch in dataloader:
            csd = self._get_csd_matrix(batch)
            z = self.model(csd)
            all_features.append(z)
            
            self.metrics.update('n_samples', csd.size(0), 1)
        
        # 计算特征统计
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            
            # 特征方差（衡量特征多样性）
            feature_var = all_features.var(dim=0).mean()
            self.metrics.update('val_feature_var', feature_var.item(), 1)
            
            # 特征范数
            feature_norm = all_features.norm(dim=-1).mean()
            self.metrics.update('val_feature_norm', feature_norm.item(), 1)
        
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
        history = {'train_loss': [], 'val_feature_var': []}
        early_stopping = EarlyStopping(patience=patience, mode='max')  # 最大化特征方差
        best_var = 0.0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics.get('loss', float('nan')))
            
            print(f"Train Loss: {train_metrics.get('loss', 0):.6f}")
            if 'pos_sim' in train_metrics:
                print(f"  Pos Similarity: {train_metrics['pos_sim']:.4f}, "
                      f"Neg Similarity: {train_metrics['neg_sim']:.4f}")
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_feature_var'].append(val_metrics.get('val_feature_var', 0))
                
                current_var = val_metrics.get('val_feature_var', 0)
                print(f"Val Feature Variance: {current_var:.6f}, "
                      f"Feature Norm: {val_metrics.get('val_feature_norm', 0):.4f}")
                
                # 学习率调度
                self.scheduler.step()
                
                # 保存最佳模型（最大化特征方差）
                if current_var > best_var:
                    best_var = current_var
                    if checkpoint_dir:
                        save_checkpoint(
                            f"{checkpoint_dir}/csd_pretrain_best.pt",
                            self.model,
                            self.optimizer,
                            epoch=epoch,
                            best_loss=current_var  # 这里存的是 var，但保持接口一致
                        )
                        print(f"Checkpoint saved (best var: {best_var:.6f})")
                
                # 早停检查（基于特征方差）
                if early_stopping(current_var):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                self.scheduler.step()
                
                # 没有验证集时，每 10 个 epoch 保存
                if checkpoint_dir and (epoch + 1) % 10 == 0:
                    save_checkpoint(
                        f"{checkpoint_dir}/csd_pretrain_epoch{epoch+1}.pt",
                        self.model,
                        self.optimizer,
                        epoch=epoch
                    )
        
        # 保存最终模型
        if checkpoint_dir:
            save_checkpoint(
                f"{checkpoint_dir}/csd_pretrain_last.pt",
                self.model,
                self.optimizer,
                epoch=self.current_epoch
            )
        
        return history
    
    def get_encoder(self) -> nn.Module:
        """获取训练好的编码器"""
        return self.model

