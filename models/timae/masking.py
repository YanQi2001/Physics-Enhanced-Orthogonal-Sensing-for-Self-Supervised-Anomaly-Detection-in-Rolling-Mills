"""
多尺度遮蔽器

实现点状遮蔽（去噪）和块状遮蔽（结构推理）的混合策略。
"""

import torch
import numpy as np
from typing import Tuple


class MultiScaleMasker:
    """
    多尺度混合遮蔽器
    
    融合两种遮蔽策略：
    - 点状遮蔽（Point Mask）：随机打孔，学习邻域插值能力（去噪）
    - 块状遮蔽（Block Mask）：连续遮蔽整块，学习结构推断能力（语义推理）
    
    两种遮蔽通过 Logical OR 融合。
    
    Args:
        point_ratio: 点状遮蔽比例（0-1）
        block_ratio: 块状遮蔽比例（0-1），指的是被块遮蔽覆盖的总比例
        block_size: 每个块的大小
        min_mask_ratio: 最小总遮蔽比例
        max_mask_ratio: 最大总遮蔽比例
    """
    
    def __init__(
        self,
        point_ratio: float = 0.5,
        block_ratio: float = 0.2,
        block_size: int = 4,
        min_mask_ratio: float = 0.3,
        max_mask_ratio: float = 0.8
    ):
        self.point_ratio = point_ratio
        self.block_ratio = block_ratio
        self.block_size = block_size
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
    
    def generate_point_mask(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        生成点状遮蔽
        
        随机选取 point_ratio 比例的位置进行遮蔽。
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            布尔遮蔽张量，True表示被遮蔽，shape (seq_len,)
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        num_mask = int(seq_len * self.point_ratio)
        
        if num_mask > 0:
            indices = torch.randperm(seq_len, device=device)[:num_mask]
            mask[indices] = True
        
        return mask
    
    def generate_block_mask(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        生成块状遮蔽
        
        随机选择起点，连续遮蔽 block_size 个位置。
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            布尔遮蔽张量，True表示被遮蔽，shape (seq_len,)
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        
        # 计算需要多少个块
        total_mask_positions = int(seq_len * self.block_ratio)
        num_blocks = max(1, total_mask_positions // self.block_size)
        
        # 可能的起点位置
        max_start = seq_len - self.block_size
        if max_start <= 0:
            # 序列太短，直接全部遮蔽
            mask[:] = True
            return mask
        
        # 随机选择块的起点
        for _ in range(num_blocks):
            start = np.random.randint(0, max_start + 1)
            end = min(start + self.block_size, seq_len)
            mask[start:end] = True
        
        return mask
    
    def __call__(
        self, 
        seq_len: int, 
        batch_size: int = 1,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        生成混合遮蔽（Logical OR）
        
        Args:
            seq_len: 序列长度
            batch_size: 批大小
            device: 设备
            
        Returns:
            遮蔽张量，True表示被遮蔽，shape (batch_size, seq_len)
        """
        masks = []
        
        for _ in range(batch_size):
            point_mask = self.generate_point_mask(seq_len, device)
            block_mask = self.generate_block_mask(seq_len, device)
            
            # Logical OR 融合
            combined_mask = point_mask | block_mask
            
            # 确保遮蔽比例在合理范围内
            mask_ratio = combined_mask.float().mean().item()
            
            if mask_ratio < self.min_mask_ratio:
                # 补充更多点状遮蔽
                additional_needed = int((self.min_mask_ratio - mask_ratio) * seq_len)
                unmasked_indices = torch.where(~combined_mask)[0]
                if len(unmasked_indices) > additional_needed:
                    perm = torch.randperm(len(unmasked_indices))[:additional_needed]
                    combined_mask[unmasked_indices[perm]] = True
            elif mask_ratio > self.max_mask_ratio:
                # 减少遮蔽
                excess = int((mask_ratio - self.max_mask_ratio) * seq_len)
                masked_indices = torch.where(combined_mask)[0]
                if len(masked_indices) > excess:
                    perm = torch.randperm(len(masked_indices))[:excess]
                    combined_mask[masked_indices[perm]] = False
            
            masks.append(combined_mask)
        
        return torch.stack(masks, dim=0)
    
    def get_visible_and_masked_indices(
        self, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取可见和被遮蔽的索引
        
        Args:
            mask: 遮蔽张量，shape (B, L)
            
        Returns:
            (visible_indices, masked_indices) 每个shape (B, num_visible/num_masked)
        """
        batch_size, seq_len = mask.shape
        
        visible_indices = []
        masked_indices = []
        
        for b in range(batch_size):
            vis_idx = torch.where(~mask[b])[0]
            mask_idx = torch.where(mask[b])[0]
            visible_indices.append(vis_idx)
            masked_indices.append(mask_idx)
        
        return visible_indices, masked_indices


class AdaptiveMultiScaleMasker(MultiScaleMasker):
    """
    自适应多尺度遮蔽器
    
    根据信号特征动态调整遮蔽策略。
    
    Args:
        base_point_ratio: 基础点状遮蔽比例
        base_block_ratio: 基础块状遮蔽比例
        block_size: 块大小
        adapt_to_variance: 是否根据方差调整遮蔽
    """
    
    def __init__(
        self,
        base_point_ratio: float = 0.5,
        base_block_ratio: float = 0.2,
        block_size: int = 4,
        adapt_to_variance: bool = True
    ):
        super().__init__(
            point_ratio=base_point_ratio,
            block_ratio=base_block_ratio,
            block_size=block_size
        )
        self.adapt_to_variance = adapt_to_variance
    
    def adaptive_mask(
        self,
        x: torch.Tensor,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        根据信号特征生成自适应遮蔽
        
        Args:
            x: 输入信号，shape (B, T) 或 (B, T, C)
            device: 设备
            
        Returns:
            遮蔽张量，shape (B, T)
        """
        if x.ndim == 3:
            x = x.mean(dim=-1)  # 多通道取平均
        
        batch_size, seq_len = x.shape
        
        if self.adapt_to_variance:
            # 计算局部方差
            window_size = min(self.block_size * 2, seq_len // 4)
            if window_size > 0:
                # 使用滑动窗口计算局部方差
                local_var = torch.zeros_like(x)
                for i in range(seq_len):
                    start = max(0, i - window_size // 2)
                    end = min(seq_len, i + window_size // 2)
                    local_var[:, i] = x[:, start:end].var(dim=1)
                
                # 方差高的区域减少遮蔽（保留更多跳变信息）
                var_normalized = (local_var - local_var.min(dim=1, keepdim=True)[0]) / \
                                (local_var.max(dim=1, keepdim=True)[0] - local_var.min(dim=1, keepdim=True)[0] + 1e-8)
                
                # 调整遮蔽概率：方差高的地方遮蔽概率低
                mask_prob = self.point_ratio * (1 - 0.5 * var_normalized)
                
                # 按概率生成遮蔽
                masks = torch.rand_like(mask_prob) < mask_prob
            else:
                masks = self(seq_len, batch_size, device)
        else:
            masks = self(seq_len, batch_size, device)
        
        return masks

