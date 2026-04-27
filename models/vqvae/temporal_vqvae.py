"""
Temporal VQ-VAE - 保留时序结构的 VQ-VAE

v3.2 架构更新：
1. 保留时序维度，不使用全局池化
2. 每个时间步独立量化，捕捉工况变化
3. 增强型统计聚合：Mean/Std/Max/Min/Diff（解决"平均值模糊跳变"问题）

架构：
- 输入: (B, 1024, 2) 压力信号
- 编码: (B, 1024, 2) -> (B, 128, 16) 保留16个时间步
- 量化: 每个时间步独立量化 -> (B, 16) 码本索引序列
- 统计聚合: Mean/Std/Max/Min/Diff -> (B, 5*128+4)
- 融合层: (B, 644) -> (B, 128)
- 输出: q_context (B, 128) 保持下游兼容，但内含跳变信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_median_filter(batch_pressure, kernel_size=5):
    """
    对压力信号应用中值滤波，去除尖刺噪声
    
    Args:
        batch_pressure: (B, T, C) 压力信号
        kernel_size: 滤波核大小
    Returns:
        (B, T, C) 滤波后信号
    """
    if kernel_size <= 1:
        return batch_pressure
    
    B, T, C = batch_pressure.shape
    x = batch_pressure.transpose(1, 2)  # (B, C, T)
    pad = kernel_size // 2
    x_padded = F.pad(x, (pad, pad), mode='replicate')
    x_unfold = x_padded.unfold(dimension=2, size=kernel_size, step=1)
    x_filtered, _ = x_unfold.median(dim=-1)
    return x_filtered.transpose(1, 2)


class TemporalEncoder(nn.Module):
    """
    保留时序结构的编码器
    
    自动调整 stride 以匹配目标时间步数
    """
    
    def __init__(self, seq_len=1024, in_channels=2, d_model=128, n_temporal=16):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_temporal = n_temporal
        
        # 计算总下采样率
        total_stride = seq_len // n_temporal
        assert seq_len % n_temporal == 0, f"seq_len ({seq_len}) must be divisible by n_temporal ({n_temporal})"
        
        # 动态分配 stride
        if total_stride == 64:  # 16步
            s1, s2, s3 = 4, 4, 4
        elif total_stride == 32:  # 32步
            s1, s2, s3 = 4, 4, 2
        elif total_stride == 16:  # 64步
            s1, s2, s3 = 4, 2, 2
        else:
            # 简单回退策略
            s1 = int(round(total_stride ** (1/3)))
            s2 = int(round((total_stride / s1) ** 0.5))
            s3 = total_stride // (s1 * s2)
        
        self.strides = (s1, s2, s3)
        
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=s1, padding=3),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 2
            nn.Conv1d(32, 64, kernel_size=5, stride=s2, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 3
            nn.Conv1d(64, d_model, kernel_size=3, stride=s3, padding=1),
            nn.ReLU(),
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, n_temporal) * 0.02)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C) -> (B, 1024, 2)
        Returns:
            z_e: (B, D, T') -> (B, 128, 16)
        """
        x = x.transpose(1, 2)  # (B, C, T)
        z_e = self.conv(x)     # (B, D, T')
        
        # 添加位置编码
        z_e = z_e + self.pos_embedding
        
        return z_e


class TemporalDecoder(nn.Module):
    """
    从时序潜在表示重建原始信号
    """
    
    def __init__(self, seq_len=1024, out_channels=2, d_model=128, n_temporal=16):
        super().__init__()
        
        self.seq_len = seq_len
        self.n_temporal = n_temporal
        
        # 计算总上采样率
        total_stride = seq_len // n_temporal
        
        # 与 Encoder 保持一致的 strides (倒序)
        if total_stride == 64:
            s1, s2, s3 = 4, 4, 4
        elif total_stride == 32:
            s1, s2, s3 = 4, 4, 2
        elif total_stride == 16:
            s1, s2, s3 = 4, 2, 2
        else:
            s1 = int(round(total_stride ** (1/3)))
            s2 = int(round((total_stride / s1) ** 0.5))
            s3 = total_stride // (s1 * s2)
        
        self.deconv = nn.Sequential(
            # Layer 1 (对应 Encoder Layer 3)
            nn.ConvTranspose1d(d_model, 64, kernel_size=4, stride=s3, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 2 (对应 Encoder Layer 2)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=s2, padding=0),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 3 (对应 Encoder Layer 1)
            nn.ConvTranspose1d(32, out_channels, kernel_size=4, stride=s1, padding=0),
        )
    
    def forward(self, z_q):
        """
        Args:
            z_q: (B, D, T') -> (B, 128, 16)
        Returns:
            recon: (B, T, C) -> (B, 1024, 2)
        """
        x = self.deconv(z_q)  # (B, C, T)
        return x.transpose(1, 2)  # (B, T, C)


class TemporalVectorQuantizer(nn.Module):
    """
    时序向量量化器 - 对每个时间步独立量化
    
    输入: (B, D, T) - 时序特征
    输出: 
        - z_q: (B, D, T) - 量化后特征
        - indices: (B, T) - 每个时间步的码本索引
    """
    
    def __init__(self, n_embeddings=8, embedding_dim=128, commitment_cost=0.5):
        super().__init__()
        
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 码本
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1.0)
        
        # 使用跟踪
        self.register_buffer('usage_count', torch.zeros(n_embeddings))
    
    def forward(self, z_e):
        """
        Args:
            z_e: (B, D, T) 时序编码器输出
        Returns:
            z_q: (B, D, T) 量化后
            loss: VQ 损失
            indices: (B, T) 码本索引序列
        """
        B, D, T = z_e.shape
        
        # 展平为 (B*T, D) 进行量化
        z_flat = z_e.permute(0, 2, 1).reshape(B * T, D)  # (B*T, D)
        
        # 计算到所有码本的距离
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.T
        )  # (B*T, K)
        
        # 找最近的码本
        indices_flat = torch.argmin(distances, dim=1)  # (B*T,)
        z_q_flat = self.embedding(indices_flat)  # (B*T, D)
        
        # 更新使用统计
        if self.training:
            for idx in indices_flat:
                self.usage_count[idx] += 1
        
        # 计算损失
        vq_loss = F.mse_loss(z_q_flat, z_flat.detach())
        commit_loss = F.mse_loss(z_flat, z_q_flat.detach())
        loss = vq_loss + self.commitment_cost * commit_loss
        
        # Straight-Through Estimator
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        
        # 恢复形状
        z_q = z_q_flat.reshape(B, T, D).permute(0, 2, 1)  # (B, D, T)
        indices = indices_flat.reshape(B, T)  # (B, T)
        
        return z_q, loss, indices
    
    def get_usage(self):
        total = self.usage_count.sum()
        if total > 0:
            return self.usage_count / total
        return torch.zeros_like(self.usage_count)
    
    def reset_usage(self):
        self.usage_count.zero_()


class TemporalVQVAE(nn.Module):
    """
    Temporal VQ-VAE - 保留时序结构的 VQ-VAE
    
    特点：
    1. 编码器输出 (B, 128, 16) 保留 16 个时间步
    2. 每个时间步独立量化，indices 形状为 (B, 16)
    3. 可以观察工况随时间的变化轨迹
    4. q_context 通过增强统计聚合保持 (B, 128) 兼容下游
    
    v3.2 更新：增强型统计聚合
    - Mean: 平均工况特征
    - Std: 工况波动程度（跳变检测关键！）
    - Max/Min: 极值状态
    - Diff: 一阶差分（变化速率）
    """
    
    def __init__(
        self,
        seq_len=1024,
        in_channels=2,
        d_model=128,
        n_embeddings=8,
        n_temporal=16,
        commitment_cost=0.5,
        use_decoder=True,
        median_kernel=0
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_embeddings = n_embeddings
        self.n_temporal = n_temporal
        self.use_decoder = use_decoder
        self.median_kernel = median_kernel
        
        # 数据统计量
        self.register_buffer('data_mean', torch.zeros(in_channels))
        self.register_buffer('data_std', torch.ones(in_channels))
        
        # 编码器 (保留时序)
        self.encoder = TemporalEncoder(seq_len, in_channels, d_model, n_temporal)
        
        # 量化器 (时序量化)
        self.quantizer = TemporalVectorQuantizer(n_embeddings, d_model, commitment_cost)
        
        # 解码器
        if use_decoder:
            self.decoder = TemporalDecoder(seq_len, in_channels, d_model, n_temporal)
        
        # 增强型统计聚合
        # 输入: 5 * d_model + 2 * in_channels = 5 * 128 + 4 = 644
        enhanced_input_dim = 5 * d_model + 2 * in_channels
        
        self.stat_fusion = nn.Sequential(
            nn.Linear(enhanced_input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 输出维度（兼容性）
        self.output_dim = d_model
    
    def set_data_stats(self, mean, std):
        """设置数据统计量用于标准化"""
        self.data_mean = mean.to(self.data_mean.device)
        self.data_std = std.to(self.data_std.device)
    
    def forward(self, x, return_losses=True):
        """
        Args:
            x: (B, T, C) 压力信号 (B, 1024, 2)
        Returns:
            dict with:
                - q_context: (B, 128) 下游兼容的上下文向量
                - indices: (B, 16) 时序码本索引
                - z_e: (B, 128, 16) 编码器输出
                - z_q: (B, 128, 16) 量化后
                - vq_loss, l_recon, l_total
        """
        B = x.size(0)
        
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        
        # 中值滤波
        if self.median_kernel > 0:
            x = apply_median_filter(x, kernel_size=self.median_kernel)
        
        # 计算原始统计量 (用于 stat_fusion)
        stats = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)  # (B, 2C)
        
        # 标准化
        x_norm = (x - self.data_mean) / (self.data_std + 1e-5)
        
        # 编码 (保留时序)
        z_e = self.encoder(x_norm)  # (B, D, T)
        
        # 量化 (每个时间步)
        z_q, vq_loss, indices = self.quantizer(z_e)  # z_q: (B, D, T), indices: (B, T)
        
        # 增强型统计聚合
        z_mean = z_q.mean(dim=-1)  # (B, D)
        z_std = z_q.std(dim=-1)    # (B, D)
        z_max, _ = z_q.max(dim=-1) # (B, D)
        z_min, _ = z_q.min(dim=-1) # (B, D)
        z_diff = (z_q[:, :, 1:] - z_q[:, :, :-1]).abs().mean(dim=-1)  # (B, D)
        
        # 拼接所有特征
        z_combined = torch.cat([z_mean, z_std, z_max, z_min, z_diff, stats], dim=-1)
        
        # 融合生成 q_context
        q_context = self.stat_fusion(z_combined)  # (B, D)
        
        result = {
            'q_context': q_context,
            'z_e': z_e,
            'z_q': z_q,
            'indices': indices,
            'stats': stats,
            'vq_loss': vq_loss,
            'z_mean': z_mean,
            'z_std': z_std,
            'z_max': z_max,
            'z_min': z_min,
            'z_diff': z_diff,
        }
        
        # 重构
        if return_losses and self.use_decoder:
            recon = self.decoder(z_q)
            recon_loss = F.mse_loss(recon, x_norm)
            result['recon'] = recon
            result['l_recon'] = recon_loss
            result['l_total'] = vq_loss + 0.1 * recon_loss
        
        return result
    
    def get_context(self, x):
        """获取上下文向量（下游接口）"""
        result = self.forward(x, return_losses=False)
        return result['q_context']
    
    def get_codebook_usage(self):
        """获取码本使用统计"""
        return self.quantizer.get_usage()


class TemporalVQVAEWithPhysicsLoss(nn.Module):
    """
    带物理约束的 Temporal VQ-VAE 包装类
    
    与 Ti-MAE 接口兼容，用于 full_model.py
    """
    
    def __init__(
        self,
        seq_len=1024,
        in_channels=2,
        d_model=128,
        n_embeddings=8,
        n_temporal=16,
        commitment_cost=0.5,
        lambda_recon=0.1,
        dropout=0.1,
        median_kernel=0
    ):
        super().__init__()
        
        self.vqvae = TemporalVQVAE(
            seq_len=seq_len,
            in_channels=in_channels,
            d_model=d_model,
            n_embeddings=n_embeddings,
            n_temporal=n_temporal,
            commitment_cost=commitment_cost,
            use_decoder=True,
            median_kernel=median_kernel
        )
        
        self.lambda_recon = lambda_recon
        self.output_dim = d_model
    
    def forward(self, x, return_losses=True):
        """
        与 Ti-MAE 接口兼容的前向传播
        """
        result = self.vqvae(x, return_losses=return_losses)
        
        # 转换为 Ti-MAE 兼容的输出格式
        output = {
            'recon': result.get('recon', x),
            'mask': torch.zeros(x.size(0), x.size(1), device=x.device),  # 占位
            'l_recon': result.get('l_recon', torch.tensor(0.0)),
            'l_smooth': result.get('vq_loss', torch.tensor(0.0)),  # VQ loss 作为平滑约束
        }
        
        return output
    
    def get_context(self, x):
        """获取上下文向量"""
        return self.vqvae.get_context(x)
    
    def set_data_stats(self, mean, std):
        """设置数据统计量"""
        self.vqvae.set_data_stats(mean, std)
    
    def get_codebook_usage(self):
        """获取码本使用统计"""
        return self.vqvae.get_codebook_usage()


