"""
数据预处理模块
- VirtualChannelExpander: 全通道自适应虚拟扩展（4→16通道）
- CSDMatrixBuilder: 16×16 CSD矩阵构建
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy import signal
from scipy.stats import kurtosis, entropy
import pywt


class VirtualChannelExpander:
    """
    全通道自适应虚拟扩展器
    
    对每个物理传感器通道进行小波包分解，选取Top-K故障敏感频带，
    将4通道扩展为16通道超向量。
    
    Args:
        wavelet: 小波基函数，默认'db4' (Daubechies 4阶小波)
        level: 分解层数，默认3 (产生2^3=8个子频带)
        top_k: 每通道保留的子频带数，默认3
        lambda_weight: 熵在混合指标中的权重，默认1.0
        mode: 静态模式('static')或动态模式('dynamic')
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 3,
        top_k: int = 3,
        lambda_weight: float = 1.0,
        mode: str = 'static'
    ):
        self.wavelet = wavelet
        self.level = level
        self.top_k = top_k
        self.lambda_weight = lambda_weight
        self.mode = mode
        
        # 静态模式下存储的固定频带索引 (每个通道的top-k索引)
        self.static_indices: Optional[List[List[int]]] = None
        
    def wpd_decompose(self, signal_1d: np.ndarray) -> List[np.ndarray]:
        """
        对单通道信号进行小波包分解
        
        Args:
            signal_1d: 1D信号数组，shape (T,)
            
        Returns:
            子频带列表，每个元素是一个子频带信号
        """
        # 创建小波包树
        wp = pywt.WaveletPacket(data=signal_1d, wavelet=self.wavelet, maxlevel=self.level)
        
        # 获取最底层的所有节点（2^level个子频带）
        nodes = [node.path for node in wp.get_level(self.level, 'natural')]
        
        subbands = []
        for node_path in nodes:
            # 获取子频带系数
            coeffs = wp[node_path].data
            subbands.append(coeffs)
            
        return subbands
    
    def _align_length(self, subbands: List[np.ndarray], target_length: int) -> List[np.ndarray]:
        """
        对齐子频带长度到目标长度（使用线性插值）
        
        Args:
            subbands: 子频带列表
            target_length: 目标长度
            
        Returns:
            对齐后的子频带列表
        """
        aligned = []
        for sb in subbands:
            if len(sb) == target_length:
                aligned.append(sb)
            else:
                # 使用线性插值对齐长度
                x_old = np.linspace(0, 1, len(sb))
                x_new = np.linspace(0, 1, target_length)
                sb_aligned = np.interp(x_new, x_old, sb)
                aligned.append(sb_aligned)
        return aligned
    
    def compute_score(self, subband: np.ndarray) -> float:
        """
        计算子频带的故障敏感性得分（峭度 + λ * 熵）
        
        峭度对冲击类故障敏感（如咬钢异常、轴承剥落）
        能量熵对磨损类故障和非平稳退化敏感
        
        Args:
            subband: 子频带信号
            
        Returns:
            混合得分
        """
        # 计算峭度 (Fisher定义，正态分布为0)
        kurt = kurtosis(subband, fisher=True)
        
        # 计算能量熵
        # 先计算归一化能量分布
        energy = subband ** 2
        energy_sum = np.sum(energy) + 1e-10  # 防止除零
        p = energy / energy_sum
        p = p[p > 0]  # 只保留正值用于计算熵
        ent = entropy(p)
        
        # 混合得分
        score = abs(kurt) + self.lambda_weight * ent
        return score
    
    def select_top_k(
        self, 
        subbands: List[np.ndarray], 
        k: int,
        return_indices: bool = False
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        选取Top-K故障敏感频带
        
        Args:
            subbands: 子频带列表
            k: 选取的频带数量
            return_indices: 是否返回索引
            
        Returns:
            (选中的子频带列表, 索引列表)
        """
        # 计算每个子频带的得分
        scores = [self.compute_score(sb) for sb in subbands]
        
        # 获取Top-K索引
        sorted_indices = np.argsort(scores)[::-1]  # 降序
        top_k_indices = sorted_indices[:k].tolist()
        
        # 选取子频带
        selected = [subbands[i] for i in top_k_indices]
        
        return selected, top_k_indices
    
    def fit(self, x_4ch: np.ndarray):
        """
        静态模式下，在训练集上拟合固定的频带索引
        
        Args:
            x_4ch: 4通道信号，shape (N, T, 4) 或 (T, 4)
        """
        if self.mode != 'static':
            return
            
        if x_4ch.ndim == 2:
            x_4ch = x_4ch[np.newaxis, ...]  # (1, T, 4)
            
        n_samples, T, n_channels = x_4ch.shape
        
        # 统计每个通道各频带的出现频率
        channel_freq_counts = []
        for ch in range(n_channels):
            freq_count = np.zeros(2 ** self.level)
            
            for i in range(n_samples):
                signal_1d = x_4ch[i, :, ch]
                subbands = self.wpd_decompose(signal_1d)
                _, indices = self.select_top_k(subbands, self.top_k)
                for idx in indices:
                    freq_count[idx] += 1
                    
            channel_freq_counts.append(freq_count)
        
        # 选取出现频率最高的Top-K频带作为固定配置
        self.static_indices = []
        for freq_count in channel_freq_counts:
            top_indices = np.argsort(freq_count)[::-1][:self.top_k].tolist()
            self.static_indices.append(top_indices)
    
    def expand_single(self, x_4ch: np.ndarray) -> np.ndarray:
        """
        对单个样本进行4通道→16通道扩展
        
        Args:
            x_4ch: 4通道信号，shape (T, 4)
            
        Returns:
            16通道超向量，shape (T, 16)
        """
        T, n_channels = x_4ch.shape
        assert n_channels == 4, f"Expected 4 channels, got {n_channels}"
        
        x_16ch = []
        
        for ch in range(n_channels):
            signal_1d = x_4ch[:, ch]
            
            # 添加原始通道
            x_16ch.append(signal_1d)
            
            # 小波包分解
            subbands = self.wpd_decompose(signal_1d)
            
            # 对齐长度
            subbands = self._align_length(subbands, T)
            
            # 选取Top-K子频带
            if self.mode == 'static' and self.static_indices is not None:
                # 使用固定索引
                indices = self.static_indices[ch]
                selected = [subbands[i] for i in indices]
            else:
                # 动态选择
                selected, _ = self.select_top_k(subbands, self.top_k)
            
            x_16ch.extend(selected)
        
        # Stack成 (T, 16)
        x_16ch = np.stack(x_16ch, axis=1)
        return x_16ch
    
    def expand(self, x_4ch: np.ndarray) -> np.ndarray:
        """
        批量进行4通道→16通道扩展
        
        Args:
            x_4ch: 4通道信号，shape (N, T, 4) 或 (T, 4)
            
        Returns:
            16通道超向量，shape (N, T, 16) 或 (T, 16)
        """
        squeeze = False
        if x_4ch.ndim == 2:
            x_4ch = x_4ch[np.newaxis, ...]
            squeeze = True
            
        n_samples = x_4ch.shape[0]
        results = []
        
        for i in range(n_samples):
            x_16 = self.expand_single(x_4ch[i])
            results.append(x_16)
            
        results = np.stack(results, axis=0)
        
        if squeeze:
            results = results[0]
            
        return results


class CSDMatrixBuilder:
    """
    16×16 CSD（互谱密度）矩阵构建器
    
    构建埃尔米特正定矩阵（HPD），用于后续的SPD流形分析。
    
    Args:
        fs: 采样频率
        nperseg: STFT窗口长度
        noverlap: 窗口重叠长度
        regularization_eps: 正则化系数，确保严格正定
        average_method: 平均方法 ('welch' 或 'mean')
    """
    
    def __init__(
        self,
        fs: int = 1000,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        regularization_eps: float = 1e-6,
        average_method: str = 'welch'
    ):
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.regularization_eps = regularization_eps
        self.average_method = average_method
        
    def _compute_stft(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算多通道信号的STFT
        
        Args:
            x: 多通道信号，shape (T, C)
            
        Returns:
            (frequencies, stft_result) where stft_result shape is (n_freq, n_segments, C)
        """
        T, C = x.shape
        
        # 对每个通道计算STFT
        stft_results = []
        for ch in range(C):
            f, t, Zxx = signal.stft(
                x[:, ch],
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                return_onesided=True
            )
            stft_results.append(Zxx)
        
        # Stack: (C, n_freq, n_segments) -> (n_freq, n_segments, C)
        stft_results = np.stack(stft_results, axis=0)
        stft_results = np.transpose(stft_results, (1, 2, 0))
        
        return f, stft_results
    
    def compute_csd_matrix(
        self, 
        x_16ch: np.ndarray,
        return_freqs: bool = False,
        normalize_input: bool = True
    ) -> np.ndarray:
        """
        计算16×16 HPD矩阵
        
        对于频域向量 X(f) ∈ C^16，CSD矩阵定义为：
        M(f) = E[X(f) X(f)^H]
        
        Args:
            x_16ch: 16通道信号，shape (T, 16)
            return_freqs: 是否返回频率轴
            normalize_input: 是否对输入信号进行Z-score归一化
            
        Returns:
            CSD矩阵，shape (n_freq, 16, 16)，复数类型
            如果return_freqs=True，返回 (freqs, csd_matrix)
        """
        T, C = x_16ch.shape
        assert C == 16, f"Expected 16 channels, got {C}"
        
        # 输入归一化：非常重要，防止CSD数值过大导致梯度爆炸
        if normalize_input:
            x_16ch = (x_16ch - np.mean(x_16ch, axis=0)) / (np.std(x_16ch, axis=0) + 1e-8)
        
        # 计算STFT
        freqs, stft_result = self._compute_stft(x_16ch)  # (n_freq, n_segments, 16)
        n_freq, n_segments, _ = stft_result.shape
        
        # 计算CSD矩阵: M(f) = E[X(f) X(f)^H]
        # 对每个频率点，计算所有时间段的平均外积
        csd_matrix = np.zeros((n_freq, C, C), dtype=np.complex128)
        
        for f_idx in range(n_freq):
            X = stft_result[f_idx, :, :]  # (n_segments, 16)
            
            # 计算 X * X^H 的平均
            # outer_products: (n_segments, 16, 16)
            outer_products = np.einsum('ni,nj->nij', X, np.conj(X))
            csd_matrix[f_idx] = np.mean(outer_products, axis=0)
        
        # 正则化确保严格正定
        csd_matrix = self.regularize(csd_matrix)
        
        if return_freqs:
            return freqs, csd_matrix
        return csd_matrix
    
    def regularize(self, M: np.ndarray) -> np.ndarray:
        """
        正则化确保矩阵严格正定
        
        M_reg = M + ε * I
        
        Args:
            M: CSD矩阵，shape (..., n, n)
            
        Returns:
            正则化后的矩阵
        """
        n = M.shape[-1]
        eye = np.eye(n, dtype=M.dtype)
        M_reg = M + self.regularization_eps * eye
        return M_reg
    
    def compute_csd_batch(
        self, 
        x_batch: np.ndarray,
        return_freqs: bool = False,
        normalize_input: bool = True
    ) -> np.ndarray:
        """
        批量计算CSD矩阵
        
        Args:
            x_batch: 批量16通道信号，shape (N, T, 16)
            return_freqs: 是否返回频率轴
            normalize_input: 是否归一化
            
        Returns:
            CSD矩阵批量，shape (N, n_freq, 16, 16)
        """
        N = x_batch.shape[0]
        results = []
        freqs = None
        
        for i in range(N):
            if return_freqs and freqs is None:
                freqs, csd = self.compute_csd_matrix(x_batch[i], return_freqs=True, normalize_input=normalize_input)
            else:
                csd = self.compute_csd_matrix(x_batch[i], return_freqs=False, normalize_input=normalize_input)
            results.append(csd)
        
        results = np.stack(results, axis=0)
        
        if return_freqs:
            return freqs, results
        return results
    
    def to_real_spd(self, csd_matrix: np.ndarray) -> np.ndarray:
        """
        将复数HPD矩阵转换为实数SPD矩阵
        
        使用块矩阵表示：
        [Re(M)  -Im(M)]
        [Im(M)   Re(M)]
        
        这保持了正定性。
        
        Args:
            csd_matrix: 复数CSD矩阵，shape (..., n, n)
            
        Returns:
            实数SPD矩阵，shape (..., 2n, 2n)
        """
        real_part = csd_matrix.real
        imag_part = csd_matrix.imag
        
        # 构建块矩阵
        top = np.concatenate([real_part, -imag_part], axis=-1)
        bottom = np.concatenate([imag_part, real_part], axis=-1)
        spd_real = np.concatenate([top, bottom], axis=-2)
        
        return spd_real
    
    def extract_single_freq(
        self, 
        csd_matrix: np.ndarray, 
        freq_idx: int = None,
        aggregate: str = 'mean'
    ) -> np.ndarray:
        """
        提取单个频率点或聚合所有频率点的CSD矩阵
        
        Args:
            csd_matrix: CSD矩阵，shape (n_freq, 16, 16)
            freq_idx: 频率索引，如果为None则聚合所有频率
            aggregate: 聚合方法 ('mean', 'sum', 'max')
            
        Returns:
            单个CSD矩阵，shape (16, 16)
        """
        if freq_idx is not None:
            return csd_matrix[freq_idx]
        
        if aggregate == 'mean':
            return np.mean(csd_matrix, axis=0)
        elif aggregate == 'sum':
            return np.sum(csd_matrix, axis=0)
        elif aggregate == 'max':
            # 取模最大
            norms = np.linalg.norm(csd_matrix, axis=(-2, -1))
            max_idx = np.argmax(norms)
            return csd_matrix[max_idx]
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")


def preprocess_pipeline(
    x_4ch: np.ndarray,
    fs: int = 1000,
    wavelet: str = 'db4',
    level: int = 3,
    top_k: int = 3,
    nperseg: int = 256,
    regularization_eps: float = 1e-6,
    aggregate_freq: str = 'mean'
) -> np.ndarray:
    """
    完整的预处理流水线：4通道信号 -> 16×16 CSD矩阵
    
    Args:
        x_4ch: 4通道信号，shape (T, 4) 或 (N, T, 4)
        fs: 采样频率
        wavelet: 小波基函数
        level: 小波分解层数
        top_k: 每通道保留的子频带数
        nperseg: STFT窗口长度
        regularization_eps: 正则化系数
        aggregate_freq: 频率聚合方法
        
    Returns:
        CSD矩阵，shape (16, 16) 或 (N, 16, 16)
    """
    # 虚拟通道扩展
    expander = VirtualChannelExpander(
        wavelet=wavelet,
        level=level,
        top_k=top_k,
        mode='dynamic'
    )
    x_16ch = expander.expand(x_4ch)
    
    # CSD矩阵构建
    builder = CSDMatrixBuilder(
        fs=fs,
        nperseg=nperseg,
        regularization_eps=regularization_eps
    )
    
    squeeze = False
    if x_16ch.ndim == 2:
        x_16ch = x_16ch[np.newaxis, ...]
        squeeze = True
    
    # 批量计算并聚合
    results = []
    for i in range(x_16ch.shape[0]):
        # 在这里执行归一化
        csd = builder.compute_csd_matrix(x_16ch[i], normalize_input=True)
        csd_single = builder.extract_single_freq(csd, aggregate=aggregate_freq)
        results.append(csd_single)
    
    results = np.stack(results, axis=0)
    
    if squeeze:
        results = results[0]
    
    return results


def csd_to_vector(csd_matrix: np.ndarray) -> np.ndarray:
    """
    将 16x16 复数 CSD 矩阵转换为实数向量
    
    取上三角（包含对角线）的 136 个复数值，
    分离实部和虚部得到 272 维实数向量。
    
    Args:
        csd_matrix: 复数 CSD 矩阵
            - 单个矩阵: shape (16, 16)
            - 批量矩阵: shape (B, 16, 16)
            
    Returns:
        实数特征向量
            - 单个: shape (272,)
            - 批量: shape (B, 272)
    """
    is_single = csd_matrix.ndim == 2
    if is_single:
        csd_matrix = csd_matrix[np.newaxis, ...]  # (1, 16, 16)
    
    B, n, _ = csd_matrix.shape
    assert n == 16, f"Expected 16x16 CSD matrix, got {n}x{n}"
    
    # 取上三角索引（包含对角线）
    # 对于 16x16 矩阵，上三角元素数量 = 16 * (16 + 1) / 2 = 136
    triu_idx = np.triu_indices(n)
    
    # 提取上三角元素: (B, 136) complex
    upper_tri = csd_matrix[:, triu_idx[0], triu_idx[1]]
    
    # 分离实部和虚部: (B, 272) real
    features = np.concatenate([upper_tri.real, upper_tri.imag], axis=-1)
    
    if is_single:
        features = features[0]  # (272,)
    
    return features.astype(np.float32)


def vector_to_csd(vector: np.ndarray) -> np.ndarray:
    """
    将 272 维实数向量还原为 16x16 复数 CSD 矩阵（逆操作）
    
    Args:
        vector: 实数特征向量
            - 单个: shape (272,)
            - 批量: shape (B, 272)
            
    Returns:
        复数 CSD 矩阵
            - 单个: shape (16, 16)
            - 批量: shape (B, 16, 16)
    """
    is_single = vector.ndim == 1
    if is_single:
        vector = vector[np.newaxis, ...]  # (1, 272)
    
    B = vector.shape[0]
    n = 16
    n_upper = n * (n + 1) // 2  # 136
    
    assert vector.shape[1] == 2 * n_upper, f"Expected 272-dim vector, got {vector.shape[1]}"
    
    # 分离实部和虚部
    real_part = vector[:, :n_upper]   # (B, 136)
    imag_part = vector[:, n_upper:]   # (B, 136)
    
    # 重建复数上三角
    upper_tri = real_part + 1j * imag_part  # (B, 136) complex
    
    # 重建完整矩阵
    triu_idx = np.triu_indices(n)
    csd_matrix = np.zeros((B, n, n), dtype=np.complex128)
    
    for b in range(B):
        csd_matrix[b, triu_idx[0], triu_idx[1]] = upper_tri[b]
        # 共轭转置填充下三角（CSD 矩阵是 Hermitian 的）
        csd_matrix[b] = csd_matrix[b] + np.conj(csd_matrix[b].T) - np.diag(np.diag(csd_matrix[b]))
    
    if is_single:
        csd_matrix = csd_matrix[0]
    
    return csd_matrix

