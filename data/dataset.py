"""
PyTorch Dataset 类
用于多模态故障检测的数据加载
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
from .preprocessing import VirtualChannelExpander, CSDMatrixBuilder
from .shaogang_loader import ShaoGangDataLoader


class MultiModalDataset(Dataset):
    """
    多模态故障检测数据集
    
    同时提供：
    - 振动分支：16×16 CSD矩阵（或扩展后的16通道信号）
    - 压力分支：原始压力信号
    
    Args:
        vibration_data: 振动+压力混合信号，shape (N, T, 4)
                       通道顺序: [P1, V1, P2, V2]
        pressure_indices: 压力通道索引，默认 [0, 2] (P1, P2)
        window_size: 滑动窗口大小
        stride: 滑动步长
        fs: 采样频率
        precompute_csd: 是否预计算CSD矩阵
        expander: 预配置的VirtualChannelExpander
        csd_builder: 预配置的CSDMatrixBuilder
    """
    
    def __init__(
        self,
        vibration_data: np.ndarray,
        pressure_indices: List[int] = [0, 2],
        window_size: int = 1024,
        stride: int = 512,
        fs: int = 1000,
        precompute_csd: bool = False,
        expander: Optional[VirtualChannelExpander] = None,
        csd_builder: Optional[CSDMatrixBuilder] = None,
        aggregate_freq: str = 'mean'
    ):
        self.pressure_indices = pressure_indices
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.precompute_csd = precompute_csd
        self.aggregate_freq = aggregate_freq
        
        # 初始化预处理器
        self.expander = expander or VirtualChannelExpander(mode='dynamic')
        self.csd_builder = csd_builder or CSDMatrixBuilder(fs=fs)
        
        # 处理输入数据形状
        if vibration_data.ndim == 2:
            # (T, 4) -> 单个长序列，需要切分
            self.windows = self._create_windows(vibration_data)
        else:
            # (N, T, 4) -> 已经是窗口形式
            self.windows = vibration_data
            
        self.n_samples = len(self.windows)
        
        # 预计算CSD矩阵（可选）
        self.precomputed_csd = None
        self.precomputed_x16 = None
        if precompute_csd:
            self._precompute()
    
    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        """
        从长序列创建滑动窗口
        
        Args:
            data: 输入数据，shape (T, 4)
            
        Returns:
            窗口数据，shape (N, window_size, 4)
        """
        T = data.shape[0]
        windows = []
        
        for start in range(0, T - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(data[start:end])
            
        return np.stack(windows, axis=0)
    
    def _precompute(self):
        """预计算所有样本的CSD矩阵和16通道扩展"""
        print("Precomputing CSD matrices...")
        
        csd_list = []
        x16_list = []
        
        for i in range(self.n_samples):
            x_4ch = self.windows[i]
            
            # 虚拟通道扩展
            x_16ch = self.expander.expand_single(x_4ch)
            x16_list.append(x_16ch)
            
            # CSD矩阵
            csd = self.csd_builder.compute_csd_matrix(x_16ch)
            csd_single = self.csd_builder.extract_single_freq(
                csd, aggregate=self.aggregate_freq
            )
            csd_list.append(csd_single)
        
        self.precomputed_x16 = np.stack(x16_list, axis=0)
        self.precomputed_csd = np.stack(csd_list, axis=0)
        
        print(f"Precomputed {self.n_samples} samples")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            dict with:
                - 'x_4ch': 原始4通道信号, shape (T, 4)
                - 'x_16ch': 扩展后16通道信号, shape (T, 16)
                - 'pressure': 压力信号, shape (T, 2)
                - 'csd_matrix': CSD矩阵, shape (16, 16) 复数
                - 'csd_real': 实数化CSD矩阵, shape (32, 32)
        """
        x_4ch = self.windows[idx]
        
        # 提取压力信号
        pressure = x_4ch[:, self.pressure_indices]
        
        if self.precomputed_csd is not None:
            x_16ch = self.precomputed_x16[idx]
            csd_matrix = self.precomputed_csd[idx]
        else:
            # 实时计算
            x_16ch = self.expander.expand_single(x_4ch)
            csd = self.csd_builder.compute_csd_matrix(x_16ch)
            csd_matrix = self.csd_builder.extract_single_freq(
                csd, aggregate=self.aggregate_freq
            )
        
        # 转换为实数SPD矩阵
        csd_real = self.csd_builder.to_real_spd(csd_matrix)
        
        # 转换为PyTorch tensor
        return {
            'x_4ch': torch.from_numpy(x_4ch).float(),
            'x_16ch': torch.from_numpy(x_16ch).float(),
            'pressure': torch.from_numpy(pressure).float(),
            'csd_matrix': torch.from_numpy(csd_matrix).cfloat(),
            'csd_real': torch.from_numpy(csd_real).float(),
        }


class SequentialMultiModalDataset(Dataset):
    """
    序列化多模态数据集
    
    用于需要时序上下文的场景（如对比学习的正负样本构建）
    
    Args:
        base_dataset: 基础MultiModalDataset
        context_length: 上下文长度（前后各取多少个窗口）
        augmentation: 可选的数据增强（用于生成负样本）
        generate_negative: 是否生成负样本
    """
    
    def __init__(
        self,
        base_dataset: MultiModalDataset,
        context_length: int = 1,
        augmentation = None,
        generate_negative: bool = True
    ):
        self.base_dataset = base_dataset
        self.context_length = context_length
        self.augmentation = augmentation
        self.generate_negative = generate_negative
        
        # 有效索引（去掉边界）
        self.valid_start = context_length
        self.valid_end = len(base_dataset) - context_length
        self.n_samples = max(0, self.valid_end - self.valid_start)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取带上下文的样本
        
        Returns:
            dict with:
                - 'current': 当前样本
                - 'prev': 前一个样本（正样本对）
                - 'next': 后一个样本（正样本对）
                - 'negative_csd': 负样本 CSD（如果启用增强）
                - 'negative_csd_real': 负样本实数 CSD
                - 'negative_csd_vector': 负样本 CSD 向量
        """
        actual_idx = self.valid_start + idx
        
        current = self.base_dataset[actual_idx]
        prev_sample = self.base_dataset[actual_idx - 1]
        next_sample = self.base_dataset[actual_idx + 1]
        
        result = {
            'current': current,
            'prev': prev_sample,
            'next': next_sample,
        }
        
        # 生成负样本（正交性扰动）
        if self.generate_negative and self.augmentation is not None:
            # 对 CSD 矩阵进行扰动
            if 'csd_matrix' in current:
                negative_csd = self.augmentation(current['csd_matrix'])
                result['negative_csd'] = negative_csd
                
                # 转换为实数 SPD 矩阵
                if negative_csd.is_complex():
                    real_part = negative_csd.real
                    imag_part = negative_csd.imag
                    top = torch.cat([real_part, -imag_part], dim=-1)
                    bottom = torch.cat([imag_part, real_part], dim=-1)
                    result['negative_csd_real'] = torch.cat([top, bottom], dim=-2).float()
                else:
                    result['negative_csd_real'] = negative_csd.float()
                
                # 同时生成负样本的 CSD 向量（用于 MLP 方案）
                if negative_csd.is_complex():
                    neg_csd_np = negative_csd.numpy()
                else:
                    # 从实数矩阵恢复复数矩阵（如果需要）
                    neg_csd_np = negative_csd.numpy()
                
                # 转换为向量（取上三角，分离实部虚部）
                triu_idx = np.triu_indices(negative_csd.shape[-1])
                upper_tri = neg_csd_np[triu_idx[0], triu_idx[1]]
                if np.iscomplexobj(upper_tri):
                    neg_vector = np.concatenate([upper_tri.real, upper_tri.imag])
                else:
                    neg_vector = upper_tri.flatten()
                result['negative_csd_vector'] = torch.from_numpy(neg_vector).float()
            
            # 如果有 csd_vector，直接在向量空间添加噪声作为负样本
            elif 'csd_vector' in current:
                csd_vec = current['csd_vector']
                # 添加噪声生成负样本（简化版扰动）
                noise_scale = 0.1 * csd_vec.std()
                noise = torch.randn_like(csd_vec) * noise_scale
                result['negative_csd_vector'] = csd_vec + noise
        
        return result


def create_dataloaders(
    data: np.ndarray,
    train_ratio: float = 0.8,
    window_size: int = 1024,
    stride: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    precompute_csd: bool = True,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        data: 输入数据，shape (T, 4) 或 (N, T, 4)
        train_ratio: 训练集比例
        window_size: 窗口大小
        stride: 滑动步长
        batch_size: 批大小
        num_workers: 数据加载线程数
        precompute_csd: 是否预计算CSD
        **kwargs: 传递给MultiModalDataset的其他参数
        
    Returns:
        (train_loader, val_loader)
    """
    # 创建数据集
    dataset = MultiModalDataset(
        vibration_data=data,
        window_size=window_size,
        stride=stride,
        precompute_csd=precompute_csd,
        **kwargs
    )
    
    # 划分训练/验证
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


class PrecomputedMultiModalDataset(Dataset):
    """
    预计算的多模态数据集
    
    加载由 scripts/preprocess_dataset.py 预处理的数据。
    训练时无需实时计算 WPD 和 CSD，大幅提升速度。
    
    Args:
        preprocessed_path: 预处理数据路径 (.pt 文件)
        pressure_indices: 压力通道索引（默认从元数据读取）
    """
    
    def __init__(
        self,
        preprocessed_path: str,
        pressure_indices: Optional[List[int]] = None
    ):
        print(f"Loading preprocessed data from {preprocessed_path}...")
        data = torch.load(preprocessed_path)
        
        self.x_4ch = data['x_4ch']
        self.x_16ch = data['x_16ch']
        self.csd_matrix = data['csd_matrix']
        self.csd_real = data['csd_real']
        self.csd_vector = data.get('csd_vector', None)  # 272 维向量化特征
        self.pressure = data['pressure']
        self.metadata = data.get('metadata', {})
        self.static_indices = data.get('static_indices', None)
        
        # 使用提供的索引或从元数据读取
        if pressure_indices is not None:
            self.pressure_indices = pressure_indices
        else:
            self.pressure_indices = self.metadata.get('pressure_indices', [0, 2])
        
        print(f"Loaded {len(self)} samples")
        print(f"Metadata: {self.metadata}")
    
    def __len__(self) -> int:
        return len(self.x_4ch)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            dict with:
                - 'x_4ch': 原始4通道信号
                - 'x_16ch': 扩展后16通道信号
                - 'pressure': 压力信号
                - 'csd_matrix': 复数 CSD 矩阵
                - 'csd_real': 实数化 CSD 矩阵
                - 'csd_vector': 272维向量化特征（如果存在）
        """
        result = {
            'x_4ch': self.x_4ch[idx],
            'x_16ch': self.x_16ch[idx],
            'pressure': self.pressure[idx],
            'csd_matrix': self.csd_matrix[idx],
            'csd_real': self.csd_real[idx],
        }
        
        # 添加向量化特征（如果存在）
        if self.csd_vector is not None:
            result['csd_vector'] = self.csd_vector[idx]
        
        return result


def create_dataloaders_from_preprocessed(
    preprocessed_path: str,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 4,
    use_sequential: bool = False,
    augmentation = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    从预处理数据创建训练和验证数据加载器
    
    Args:
        preprocessed_path: 预处理数据路径
        train_ratio: 训练集比例
        batch_size: 批大小
        num_workers: 数据加载线程数
        use_sequential: 是否使用序列化数据集（用于对比学习）
        augmentation: 数据增强（用于生成负样本）
        
    Returns:
        (train_loader, val_loader)
    """
    # 创建基础数据集
    base_dataset = PrecomputedMultiModalDataset(preprocessed_path)
    
    # 划分训练/验证
    n_total = len(base_dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        base_dataset, [n_train, n_val]
    )
    
    # 如果需要序列化数据集
    if use_sequential:
        # 注意：random_split 后的数据集不能直接用于 SequentialMultiModalDataset
        # 这里使用完整数据集，然后在内部划分
        train_sequential = SequentialMultiModalDataset(
            base_dataset,
            context_length=1,
            augmentation=augmentation,
            generate_negative=augmentation is not None
        )
        # 只取前 n_train 个样本的有效部分
        train_indices = list(range(min(n_train, len(train_sequential))))
        train_dataset = torch.utils.data.Subset(train_sequential, train_indices)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 防止 BatchNorm 在 batch_size=1 时报错
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # 验证时保留所有数据
    )
    
    return train_loader, val_loader


class ShaoGangDataset(Dataset):
    """
    韶钢数据集
    
    直接从 CSV 文件加载数据，适合调试和小规模实验。
    对于大规模训练，建议使用 preprocess_dataset.py 预处理后使用 PrecomputedMultiModalDataset。
    
    CSV 列结构：
        - acceleration01: 振动传感器1 (V1)
        - acceleration02: 振动传感器2 (V2)
        - stress01: 压力传感器1 (P1)
        - stress02: 压力传感器2 (P2)
        - timestamp: 时间戳
    
    转换后通道顺序: [P1, V1, P2, V2]（与研究报告一致）
    
    Args:
        csv_path: CSV 文件路径
        window_size: 滑动窗口大小
        stride: 滑动步长
        fs: 采样频率 (Hz)
        max_windows: 最大窗口数限制
        normalize: 是否归一化数据
        precompute_csd: 是否预计算 CSD 矩阵
        expander: 预配置的 VirtualChannelExpander
        csd_builder: 预配置的 CSDMatrixBuilder
    """
    
    # 压力通道索引（重排后的 [P1, V1, P2, V2] 顺序）
    PRESSURE_INDICES = [0, 2]
    
    def __init__(
        self,
        csv_path: str,
        window_size: int = 1024,
        stride: int = 512,
        fs: int = 100,
        max_windows: Optional[int] = None,
        normalize: bool = False,
        precompute_csd: bool = False,
        expander: Optional[VirtualChannelExpander] = None,
        csd_builder: Optional[CSDMatrixBuilder] = None,
        aggregate_freq: str = 'mean',
        show_progress: bool = True
    ):
        self.csv_path = Path(csv_path)
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.precompute_csd = precompute_csd
        self.aggregate_freq = aggregate_freq
        
        # 初始化预处理器
        self.expander = expander or VirtualChannelExpander(mode='dynamic')
        self.csd_builder = csd_builder or CSDMatrixBuilder(fs=fs)
        
        # 使用 ShaoGangDataLoader 加载数据
        print(f"Loading ShaoGang dataset from {csv_path}...")
        loader = ShaoGangDataLoader(
            csv_path=csv_path,
            fs=fs,
            normalize=normalize
        )
        
        if show_progress:
            print(loader.info())
        
        # 创建滑动窗口
        self.windows = loader.create_windows(
            window_size=window_size,
            stride=stride,
            max_windows=max_windows,
            show_progress=show_progress
        )
        
        self.n_samples = len(self.windows)
        print(f"Created {self.n_samples} windows")
        
        # 预计算 CSD 矩阵（可选）
        self.precomputed_csd = None
        self.precomputed_x16 = None
        self.precomputed_csd_real = None
        if precompute_csd:
            self._precompute()
    
    def _precompute(self):
        """预计算所有样本的 CSD 矩阵和 16 通道扩展"""
        from tqdm import tqdm
        print("Precomputing CSD matrices for ShaoGang dataset...")
        
        csd_list = []
        csd_real_list = []
        x16_list = []
        
        for i in tqdm(range(self.n_samples), desc="Preprocessing"):
            x_4ch = self.windows[i]
            
            # 虚拟通道扩展 (4 -> 16)
            x_16ch = self.expander.expand_single(x_4ch)
            x16_list.append(x_16ch)
            
            # CSD 矩阵
            csd = self.csd_builder.compute_csd_matrix(x_16ch)
            csd_single = self.csd_builder.extract_single_freq(
                csd, aggregate=self.aggregate_freq
            )
            csd_list.append(csd_single)
            
            # 实数 SPD 矩阵
            csd_real = self.csd_builder.to_real_spd(csd_single)
            csd_real_list.append(csd_real)
        
        self.precomputed_x16 = np.stack(x16_list, axis=0)
        self.precomputed_csd = np.stack(csd_list, axis=0)
        self.precomputed_csd_real = np.stack(csd_real_list, axis=0)
        
        print(f"Precomputed {self.n_samples} samples")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            dict with:
                - 'x_4ch': 原始 4 通道信号, shape (T, 4)，顺序 [P1, V1, P2, V2]
                - 'x_16ch': 扩展后 16 通道信号, shape (T, 16)
                - 'pressure': 压力信号, shape (T, 2)
                - 'csd_matrix': CSD 矩阵, shape (16, 16) 复数
                - 'csd_real': 实数化 CSD 矩阵, shape (32, 32)
        """
        x_4ch = self.windows[idx]
        
        # 提取压力信号 (P1, P2)
        pressure = x_4ch[:, self.PRESSURE_INDICES]
        
        if self.precomputed_csd is not None:
            x_16ch = self.precomputed_x16[idx]
            csd_matrix = self.precomputed_csd[idx]
            csd_real = self.precomputed_csd_real[idx]
        else:
            # 实时计算
            x_16ch = self.expander.expand_single(x_4ch)
            csd = self.csd_builder.compute_csd_matrix(x_16ch)
            csd_matrix = self.csd_builder.extract_single_freq(
                csd, aggregate=self.aggregate_freq
            )
            csd_real = self.csd_builder.to_real_spd(csd_matrix)
        
        # 转换为 PyTorch tensor
        return {
            'x_4ch': torch.from_numpy(x_4ch).float(),
            'x_16ch': torch.from_numpy(x_16ch).float(),
            'pressure': torch.from_numpy(pressure).float(),
            'csd_matrix': torch.from_numpy(csd_matrix).cfloat(),
            'csd_real': torch.from_numpy(csd_real).float(),
        }
    
    def get_metadata(self) -> Dict:
        """获取数据集元数据"""
        return {
            'source_file': str(self.csv_path),
            'n_samples': self.n_samples,
            'window_size': self.window_size,
            'stride': self.stride,
            'fs': self.fs,
            'channel_order': ['P1', 'V1', 'P2', 'V2'],
            'pressure_indices': self.PRESSURE_INDICES,
        }


def create_shaogang_dataloaders(
    csv_path: str,
    train_ratio: float = 0.8,
    window_size: int = 1024,
    stride: int = 512,
    fs: int = 100,
    max_windows: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    precompute_csd: bool = True,
    normalize: bool = False,
    use_sequential: bool = False,
    augmentation = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    从韶钢 CSV 文件创建训练和验证数据加载器
    
    Args:
        csv_path: CSV 文件路径
        train_ratio: 训练集比例
        window_size: 窗口大小
        stride: 滑动步长
        fs: 采样频率
        max_windows: 最大窗口数
        batch_size: 批大小
        num_workers: 数据加载线程数
        precompute_csd: 是否预计算 CSD
        normalize: 是否归一化数据
        use_sequential: 是否使用序列化数据集
        augmentation: 数据增强
        
    Returns:
        (train_loader, val_loader)
    """
    # 创建数据集
    dataset = ShaoGangDataset(
        csv_path=csv_path,
        window_size=window_size,
        stride=stride,
        fs=fs,
        max_windows=max_windows,
        normalize=normalize,
        precompute_csd=precompute_csd,
    )
    
    # 划分训练/验证
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    # 如果需要序列化数据集
    if use_sequential:
        train_sequential = SequentialMultiModalDataset(
            dataset,
            context_length=1,
            augmentation=augmentation,
            generate_negative=augmentation is not None
        )
        train_indices = list(range(min(n_train, len(train_sequential))))
        train_dataset = torch.utils.data.Subset(train_sequential, train_indices)
    
    # 创建数据加载器
    shaogang_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    shaogang_val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return shaogang_train_loader, shaogang_val_loader
