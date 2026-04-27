"""
韶钢数据集加载器

专门用于加载韶钢 CSV 格式的多传感器数据，支持：
- 分块读取大型 CSV 文件（避免内存溢出）
- 通道顺序重排（从 [V1, V2, P1, P2] 到 [P1, V1, P2, V2]）
- 时间戳解析
- 可选数据归一化

CSV 列结构：
- acceleration01: 振动传感器1 (V1)
- acceleration02: 振动传感器2 (V2)
- stress01: 压力传感器1 (P1)
- stress02: 压力传感器2 (P2)
- timestamp: 时间戳
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Generator, List, Dict, Union
from tqdm import tqdm


class ShaoGangDataLoader:
    """
    韶钢 CSV 数据加载器
    
    将原始 CSV 数据转换为模型所需的格式。
    
    原始列顺序: [acceleration01, acceleration02, stress01, stress02]
                即 [V1, V2, P1, P2]
    
    目标列顺序: [P1, V1, P2, V2] (研究报告期望的顺序)
    
    Args:
        csv_path: CSV 文件路径
        fs: 采样频率 (Hz)，默认 100
        chunk_size: 分块读取大小，默认 1000000 行
        normalize: 是否归一化数据
        normalize_method: 归一化方法 ('minmax', 'zscore', 'none')
    """
    
    # 列名映射（原始列名 -> 物理含义）
    COLUMN_MAPPING = {
        'acceleration01': 'V1',
        'acceleration02': 'V2',
        'stress01': 'P1',
        'stress02': 'P2',
        'timestamp': 'timestamp',
    }
    
    # 重排为研究报告期望的顺序 [P1, V1, P2, V2]
    CHANNEL_ORDER = ['stress01', 'acceleration01', 'stress02', 'acceleration02']
    
    # 压力通道索引（重排后）
    PRESSURE_INDICES = [0, 2]  # P1 和 P2 的位置
    
    def __init__(
        self,
        csv_path: str,
        fs: int = 100,
        chunk_size: int = 1000000,
        normalize: bool = False,
        normalize_method: str = 'zscore'
    ):
        self.csv_path = Path(csv_path)
        self.fs = fs
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.normalize_method = normalize_method
        
        # 验证文件存在
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        # 统计信息（懒加载）
        self._total_rows = None
        self._columns = None
        self._stats = None
    
    @property
    def total_rows(self) -> int:
        """获取总行数（懒加载）"""
        if self._total_rows is None:
            self._total_rows = sum(1 for _ in open(self.csv_path)) - 1  # 减去表头
        return self._total_rows
    
    @property
    def columns(self) -> List[str]:
        """获取列名"""
        if self._columns is None:
            df = pd.read_csv(self.csv_path, nrows=1)
            self._columns = df.columns.tolist()
        return self._columns
    
    @property
    def duration_seconds(self) -> float:
        """数据总时长（秒）"""
        return self.total_rows / self.fs
    
    @property
    def duration_hours(self) -> float:
        """数据总时长（小时）"""
        return self.duration_seconds / 3600
    
    def get_stats(self, sample_size: int = 100000) -> Dict[str, Dict[str, float]]:
        """
        获取数据统计信息（使用采样估计）
        
        Args:
            sample_size: 采样大小
            
        Returns:
            各通道的统计信息
        """
        if self._stats is None:
            df = pd.read_csv(self.csv_path, nrows=sample_size)
            self._stats = {}
            for col in self.CHANNEL_ORDER:
                self._stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                }
        return self._stats
    
    def iter_chunks(self, columns: List[str] = None) -> Generator[pd.DataFrame, None, None]:
        """
        分块迭代读取 CSV
        
        Args:
            columns: 要读取的列名，None 表示全部
            
        Yields:
            DataFrame 块
        """
        if columns is None:
            columns = self.CHANNEL_ORDER + ['timestamp']
        
        for chunk in pd.read_csv(
            self.csv_path,
            usecols=columns,
            chunksize=self.chunk_size
        ):
            yield chunk
    
    def load_chunk(
        self,
        start_row: int,
        num_rows: int,
        include_timestamp: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        加载指定范围的数据
        
        Args:
            start_row: 起始行（0-indexed）
            num_rows: 读取行数
            include_timestamp: 是否返回时间戳
            
        Returns:
            data: 4通道数据，shape (num_rows, 4)，顺序 [P1, V1, P2, V2]
            timestamps: 时间戳（如果 include_timestamp=True）
        """
        columns = self.CHANNEL_ORDER.copy()
        if include_timestamp:
            columns.append('timestamp')
        
        df = pd.read_csv(
            self.csv_path,
            usecols=columns,
            skiprows=range(1, start_row + 1),  # 跳过表头和前 start_row 行
            nrows=num_rows
        )
        
        # 按目标顺序提取数据
        data = df[self.CHANNEL_ORDER].values.astype(np.float32)
        
        # 归一化
        if self.normalize:
            data = self._normalize(data)
        
        if include_timestamp:
            timestamps = pd.to_datetime(df['timestamp']).values
            return data, timestamps
        
        return data
    
    def load_all(
        self,
        max_rows: int = None,
        include_timestamp: bool = False,
        show_progress: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        加载全部数据（分块读取后合并）
        
        Args:
            max_rows: 最大行数限制，None 表示全部
            include_timestamp: 是否返回时间戳
            show_progress: 是否显示进度条
            
        Returns:
            data: 4通道数据，shape (T, 4)，顺序 [P1, V1, P2, V2]
            timestamps: 时间戳（如果 include_timestamp=True）
        """
        columns = self.CHANNEL_ORDER.copy()
        if include_timestamp:
            columns.append('timestamp')
        
        data_list = []
        timestamp_list = []
        total_loaded = 0
        
        iterator = self.iter_chunks(columns)
        if show_progress:
            total_chunks = (self.total_rows + self.chunk_size - 1) // self.chunk_size
            iterator = tqdm(iterator, total=total_chunks, desc="Loading CSV")
        
        for chunk in iterator:
            # 检查是否达到最大行数
            if max_rows is not None and total_loaded >= max_rows:
                break
            
            # 裁剪到最大行数
            if max_rows is not None:
                remaining = max_rows - total_loaded
                chunk = chunk.iloc[:remaining]
            
            # 提取数据
            data = chunk[self.CHANNEL_ORDER].values.astype(np.float32)
            data_list.append(data)
            
            if include_timestamp:
                timestamps = pd.to_datetime(chunk['timestamp']).values
                timestamp_list.append(timestamps)
            
            total_loaded += len(chunk)
        
        # 合并
        data = np.concatenate(data_list, axis=0)
        
        # 归一化
        if self.normalize:
            data = self._normalize(data)
        
        if include_timestamp:
            timestamps = np.concatenate(timestamp_list, axis=0)
            return data, timestamps
        
        return data
    
    def load_time_range(
        self,
        start_time: str,
        end_time: str,
        include_timestamp: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        加载指定时间范围的数据
        
        Args:
            start_time: 起始时间（格式：'2025-09-06 21:30:00'）
            end_time: 结束时间
            include_timestamp: 是否返回时间戳
            
        Returns:
            data: 4通道数据
            timestamps: 时间戳（如果 include_timestamp=True）
        """
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        
        columns = self.CHANNEL_ORDER + ['timestamp']
        data_list = []
        timestamp_list = []
        
        for chunk in tqdm(self.iter_chunks(columns), desc="Loading time range"):
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
            mask = (chunk['timestamp'] >= start_dt) & (chunk['timestamp'] <= end_dt)
            filtered = chunk[mask]
            
            if len(filtered) > 0:
                data = filtered[self.CHANNEL_ORDER].values.astype(np.float32)
                data_list.append(data)
                
                if include_timestamp:
                    timestamps = filtered['timestamp'].values
                    timestamp_list.append(timestamps)
        
        if len(data_list) == 0:
            raise ValueError(f"No data found in time range [{start_time}, {end_time}]")
        
        data = np.concatenate(data_list, axis=0)
        
        if self.normalize:
            data = self._normalize(data)
        
        if include_timestamp:
            timestamps = np.concatenate(timestamp_list, axis=0)
            return data, timestamps
        
        return data
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        归一化数据
        
        Args:
            data: 原始数据，shape (T, 4)
            
        Returns:
            归一化后的数据
        """
        if self.normalize_method == 'none':
            return data
        
        stats = self.get_stats()
        
        if self.normalize_method == 'zscore':
            # Z-score 归一化
            for i, col in enumerate(self.CHANNEL_ORDER):
                mean = stats[col]['mean']
                std = stats[col]['std']
                if std > 0:
                    data[:, i] = (data[:, i] - mean) / std
                    
        elif self.normalize_method == 'minmax':
            # Min-Max 归一化到 [0, 1]
            for i, col in enumerate(self.CHANNEL_ORDER):
                min_val = stats[col]['min']
                max_val = stats[col]['max']
                if max_val > min_val:
                    data[:, i] = (data[:, i] - min_val) / (max_val - min_val)
        
        return data
    
    def create_windows(
        self,
        window_size: int = 1024,
        stride: int = 512,
        max_windows: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        创建滑动窗口
        
        Args:
            window_size: 窗口大小
            stride: 滑动步长
            max_windows: 最大窗口数限制
            show_progress: 是否显示进度条
            
        Returns:
            windows: shape (N, window_size, 4)
        """
        # 估计需要加载的数据量
        if max_windows is not None:
            max_rows = max_windows * stride + window_size
        else:
            max_rows = None
        
        # 加载数据
        data = self.load_all(max_rows=max_rows, show_progress=show_progress)
        
        # 创建窗口
        T = len(data)
        n_windows = (T - window_size) // stride + 1
        
        if max_windows is not None:
            n_windows = min(n_windows, max_windows)
        
        windows = []
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            windows.append(data[start:end])
        
        return np.stack(windows, axis=0)
    
    def info(self) -> str:
        """打印数据集信息"""
        stats = self.get_stats()
        
        info_str = f"""
=== ShaoGang Dataset Info ===
File: {self.csv_path}
Total rows: {self.total_rows:,}
Sampling rate: {self.fs} Hz
Duration: {self.duration_seconds:.1f} seconds ({self.duration_hours:.2f} hours)

Channel Statistics (sampled):
"""
        for col in self.CHANNEL_ORDER:
            s = stats[col]
            physical = self.COLUMN_MAPPING.get(col, col)
            info_str += f"  {col} ({physical}): mean={s['mean']:.2f}, std={s['std']:.2f}, range=[{s['min']:.0f}, {s['max']:.0f}]\n"
        
        return info_str


def load_shaogang_csv(
    csv_path: str,
    window_size: int = 1024,
    stride: int = 512,
    fs: int = 100,
    max_windows: int = None,
    normalize: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    便捷函数：加载韶钢 CSV 并创建窗口
    
    Args:
        csv_path: CSV 文件路径
        window_size: 窗口大小
        stride: 滑动步长
        fs: 采样频率
        max_windows: 最大窗口数
        normalize: 是否归一化
        
    Returns:
        windows: shape (N, window_size, 4)，通道顺序 [P1, V1, P2, V2]
        metadata: 元数据字典
    """
    loader = ShaoGangDataLoader(
        csv_path=csv_path,
        fs=fs,
        normalize=normalize
    )
    
    print(loader.info())
    
    windows = loader.create_windows(
        window_size=window_size,
        stride=stride,
        max_windows=max_windows
    )
    
    metadata = {
        'source_file': str(loader.csv_path),
        'fs': fs,
        'window_size': window_size,
        'stride': stride,
        'n_windows': len(windows),
        'channel_order': ['P1', 'V1', 'P2', 'V2'],
        'pressure_indices': ShaoGangDataLoader.PRESSURE_INDICES,
        'total_rows': loader.total_rows,
        'duration_hours': loader.duration_hours,
    }
    
    return windows, metadata

