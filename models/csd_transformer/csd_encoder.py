"""
CSD Pair-Token Transformer 编码器

将 CSD 矩阵的上三角元素视为传感器对的关系 Token，
使用 Transformer 学习全局耦合关系。

物理结构编码：
- 16×16 CSD 矩阵对应 16 个虚拟通道
- 通道顺序: [P1(0-3), V1(4-7), P2(8-11), V2(12-15)]
  - 每个物理通道扩展为 4 个子通道（原始 + 3 个小波子频带）
- 上三角共 136 个传感器对 (16*17/2)
- 物理块类型: P-P内部 / V-V内部 / P-V耦合 / 跨臂耦合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def get_block_type(i: int, j: int) -> int:
    """
    根据传感器对索引判断物理块类型
    
    通道布局 (16 通道):
    - Arm1 P1: 0-3   (压力1 + 3个子频带)
    - Arm1 V1: 4-7   (振动1 + 3个子频带)
    - Arm2 P2: 8-11  (压力2 + 3个子频带)
    - Arm2 V2: 12-15 (振动2 + 3个子频带)
    
    块类型:
    - 0: 对角块（同通道组内部的跨频带耦合）
    - 1: P-V 耦合块（同臂压力-振动耦合）
    - 2: 跨臂同模态块（P1-P2 或 V1-V2）
    - 3: 跨臂跨模态块（P1-V2 或 V1-P2）
    
    Args:
        i: 行索引 (0-15)
        j: 列索引 (0-15)
        
    Returns:
        块类型 (0-3)
    """
    # 定义通道组
    arm1_p = set(range(0, 4))    # P1 组
    arm1_v = set(range(4, 8))    # V1 组
    arm2_p = set(range(8, 12))   # P2 组
    arm2_v = set(range(12, 16))  # V2 组
    
    # 获取 i, j 所属的组
    def get_group(idx):
        if idx in arm1_p:
            return 'arm1_p'
        elif idx in arm1_v:
            return 'arm1_v'
        elif idx in arm2_p:
            return 'arm2_p'
        else:
            return 'arm2_v'
    
    gi = get_group(i)
    gj = get_group(j)
    
    # 同组（对角块）
    if gi == gj:
        return 0
    
    # 同臂 P-V 耦合
    if (gi == 'arm1_p' and gj == 'arm1_v') or (gi == 'arm1_v' and gj == 'arm1_p'):
        return 1
    if (gi == 'arm2_p' and gj == 'arm2_v') or (gi == 'arm2_v' and gj == 'arm2_p'):
        return 1
    
    # 跨臂同模态
    if (gi == 'arm1_p' and gj == 'arm2_p') or (gi == 'arm2_p' and gj == 'arm1_p'):
        return 2
    if (gi == 'arm1_v' and gj == 'arm2_v') or (gi == 'arm2_v' and gj == 'arm1_v'):
        return 2
    
    # 跨臂跨模态
    return 3


def create_pair_indices(matrix_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建上三角索引和对应的物理块类型
    
    Args:
        matrix_size: CSD 矩阵大小（默认 16）
        
    Returns:
        pair_indices: (2, n_pairs) 行列索引
        block_types: (n_pairs,) 物理块类型
    """
    # 上三角索引（包含对角线）
    rows, cols = torch.triu_indices(matrix_size, matrix_size, offset=0)
    n_pairs = rows.size(0)
    
    # 计算每对的物理块类型
    block_types = torch.zeros(n_pairs, dtype=torch.long)
    for k in range(n_pairs):
        i, j = rows[k].item(), cols[k].item()
        block_types[k] = get_block_type(i, j)
    
    return torch.stack([rows, cols], dim=0), block_types


class PairPositionEncoding(nn.Module):
    """
    物理位置编码
    
    为每个传感器对 (i, j) 提供位置信息：
    1. 行索引嵌入 (i)
    2. 列索引嵌入 (j)  
    3. 物理块类型嵌入（对角/P-V/跨臂同模态/跨臂跨模态）
    
    Args:
        d_model: 模型维度
        matrix_size: CSD 矩阵大小（默认 16）
        n_block_types: 物理块类型数量（默认 4）
    """
    
    def __init__(
        self,
        d_model: int,
        matrix_size: int = 16,
        n_block_types: int = 4
    ):
        super().__init__()
        
        self.d_model = d_model
        self.matrix_size = matrix_size
        
        # 行列索引嵌入
        self.row_embedding = nn.Embedding(matrix_size, d_model // 3)
        self.col_embedding = nn.Embedding(matrix_size, d_model // 3)
        
        # 物理块类型嵌入
        self.block_embedding = nn.Embedding(n_block_types, d_model - 2 * (d_model // 3))
        
        # 预计算索引和块类型
        pair_indices, block_types = create_pair_indices(matrix_size)
        self.register_buffer('row_indices', pair_indices[0])
        self.register_buffer('col_indices', pair_indices[1])
        self.register_buffer('block_types', block_types)
        
        self.n_pairs = pair_indices.size(1)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        生成位置编码
        
        Args:
            batch_size: 批大小
            
        Returns:
            位置编码，shape (B, n_pairs, d_model)
        """
        # 获取各部分嵌入
        row_emb = self.row_embedding(self.row_indices)   # (n_pairs, d1)
        col_emb = self.col_embedding(self.col_indices)   # (n_pairs, d2)
        block_emb = self.block_embedding(self.block_types)  # (n_pairs, d3)
        
        # 拼接
        pos_emb = torch.cat([row_emb, col_emb, block_emb], dim=-1)  # (n_pairs, d_model)
        
        # 扩展到 batch
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, n_pairs, d_model)
        
        return pos_emb


class CSDTransformerEncoder(nn.Module):
    """
    CSD Pair-Token Transformer 编码器
    
    将 CSD 矩阵的上三角元素视为 136 个 Token，
    使用 Transformer Self-Attention 学习传感器间的全局耦合关系。
    
    核心思想：
    - 每个 Token 代表一个传感器对 (i, j) 的互谱密度
    - Token 特征 = [Real, Imag, Magnitude, Phase] 或 [Real, Imag]
    - 位置编码 = 行索引 + 列索引 + 物理块类型
    - Self-Attention 自动学习哪些耦合关系最关键
    
    Args:
        matrix_size: CSD 矩阵大小（默认 16，对应 16 虚拟通道）
        token_dim: Token 原始特征维度（默认 4: Re/Im/Mag/Phase）
        d_model: Transformer 模型维度
        n_heads: 注意力头数
        n_layers: Encoder 层数
        d_ff: 前馈网络维度（默认 4*d_model）
        dropout: Dropout 比例
        use_cls_token: 是否使用 CLS token 聚合
        projection_dim: 输出投影维度（None 表示不投影）
    """
    
    def __init__(
        self,
        matrix_size: int = 16,
        token_dim: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        projection_dim: Optional[int] = 128
    ):
        super().__init__()
        
        self.matrix_size = matrix_size
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.token_dim = token_dim
        
        # 计算上三角对数
        self.n_pairs = (matrix_size * (matrix_size + 1)) // 2
        
        # 预计算上三角索引
        rows, cols = torch.triu_indices(matrix_size, matrix_size, offset=0)
        self.register_buffer('triu_rows', rows)
        self.register_buffer('triu_cols', cols)
        
        # Token 嵌入层（将 token_dim 投影到 d_model）
        self.token_embedding = nn.Linear(token_dim, d_model)
        
        # 位置编码
        self.position_encoding = PairPositionEncoding(
            d_model=d_model,
            matrix_size=matrix_size,
            n_block_types=4
        )
        
        # CLS Token（可选）
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer Encoder
        d_ff = d_ff or 4 * d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN 更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Layer Norm
        self.norm = nn.LayerNorm(d_model)
        
        # 输出投影（可选）
        self.projection_dim = projection_dim
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(d_model, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim)
            )
        else:
            self.projection = None
        
        # 输出维度
        self.output_dim = projection_dim if projection_dim else d_model
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def _extract_tokens(self, csd_matrix: torch.Tensor) -> torch.Tensor:
        """
        从 CSD 矩阵提取上三角元素作为 Token
        
        支持两种输入格式：
        1. 复数矩阵 (B, 16, 16) complex
        2. 实数化矩阵 (B, 32, 32) real（需要转换回复数）
        
        Args:
            csd_matrix: CSD 矩阵
            
        Returns:
            tokens: (B, n_pairs, token_dim)
        """
        B = csd_matrix.size(0)
        
        # 处理实数化的 32×32 矩阵
        if csd_matrix.size(-1) == 32 and not csd_matrix.is_complex():
            # 假设是块结构 [[Re, -Im], [Im, Re]]
            # 提取复数矩阵
            n = self.matrix_size
            real_part = csd_matrix[:, :n, :n]
            imag_part = csd_matrix[:, n:, :n]
            csd_complex = torch.complex(real_part, imag_part)
        elif csd_matrix.is_complex():
            csd_complex = csd_matrix
        else:
            # 假设是 16×16 实数矩阵，虚部为 0
            csd_complex = torch.complex(csd_matrix, torch.zeros_like(csd_matrix))
        
        # 提取上三角元素
        # shape: (B, n_pairs)
        upper_tri = csd_complex[:, self.triu_rows, self.triu_cols]
        
        # 构建 Token 特征
        real = upper_tri.real  # (B, n_pairs)
        imag = upper_tri.imag  # (B, n_pairs)
        
        if self.token_dim >= 4:
            # [Re, Im, Mag, Phase]
            mag = torch.abs(upper_tri)
            phase = torch.angle(upper_tri)
            tokens = torch.stack([real, imag, mag, phase], dim=-1)  # (B, n_pairs, 4)
        else:
            # [Re, Im]
            tokens = torch.stack([real, imag], dim=-1)  # (B, n_pairs, 2)
        
        # 确保数据类型为 float32（模型权重是 float32）
        tokens = tokens.float()
        
        return tokens
    
    def forward(
        self,
        csd_matrix: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            csd_matrix: CSD 矩阵
                - 复数格式: (B, 16, 16) complex
                - 实数格式: (B, 32, 32) real
            return_attention: 是否返回注意力权重（用于可视化）
            
        Returns:
            特征向量 (B, output_dim)
            或 (特征向量, 注意力权重) 如果 return_attention=True
        """
        B = csd_matrix.size(0)
        
        # 1. 提取 Token
        tokens = self._extract_tokens(csd_matrix)  # (B, n_pairs, token_dim)
        
        # 2. Token 嵌入
        x = self.token_embedding(tokens)  # (B, n_pairs, d_model)
        
        # 3. 添加位置编码
        pos_emb = self.position_encoding(B)  # (B, n_pairs, d_model)
        x = x + pos_emb
        
        # 4. 添加 CLS Token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, n_pairs+1, d_model)
        
        # 5. Transformer Encoder
        x = self.transformer(x)  # (B, n_pairs+1, d_model)
        
        # 6. 聚合
        if self.use_cls_token:
            # 使用 CLS Token
            output = x[:, 0, :]  # (B, d_model)
        else:
            # Global Average Pooling
            output = x.mean(dim=1)  # (B, d_model)
        
        # 7. Layer Norm
        output = self.norm(output)
        
        # 8. 输出投影
        if self.projection is not None:
            output = self.projection(output)
        
        return output
    
    def get_attention_weights(self, csd_matrix: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重用于可视化
        
        Args:
            csd_matrix: CSD 矩阵
            
        Returns:
            attention_weights: (B, n_layers, n_heads, seq_len, seq_len)
        """
        B = csd_matrix.size(0)
        
        # 准备输入
        tokens = self._extract_tokens(csd_matrix)
        x = self.token_embedding(tokens)
        pos_emb = self.position_encoding(B)
        x = x + pos_emb
        
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # 收集注意力权重
        attention_weights = []
        
        for layer in self.transformer.layers:
            # 手动调用 self-attention 并获取权重
            # 注意：这需要 PyTorch 版本支持
            attn_output, attn_weights = layer.self_attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=False
            )
            attention_weights.append(attn_weights)
            
            # 继续前向传播
            x = layer(x)
        
        return torch.stack(attention_weights, dim=1)


class CSDTransformerEncoderWithProjection(nn.Module):
    """
    带投影层的 CSD Transformer 编码器
    
    封装 CSDTransformerEncoder，确保输出维度与融合层兼容。
    
    Args:
        与 CSDTransformerEncoder 相同
        output_dim: 最终输出维度
    """
    
    def __init__(
        self,
        matrix_size: int = 16,
        token_dim: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        output_dim: int = 128
    ):
        super().__init__()
        
        self.encoder = CSDTransformerEncoder(
            matrix_size=matrix_size,
            token_dim=token_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_cls_token=use_cls_token,
            projection_dim=output_dim
        )
        
        self.output_dim = output_dim
    
    def forward(self, csd_matrix: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.encoder(csd_matrix)

