"""
Spatial-Aware Attention Module

实现结合特征相似度和空间距离的注意力机制
用于Leiden原型表示的空间感知建模

Author: Based on DMSS paper architecture
Date: 2025-01
"""

import torch
import torch.nn as nn
import numpy as np


class SpatialAwareAttention(nn.Module):
    """
    空间感知注意力层
    
    核心思想：
    1. 计算特征注意力分数（Q·K^T）
    2. 利用原型的空间坐标计算空间相似度
    3. 加权融合：attn = (1-α)*feat_attn + α*spatial_attn
    
    Args:
        dim (int): 特征维度
        spatial_centers (np.ndarray): 原型空间坐标 (n_proto, 2)
        heads (int): 多头注意力的头数
        spatial_weight (float): 空间权重初始值（会变成可学习参数）
        spatial_sigma (float): 高斯核的sigma（控制空间相似度衰减速度）
    
    Example:
        >>> spatial_centers = np.random.rand(400, 2) * 1000
        >>> attn = SpatialAwareAttention(dim=512, spatial_centers=spatial_centers)
        >>> x = torch.randn(2, 401, 512)  # (batch, 1+n_proto, dim)
        >>> out = attn(x)
    """
    
    def __init__(self, dim, spatial_centers, heads=8, 
                 spatial_weight=0.3, spatial_sigma=None):
        super(SpatialAwareAttention, self).__init__()
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        
        # QKV投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # LayerNorm（用于残差连接）
        self.norm = nn.LayerNorm(dim)
        
        # === 🔥 空间相似度计算 ===
        n_proto = spatial_centers.shape[0]
        spatial_centers_tensor = torch.from_numpy(spatial_centers).float()
        
        # 计算空间欧氏距离矩阵
        spatial_dist = torch.cdist(spatial_centers_tensor, spatial_centers_tensor)
        # spatial_dist shape: (n_proto, n_proto)
        
        # 自适应确定sigma（如果未指定）
        if spatial_sigma is None:
            spatial_sigma = spatial_dist.mean().item()
            if spatial_sigma < 1e-6:
                spatial_sigma = 1.0  # 防止除零
        
        # 🔥 使用高斯核将距离转换为相似度
        # similarity = exp(-distance^2 / (2 * sigma^2))
        spatial_sim = torch.exp(-spatial_dist ** 2 / (2 * spatial_sigma ** 2))
        
        # 归一化到 [0, 1] 范围
        max_sim = spatial_sim.max()
        if max_sim > 1e-6:
            spatial_sim = spatial_sim / max_sim
        
        # 🔥 扩展以包含 cls_token
        # 最终矩阵大小: (n_proto+1, n_proto+1)
        extended_spatial = torch.zeros(n_proto + 1, n_proto + 1)
        extended_spatial[0, :] = 1.0  # cls_token 与所有token全连接
        extended_spatial[:, 0] = 1.0
        extended_spatial[1:, 1:] = spatial_sim  # 原型部分使用空间相似度
        
        # 注册为buffer（不参与梯度更新）
        self.register_buffer('spatial_similarity', extended_spatial)
        
        # 🔥 可学习的空间权重
        # 使用logit形式，通过sigmoid映射到[0,1]
        initial_logit = self._inverse_sigmoid(spatial_weight)
        self.spatial_weight_logit = nn.Parameter(torch.tensor(initial_logit))
        
        # 保存配置信息
        self.n_proto = n_proto
        self.spatial_sigma = spatial_sigma
        
        print(f"[SpatialAwareAttention] Initialized")
        print(f"  - Number of prototypes: {n_proto}")
        print(f"  - Spatial sigma: {spatial_sigma:.2f}")
        print(f"  - Initial spatial weight: {spatial_weight:.3f}")
        print(f"  - Attention heads: {heads}")
    
    @staticmethod
    def _inverse_sigmoid(x):
        """sigmoid的反函数，用于初始化logit"""
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征 (B, N, D)
                - B: batch size
                - N: 序列长度 = 1(cls_token) + n_proto + padding
                - D: 特征维度
        
        Returns:
            torch.Tensor: 输出特征 (B, N, D)
        """
        B, N, D = x.shape
        
        # === 1. QKV投影和重塑 ===
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # === 2. 计算特征注意力分数 ===
        attn_feat = (q @ k.transpose(-2, -1)) * self.scale
        # attn_feat shape: (B, heads, N, N)
        
        # === 3. 构建空间注意力 ===
        n_valid = self.spatial_similarity.shape[0]  # cls + n_proto
        
        if N >= n_valid:
            # 处理padding情况
            spatial_attn = torch.zeros(N, N, device=x.device, dtype=x.dtype)
            
            # 有效部分使用预计算的空间相似度
            spatial_attn[:n_valid, :n_valid] = self.spatial_similarity
            
            # padding部分：只与自己连接（对角矩阵）
            if N > n_valid:
                spatial_attn[n_valid:, n_valid:] = torch.eye(
                    N - n_valid, device=x.device, dtype=x.dtype
                )
        else:
            # 如果N < n_valid（理论上不会发生）
            spatial_attn = self.spatial_similarity[:N, :N]
        
        # === 4. 🔥 混合特征和空间注意力 ===
        # 获取当前的空间权重（可学习）
        alpha = torch.sigmoid(self.spatial_weight_logit)
        
        # 混合公式：attn = (1-α)*feat + α*spatial
        # spatial_attn需要扩展到 (B, heads, N, N)
        spatial_attn_expanded = spatial_attn.unsqueeze(0).unsqueeze(0)
        
        attn = (1 - alpha) * attn_feat + alpha * spatial_attn_expanded
        
        # === 5. Softmax归一化 ===
        attn = attn.softmax(dim=-1)
        # attn shape: (B, heads, N, N)
        
        # === 6. 加权聚合 ===
        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2)  # (B, N, heads, head_dim)
        out = out.reshape(B, N, D)  # (B, N, D)
        
        # === 7. 输出投影 ===
        out = self.proj(out)
        
        # === 8. 残差连接 ===
        out = out + self.norm(x)
        
        return out
    
    def get_spatial_weight(self):
        """
        获取当前的空间权重值（用于监控和调试）
        
        Returns:
            float: 当前空间权重，范围[0, 1]
        """
        return torch.sigmoid(self.spatial_weight_logit).item()
    
    def extra_repr(self):
        """额外的模块信息，用于print(model)"""
        return f'dim={self.dim}, heads={self.heads}, n_proto={self.n_proto}, ' \
               f'spatial_weight={self.get_spatial_weight():.3f}'


# ============================================================
# 可选：轻量级版本（如果方案A太重或遇到问题）
# ============================================================

class SpatialBiasedAttention(nn.Module):
    """
    轻量级空间偏置注意力
    
    在标准注意力基础上添加空间距离偏置，而不是完全混合
    计算量更小，但效果可能略逊于SpatialAwareAttention
    
    Args:
        dim (int): 特征维度
        spatial_centers (np.ndarray): 原型空间坐标 (n_proto, 2)
        heads (int): 多头注意力的头数
        bias_strength (float): 空间偏置强度
    """
    
    def __init__(self, dim, spatial_centers, heads=8, bias_strength=0.1):
        super(SpatialBiasedAttention, self).__init__()
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
        # 计算空间偏置
        n_proto = spatial_centers.shape[0]
        spatial_centers_tensor = torch.from_numpy(spatial_centers).float()
        spatial_dist = torch.cdist(spatial_centers_tensor, spatial_centers_tensor)
        
        # 转换为偏置项（距离越近，偏置越大）
        # 归一化到 [-1, 0] 范围
        max_dist = spatial_dist.max()
        if max_dist > 1e-6:
            spatial_bias = -spatial_dist / max_dist
        else:
            spatial_bias = torch.zeros_like(spatial_dist)
        
        spatial_bias = spatial_bias * bias_strength  # 控制强度
        
        # 扩展包含cls_token
        extended_bias = torch.zeros(n_proto + 1, n_proto + 1)
        extended_bias[1:, 1:] = spatial_bias
        
        self.register_buffer('spatial_bias', extended_bias)
        self.n_proto = n_proto
        
        print(f"[SpatialBiasedAttention] Initialized")
        print(f"  - Number of prototypes: {n_proto}")
        print(f"  - Bias strength: {bias_strength:.3f}")
        print(f"  - Attention heads: {heads}")
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, N, D)
        
        Returns:
            torch.Tensor: (B, N, D)
        """
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 特征注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 🔥 添加空间偏置
        n_valid = self.spatial_bias.shape[0]
        if N >= n_valid:
            bias = torch.zeros(N, N, device=x.device, dtype=x.dtype)
            bias[:n_valid, :n_valid] = self.spatial_bias
            
            # padding部分不添加偏置
            attn = attn + bias.unsqueeze(0).unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        
        return out + self.norm(x)
    
    def extra_repr(self):
        return f'dim={self.dim}, heads={self.heads}, n_proto={self.n_proto}'


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SpatialAwareAttention")
    print("=" * 60)
    
    # 创建假数据
    batch_size = 2
    n_proto = 400
    dim = 512
    
    # 模拟原型空间坐标（在1000x1000的图像上）
    np.random.seed(42)
    spatial_centers = np.random.rand(n_proto, 2) * 1000
    
    # 初始化模块
    attn = SpatialAwareAttention(
        dim=dim,
        spatial_centers=spatial_centers,
        heads=8,
        spatial_weight=0.3
    )
    
    # 创建输入（包含cls_token）
    x = torch.randn(batch_size, n_proto + 1, dim)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Initial spatial weight: {attn.get_spatial_weight():.4f}")
    
    # 前向传播
    try:
        out = attn(x)
        print(f"✅ Forward pass successful!")
        print(f"Output shape: {out.shape}")
        
        # 检查输出形状
        assert out.shape == x.shape, "Output shape mismatch!"
        print("✅ Shape verification passed!")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Testing SpatialBiasedAttention")
    print("=" * 60)
    
    # 测试轻量级版本
    attn_lite = SpatialBiasedAttention(
        dim=dim,
        spatial_centers=spatial_centers,
        heads=8,
        bias_strength=0.1
    )
    
    try:
        out_lite = attn_lite(x)
        print(f"✅ Lite version forward pass successful!")
        print(f"Output shape: {out_lite.shape}")
        
    except Exception as e:
        print(f"❌ Lite version forward pass failed: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)