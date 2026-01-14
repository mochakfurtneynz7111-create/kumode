## DMSS的components.py
import torch.nn as nn
import torch
from torch import einsum
from tqdm import tqdm
from einops import rearrange, reduce

from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from sksurv.util import Surv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import *
from math import ceil

import numpy as np



import torch
from torch import Tensor

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module
from typing import Optional



def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# src/mil_models/components.py
# 在文件末尾，SpatialAwareAttention类之后添加：

class SpatialWeightedAttention(nn.Module):
    """
    空间加权注意力层 - 修复版
    
    关键修复：
    1. ✅ 残差连接顺序：先norm(x)再attn，最后 x + attn_out
    2. ✅ 重要性加权：使用log而非倒数，避免极端值
    3. ✅ 可选禁用重要性加权
    """
    
    def __init__(self, dim, spatial_centers, spatial_spreads=None, heads=8, 
                 spatial_weight=0.3, spatial_sigma=None, 
                 use_importance_weighting=False):  # ← 默认改为False
        super(SpatialWeightedAttention, self).__init__()
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.use_importance_weighting = use_importance_weighting
        
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        
        # QKV投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # ✅ LayerNorm（先归一化）
        self.norm = nn.LayerNorm(dim)
        
        # === 空间相似度计算 ===
        n_proto = spatial_centers.shape[0]
        spatial_centers_tensor = torch.from_numpy(spatial_centers).float()
        
        # 计算空间欧氏距离矩阵
        spatial_dist = torch.cdist(spatial_centers_tensor, spatial_centers_tensor)
        
        # 自适应确定sigma
        if spatial_sigma is None:
            spatial_sigma = spatial_dist.mean().item()
            if spatial_sigma < 1e-6:
                spatial_sigma = 1.0
        
        # 高斯核相似度
        spatial_sim = torch.exp(-spatial_dist ** 2 / (2 * spatial_sigma ** 2))
        
        # 归一化
        max_sim = spatial_sim.max()
        if max_sim > 1e-6:
            spatial_sim = spatial_sim / max_sim
        
        # === 🔥 原型重要性计算（修复版）===
        if use_importance_weighting and spatial_spreads is not None:
            spreads_tensor = torch.from_numpy(spatial_spreads).float()
            
            # 🔥 方法1: 使用log而非倒数（更稳定）
            # importance = -log(spread + 1)，spread小 → log接近0 → importance高
            proto_importance = -torch.log(spreads_tensor + 1.0)
            
            # 归一化到 [0, 1]
            proto_importance = proto_importance - proto_importance.min()  # 平移到非负
            proto_importance = proto_importance / (proto_importance.max() + 1e-6)
            
            # 🔥 重要性矩阵
            importance_matrix = proto_importance.unsqueeze(0) * proto_importance.unsqueeze(1)
            
            # 🔥 加权空间相似度（使用加法混合而非乘法）
            # 混合系数：0.7来自空间相似度，0.3来自重要性
            spatial_sim_weighted = 0.7 * spatial_sim + 0.3 * importance_matrix
            
            print(f"[SpatialWeightedAttention] Using importance weighting (log method)")
            print(f"  - Spread range: [{spreads_tensor.min():.2f}, {spreads_tensor.max():.2f}]")
            print(f"  - Importance range: [{proto_importance.min():.3f}, {proto_importance.max():.3f}]")
        else:
            spatial_sim_weighted = spatial_sim
            proto_importance = None
            print(f"[SpatialWeightedAttention] NOT using importance weighting")
        
        # === 扩展到包含cls_token ===
        extended_spatial = torch.zeros(n_proto + 1, n_proto + 1)
        extended_spatial[0, :] = 1.0
        extended_spatial[:, 0] = 1.0
        extended_spatial[1:, 1:] = spatial_sim_weighted
        
        self.register_buffer('spatial_similarity', extended_spatial)
        
        if proto_importance is not None:
            importance_with_cls = torch.cat([
                torch.ones(1),
                proto_importance
            ])
            self.register_buffer('proto_importance', importance_with_cls)
        else:
            self.proto_importance = None
        
        # 可学习的空间权重
        initial_logit = self._inverse_sigmoid(spatial_weight)
        self.spatial_weight_logit = nn.Parameter(torch.tensor(initial_logit))
        
        self.n_proto = n_proto
        self.spatial_sigma = spatial_sigma
        
        print(f"[SpatialWeightedAttention] Initialized")
        print(f"  - Number of prototypes: {n_proto}")
        print(f"  - Spatial sigma: {spatial_sigma:.2f}")
        print(f"  - Initial spatial weight: {spatial_weight:.3f}")
        print(f"  - Attention heads: {heads}")
    
    @staticmethod
    def _inverse_sigmoid(x):
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))
    
    def forward(self, x):
        """
        ✅ 修复后的前向传播
        顺序：norm → attn → 残差
        """
        B, N, D = x.shape
        
        # ✅ 1. 先归一化
        x_norm = self.norm(x)
        
        # 2. QKV投影
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 3. 计算特征注意力分数
        attn_feat = (q @ k.transpose(-2, -1)) * self.scale
        
        # 4. 构建空间注意力
        n_valid = self.spatial_similarity.shape[0]
        
        if N >= n_valid:
            spatial_attn = torch.zeros(N, N, device=x.device, dtype=x.dtype)
            spatial_attn[:n_valid, :n_valid] = self.spatial_similarity
            
            if N > n_valid:
                spatial_attn[n_valid:, n_valid:] = torch.eye(
                    N - n_valid, device=x.device, dtype=x.dtype
                )
        else:
            spatial_attn = self.spatial_similarity[:N, :N]
        
        # 5. 混合注意力
        alpha = torch.sigmoid(self.spatial_weight_logit)
        spatial_attn_expanded = spatial_attn.unsqueeze(0).unsqueeze(0)
        attn = (1 - alpha) * attn_feat + alpha * spatial_attn_expanded
        
        # 6. Softmax归一化
        attn = attn.softmax(dim=-1)
        
        # 7. 加权聚合
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        
        # 8. 输出投影
        out = self.proj(out)
        
        # ✅ 9. 残差连接（正确顺序）
        return x + out  # ← 关键：x + attn_out，而非 attn_out + norm(x)
    
    def get_learned_weights(self):
        alpha = torch.sigmoid(self.spatial_weight_logit).item()
        return {
            'spatial_weight': alpha,
            'proto_importance': self.proto_importance if self.proto_importance is not None else None
        }

# main attention class
class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=32,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.0,
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out
# helper functions
def exists(val):
    return val is not None
class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_landmarks=32,
        pinv_iterations=6,
        attn_values_residual=True,
        attn_values_residual_conv_kernel=33,
        attn_dropout=0.0,
        ff_dropout=0.0
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            NystromAttention(
                                dim=dim,
                                dim_head=dim_head,
                                heads=heads,
                                num_landmarks=num_landmarks,
                                pinv_iterations=pinv_iterations,
                                residual=attn_values_residual,
                                residual_conv_kernel=attn_values_residual_conv_kernel,
                                dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim=dim, dropout=ff_dropout)),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

# src/mil_models/components.py

class GraphGuidedAttention(nn.Module):
    """
    基于Leiden邻居图的图引导注意力
    创新点：使用可学习的图权重增强Leiden拓扑结构
    """
    def __init__(self, dim=512, adjacency_matrix=None, heads=8, 
                 learnable_graph=True, init_edge_weight=2.0, init_non_edge_weight=0.5):
        super(GraphGuidedAttention, self).__init__()
        
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.learnable_graph = learnable_graph
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # 🔥 创新点：可学习的图权重
        if adjacency_matrix is not None:
            n_proto = adjacency_matrix.shape[0]
            
            # 扩展邻接矩阵（包含cls_token）
            extended_adj = np.zeros((n_proto + 1, n_proto + 1))
            extended_adj[0, :] = 1
            extended_adj[:, 0] = 1
            proto_adj_with_self = adjacency_matrix + np.eye(n_proto)
            extended_adj[1:, 1:] = proto_adj_with_self
            
            if learnable_graph:
                # 🔥 可学习版本：边和非边都有初始权重
                # 边 → init_edge_weight (例如2.0)
                # 非边 → init_non_edge_weight (例如0.5)
                init_weights = np.where(
                    extended_adj > 0,
                    init_edge_weight,      # Leiden邻居：初始权重高
                    init_non_edge_weight   # 非邻居：初始权重低但非零
                )
                
                # 🔥 注册为可学习参数
                self.graph_weights = nn.Parameter(
                    torch.from_numpy(init_weights).float()
                )
                
                print(f"[GraphGuidedAttention] LEARNABLE graph weights")
                print(f"  - Edges init: {init_edge_weight}")
                print(f"  - Non-edges init: {init_non_edge_weight}")
                print(f"  - Total parameters: {(n_proto+1)**2}")
            else:
                # 固定版本（当前实现）
                self.register_buffer('graph_weights', 
                                   torch.from_numpy(extended_adj).float())
                print(f"[GraphGuidedAttention] FIXED graph mask")
            
            print(f"[GraphGuidedAttention] Graph: {n_proto} nodes, "
                  f"{adjacency_matrix.sum():.0f} edges")
        else:
            self.graph_weights = None
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 🔥 应用可学习的图权重
        if self.graph_weights is not None:
            # 使用softplus确保权重为正
            # softplus(x) = log(1 + e^x)，平滑且总是>0
            graph_mask = torch.nn.functional.softplus(self.graph_weights)
            
            # 扩展到 batch 和 heads 维度
            graph_mask = graph_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            
            # 🔥 乘法mask（而非masked_fill）
            # 这样梯度可以流动，模型可以学习调整
            attn = attn * graph_mask
        
        attn = attn.softmax(dim=-1)
        
        # 聚合
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        
        return x + self.norm(x)

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class PrototypePPEG(nn.Module):
    """
    Prototype-aware Position Encoding Generator
    利用原型的真实空间位置信息
    """
    def __init__(self, dim=512, num_prototypes=400, 
                 spatial_centers=None, use_spatial_bias=True):
        super(PrototypePPEG, self).__init__()
        
        # 标准PPEG卷积（多尺度）
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)
        
        # 🔥 保存原型数量
        self.num_prototypes = num_prototypes
        
        # 空间位置编码
        self.use_spatial_bias = use_spatial_bias
        if use_spatial_bias and spatial_centers is not None:
            self.register_buffer('spatial_centers', 
                               torch.from_numpy(spatial_centers).float())
            
            # 可学习的空间编码MLP
            self.spatial_encoder = nn.Sequential(
                nn.Linear(2, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, dim)
            )
            print(f"[PrototypePPEG] Initialized with {num_prototypes} spatial centers")
        else:
            self.spatial_centers = None
            self.spatial_encoder = None
            print(f"[PrototypePPEG] Initialized without spatial bias")

    def forward(self, x, H, W):
        # x: (B, N, D) where N = 1(cls) + n_proto + padding
        B, N, C = x.shape
        
        # 分离 cls_token 和 features
        cls_token, feat_token = x[:, 0], x[:, 1:]  # (B, 1, D), (B, N-1, D)
        
        # === 标准PPEG：2D卷积位置编码 ===
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x_pos = self.proj(cnn_feat) + cnn_feat + \
                self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x_pos = x_pos.flatten(2).transpose(1, 2)  # (B, N-1, D)
        
        # === 🔥 新增：空间位置偏置 ===
        if self.use_spatial_bias and self.spatial_encoder is not None:
            # 利用原型的实际空间坐标
            spatial_bias = self.spatial_encoder(self.spatial_centers)  # (n_proto, D)
            spatial_bias = spatial_bias.unsqueeze(0)  # (1, n_proto, D)
            
            # 🔥 只对有效原型部分添加空间偏置
            # x_pos: (B, N-1, D) where N-1 = n_proto + padding
            n_proto = spatial_bias.shape[1]  # 409
            
            # 只给前 n_proto 个 token 添加空间偏置
            x_pos[:, :n_proto, :] = x_pos[:, :n_proto, :] + spatial_bias
            # padding 部分 x_pos[:, n_proto:, :] 保持不变
        
        # 重新拼接 cls token
        x = torch.cat((cls_token.unsqueeze(1), x_pos), dim=1)
        return x
  
    
class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512, leiden_info=None, config=None):
        super(Transformer_P, self).__init__()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        
        # 默认配置（全部禁用）
        if config is None:
            config = {}
        
        use_graph_attn = config.get('use_graph_attn', False)
        use_spatial_ppeg = config.get('use_spatial_ppeg', False)
        use_spatial_attn = config.get('use_spatial_attn', False)
        
        # 提取Leiden信息
        if leiden_info is not None:
            self.n_proto = leiden_info.get('n_proto')
            feature_adj = leiden_info.get('feature_adjacency')
            spatial_centers = leiden_info.get('spatial_centers')
        else:
            self.n_proto = None
            feature_adj = None
            spatial_centers = None
        
        # Layer 1
        if use_graph_attn and feature_adj is not None:
            """
            # 在Transformer_P中
            self.layer1 = GraphGuidedAttention(
                dim=feature_dim,
                adjacency_matrix=leiden_info['feature_adjacency'],
                learnable_graph=True,        # ← 启用可学习
                init_edge_weight=2.0,        # Leiden邻居初始权重
                init_non_edge_weight=0.5     # 非邻居初始权重
            )
            """
            self.layer1 = TransLayer(dim=feature_dim)
            self.use_leiden = True
        else:
            self.layer1 = TransLayer(dim=feature_dim)
            self.use_leiden = False
        
        # PPEG
        if use_spatial_ppeg and spatial_centers is not None:
            self.pos_layer = PrototypePPEG(
                dim=feature_dim,
                num_prototypes=leiden_info['n_proto'],
                spatial_centers=leiden_info['spatial_centers'],
                use_spatial_bias=True
            )
        elif spatial_centers is not None:
            # 有spatial_centers但不用
            self.pos_layer = PrototypePPEG(
                dim=feature_dim,
                num_prototypes=self.n_proto,
                spatial_centers=spatial_centers,
                use_spatial_bias=False
            )
        else:
            self.pos_layer = PPEG(dim=feature_dim)
        
        # Layer 2
        if use_spatial_attn and spatial_centers is not None:
            self.layer2 = SpatialWeightedAttention(
                dim=feature_dim,
                spatial_centers=leiden_info['spatial_centers'],
                heads=8
            )
        else:
            self.layer2 = TransLayer(dim=feature_dim)
        
        self.norm = nn.LayerNorm(feature_dim)
        
        # 打印配置
        print(f"[Transformer_P] Config: graph={use_graph_attn}, "
              f"spatial_ppeg={use_spatial_ppeg}, spatial_attn={use_spatial_attn}")

    def forward(self, features):
        # features: (B, H, D)
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        
        # Pad features
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # (B, H+pad, D)
        
        # Add cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)  # (B, 1+H+pad, D)
        
        # 🔥 如果使用 Leiden，只对有效原型应用图注意力
        if self.use_leiden:
            # 分离 cls_token, 有效原型, padding
            cls_tok = h[:, :1, :]  # (B, 1, D)
            valid_protos = h[:, 1:self.n_proto+1, :]  # (B, n_proto, D)
            padding = h[:, self.n_proto+1:, :]  # (B, pad, D)
            
            # 只对有效原型应用图注意力
            valid_with_cls = torch.cat([cls_tok, valid_protos], dim=1)
            valid_attended = self.layer1(valid_with_cls)
            
            # 重新组合（padding 不经过图注意力）
            h = torch.cat([valid_attended, padding], dim=1)
        else:
            # 标准路径
            h = self.layer1(h)
        
        # PPEG
        h = self.pos_layer(h, _H, _W)
        
        # Layer 2
        h = self.layer2(h)
        
        # Normalization
        h = self.norm(h)
        
        return h[:, 0], h[:, 1:]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_G, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]

class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, need_raw=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_raw=need_raw,
                attn_mask=attn_mask)
        

def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, end_with_fc=True, bias=True):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        mlp = nn.Sequential(*layers)
    return mlp

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    need_raw: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            need_raw=need_raw,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    
    attn_output_weights_raw = attn_output_weights
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    
    if need_weights:
        if need_raw:
            
            attn_output_weights_raw = attn_output_weights_raw.view(bsz, num_heads, tgt_len, src_len)
            return attn_output,attn_output_weights_raw
            
            #attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            #return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_raw, attn_output_weights_raw.sum(dim=1) / num_heads
        else:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
    

#
# Multimodal components (Some of the functions were adapted from SurvPath)
#
class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mult), int(dim * mult))
        )

    def forward(self, x):
        return self.net(self.norm(x))


class FeedForwardEnsemble(nn.Module):
    def __init__(self, dim, mult=1, dropout=0., num=16):
        super().__init__()
        self.num = num
        self.norm = nn.LayerNorm(dim)
        self.net = nn.ModuleList([FeedForward(dim, mult, dropout) for _ in range(num)])

    def forward(self, x):
        """
        Args:
            x: (B, proto, d)
        """
        # assert x.shape[1] == self.num
        out = []
        for idx in range(x.shape[1]):
            out.append(self.net[idx](x[:,idx:idx+1,:]))
        out = torch.cat(out, dim=1)

        return out

class MMAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            residual=True,
            residual_conv_kernel=33,
            eps=1e-8,
            dropout=0.,
            num_pathways=281,
            attn_mode='full'
    ):
        """

        Args:
            dim:
            dim_head:
            heads:
            residual:
            residual_conv_kernel:
            eps:
            dropout:
            num_pathways:
            attn_mode: ['full', 'partial', 'cross']
                'full': All pairs between P and H
                'partial': P->P, H->P, P->H
                'cross': P->H, H->P
                'self': P->P, H->H
        """
        super().__init__()
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.residual = residual
        self.attn_mode = attn_mode

        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def set_attn_mode(self, attn_mode):
        self.attn_mode = attn_mode

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        k_pathways = k[:, :, :self.num_pathways, :]

        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim
        k_histology = k[:, :, self.num_pathways:, :]

        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)
        attn_histology = einsum(einops_eq, q_histology, k_histology)

        # softmax
        pre_softmax_cross_attn_histology = cross_attn_histology
        if self.attn_mode == 'full': # H->P, P->H, P->P, H->H
            cross_attn_histology = cross_attn_histology.softmax(dim=-1)
            attn_histology_pathways = torch.cat((cross_attn_histology, attn_histology), dim=-1).softmax(dim=-1)
            attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

            # compute output
            out_pathways = attn_pathways_histology @ v
            out_histology = attn_histology_pathways @ v
        elif self.attn_mode == 'cross': # P->H, H->P
            cross_attn_histology = cross_attn_histology.softmax(dim=-1)
            cross_attn_pathways = cross_attn_pathways.softmax(dim=-1)

            # compute output
            out_pathways = cross_attn_pathways @ v[:, :, self.num_pathways:]
            out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]
        elif self.attn_mode == 'self': # P->P, H->H (Late fusion)
            attn_histology = attn_histology.softmax(dim=-1)
            attn_pathways = attn_pathways.softmax(dim=-1)

            out_pathways = attn_pathways @ v[:, :, :self.num_pathways]
            out_histology = attn_histology @ v[:, :, self.num_pathways:]
        elif self.attn_mode == 'partial': # H->P, P->H, P->P (SURVPATH)
            cross_attn_histology = cross_attn_histology.softmax(dim=-1)
            attn_pathways_histology = torch.cat((attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

            # compute output
            out_pathways = attn_pathways_histology @ v
            out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]
        elif self.attn_mode == 'mcat': # P->P, P->H
            cross_attn_pathways = cross_attn_pathways.softmax(dim=-1)

            out_pathways = q_pathways
            out_histology = cross_attn_pathways @ v[:, :, self.num_pathways:]
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}")

        out = torch.cat((out_pathways, out_histology), dim=2)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)

        if return_attn:
            # return three matrices
            return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()

        return out


class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
            self,
            norm_layer=nn.LayerNorm,
            dim=512,
            dim_head=64,
            heads=6,
            residual=True,
            dropout=0.,
            num_pathways=281,
            attn_mode='full'
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn_mode = attn_mode

        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways,
            attn_mode=attn_mode
        )

    def set_attn_mode(self, attn_mode):
        self.attn.set_attn_mode(attn_mode)

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(x=self.norm(x), mask=mask,
                                                                                    return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))



#
# Model processing
#
def predict_emb(self, dataset, use_cuda=True, permute=False):
    """
    Create prototype-based slide representation

    Returns
    - X (torch.Tensor): (n_data x output_set_dim)
    - y (torch.Tensor): (n_data)
    """

    X = []

    for i in tqdm(range(len(dataset))):
        batch = dataset.__getitem__(i)
        data = batch['img'].unsqueeze(dim=0)
        if use_cuda:
            data = data.cuda()
        
        with torch.no_grad():
            out = self.representation(data)
            out = out['repr'].data.detach().cpu()

        X.append(out)

    X = torch.cat(X)

    return X

def predict_surv(self, dataset,  use_cuda=True, permute=False):
    """
    Create prototype-based slide representation
    """

    output = []
    label_output = []
    censor_output = []
    time_output = []

    for i in tqdm(range(len(dataset))):
        batch = dataset.__getitem__(i)
        data, label, censorship, time = batch['img'].unsqueeze(dim=0), batch['label'].unsqueeze(dim=0), batch['censorship'].unsqueeze(dim=0), batch['survival_time'].unsqueeze(dim=0)
        batch_size = data.shape[0]

        if use_cuda:
            data = data.cuda()

        with torch.no_grad():
            batch_out = self.representation(data)
            batch_out = batch_out['repr'].data.cpu()

        output.append(batch_out)
        label_output.append(label)
        censor_output.append(censorship)
        time_output.append(time)

    output = torch.cat(output)
    label_output = torch.cat(label_output)
    censor_output = torch.cat(censor_output)
    time_output = torch.cat(time_output)

    y = Surv.from_arrays(~censor_output.numpy().astype('bool').squeeze(),
                            time_output.numpy().squeeze()
                            )
    
    return output, y


def process_surv(logits, label, censorship, loss_fn=None):
    results_dict = {'logits': logits}
    log_dict = {}

    if loss_fn is not None and label is not None:
        if isinstance(loss_fn, NLLSurvLoss):
            surv_loss_dict = loss_fn(logits=logits, times=label, censorships=censorship)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).unsqueeze(dim=1)
            results_dict.update({'hazards': hazards,
                                    'survival': survival,
                                    'risk': risk})
        elif isinstance(loss_fn, CoxLoss):
            # logits is log risk
            surv_loss_dict = loss_fn(logits=logits, times=label, censorships=censorship)
            risk = torch.exp(logits)
            results_dict['risk'] = risk

        elif isinstance(loss_fn, SurvRankingLoss):
                surv_loss_dict = loss_fn(z=logits, times=label, censorships=censorship)
                results_dict['risk'] = logits

        loss = surv_loss_dict['loss']
        log_dict['surv_loss'] = surv_loss_dict['loss'].item()
        log_dict.update(
            {k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})
        results_dict.update({'loss': loss})

    return results_dict, log_dict

def predict_clf(self, dataset, use_cuda=True, permute=False):
    """
    Create prototype-based slide representation

    Returns
    - X (torch.Tensor): (n_data x output_set_dim)
    - y (torch.Tensor): (n_data)
    """

    X, y = [], []

    for i in tqdm(range(len(dataset))):
        batch = dataset.__getitem__(i)
        data = batch['img'].unsqueeze(dim=0)
        label = batch['label']
        if use_cuda:
            data = data.cuda()
        
        with torch.no_grad():
            out = self.representation(data)
            out = out['repr'].data.detach().cpu()

        X.append(out)
        y.append(label)

    X = torch.cat(X)
    y = torch.Tensor(y).to(torch.long)

    return X, y

def process_clf(logits, label, loss_fn):
    results_dict = {'logits': logits}
    log_dict = {}

    if loss_fn is not None and label is not None:
        cls_loss = loss_fn(logits, label)
        loss = cls_loss
        log_dict.update({
            'cls_loss': cls_loss.item(),
            'loss': loss.item()})
        results_dict.update({'loss': loss})
    
    return results_dict, log_dict


#
# Attention networks
#
class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, n_classes)]

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out

class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid(),
                            nn.Dropout(dropout)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A,x
