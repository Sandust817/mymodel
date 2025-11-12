import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from typing import Optional

# ----------------------------
# Prototype-Augmented Transformer Backbone
# ----------------------------
class PrototypeCrossAttention(nn.Module):
    """
    Cross-Attention between transformer features and prototypes.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, prototypes: torch.Tensor):
        """
        x: [B, N, D]
        prototypes: [K, D]
        """
        # repeat prototypes across batch
        p = repeat(prototypes, "k d -> b k d", b=x.size(0))
        # Cross-attention: query = x, key/value = prototypes
        attn_out, _ = self.attn(query=x, key=p, value=p)
        # residual connection
        x = self.norm(x + attn_out)
        return x


class PNPTransformerBackbone(nn.Module):
    """
    Transformer backbone without patch embedding.
    """
    def __init__(self, d_model, n_heads, e_layers, dropout, seq_len):
        super().__init__()
        self.embed_dim = d_model
        self.seq_len = seq_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)

        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)

    def forward(self, x):
        """
        x: [B, C, L]
        """
        B, C, L = x.shape
        # Flatten channels to feature dim
        x = x.transpose(1, 2)  # [B, L, C]
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = self.norm(x)
        return x  # [B, L, D]
# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F

# class ViTPatchEmbedding(nn.Module):
#     """
#     Standard ViT-style Patch Embedding for 1D sequences.
#     Input:  [B, C, L]
#     Output: [B, n_patches, embed_dim]
#     """
#     def __init__(self, seq_len: int, patch_size: int, in_channels: int, embed_dim: int):
#         super().__init__()
#         self.seq_len = seq_len
#         self.patch_size = patch_size
#         self.embed_dim = embed_dim
#         self.in_channels = in_channels

#         # Conv1d acts as patchify + linear projection
#         self.proj = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size
#         )

#         # compute number of patches after padding
#         self.n_patches = math.ceil(seq_len / patch_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, C, L]
#         returns: [B, n_patches, embed_dim]
#         """
#         B, C, L = x.shape
#         pad_len = (self.patch_size - L % self.patch_size) % self.patch_size
#         if pad_len > 0:
#             x = F.pad(x, (0, pad_len), mode='constant', value=0)
#         x = self.proj(x)               # [B, embed_dim, n_patches]
#         x = x.transpose(1, 2)          # [B, n_patches, embed_dim]
#         return x


# class PrototypeEnhancedEncoderLayer(nn.Module):
#     def __init__(self, d_model, n_heads, dim_feedforward, dropout, num_prototypes):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
#         self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.num_prototypes = num_prototypes

#     def forward(self, src, prototypes):
#         """
#         src: [B, L, D] - 输入序列
#         prototypes: [K, D] - 原型矩阵
#         """
#         # Self-attention over input sequence
#         src2 = self.norm1(src)
#         src2, _ = self.self_attn(src2, src2, src2)
#         src = src + self.dropout1(src2)

#         # Cross-attention: sequence queries attend to prototypes
#         src2 = self.norm2(src)
#         prototypes_expanded = prototypes.unsqueeze(0).expand(src.size(0), -1, -1)  # [B, K, D]
#         src2, _ = self.cross_attn(src2, prototypes_expanded, prototypes_expanded)
#         src = src + self.dropout2(src2)

#         # Feedforward
#         src2 = self.norm3(src)
#         src2 = self.linear2(self.dropout(self.linear1(src2)))
#         src = src + self.dropout3(src2)
#         return src


# class PrototypeEnhancedTransformerBackbone(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.enc_in = config.d_model 
#         self.seq_len = config.seq_len
#         self.embed_dim = config.d_model
#         self.num_layers = config.e_layers
#         self.dropout = config.dropout
#         self.num_heads = config.n_heads
        
#         # 从 config 获取原型数量
#         self.k = getattr(config, "k", 5)
#         self.num_class = config.num_class
#         self.num_prototypes = self.k * self.num_class

#         # Patching config
#         self.use_patch = getattr(config, "use_patch", False)  # 默认使用 patch
#         self.patch_size = getattr(config, "patch_size", 16)

#         if self.use_patch:
#             # Patch 模式：ViT 风格
#             self.patch_embed = ViTPatchEmbedding(
#                 seq_len=config.seq_len,
#                 patch_size=self.patch_size,
#                 in_channels=config.d_model,  # feature_extractor 输出通道
#                 embed_dim=self.embed_dim,
#             )
#             self.seq_len_processed = self.patch_embed.n_patches
#         else:
#             # 非 Patch 模式：直接投影
#             self.input_proj = nn.Linear(config.d_model, self.embed_dim)
#             self.seq_len_processed = config.seq_len

#         # Positional encoding
#         self.pos_encoding = nn.Parameter(torch.zeros(1, self.seq_len_processed, self.embed_dim))
#         nn.init.normal_(self.pos_encoding, std=0.02)

#         # Prototype-enhanced transformer layers
#         self.layers = nn.ModuleList([
#             PrototypeEnhancedEncoderLayer(
#                 d_model=self.embed_dim,
#                 n_heads=self.num_heads,
#                 dim_feedforward=self.embed_dim * 4,
#                 dropout=self.dropout,
#                 num_prototypes=self.num_prototypes
#             ) for _ in range(self.num_layers)
#         ])
#         self.norm = nn.LayerNorm(self.embed_dim)

#     def forward(self, x, prototypes):
#         """
#         x: [B, C, L] - feature extractor output
#         prototypes: [K, D] - normalized prototypes
#         returns: [B, D] - final features
#         """
#         B, C, L = x.shape
        
#         if self.use_patch:
#             # Patch 模式
#             x = self.patch_embed(x)  # [B, n_patches, embed_dim]
#         else:
#             # 非 Patch 模式：转置并投影
#             x = x.transpose(1, 2)    # [B, L, C]
#             x = self.input_proj(x)   # [B, L, embed_dim]
        
#         # Add positional encoding
#         x = x + self.pos_encoding  # [B, seq_len_processed, D]

#         # Apply prototype-enhanced transformer layers
#         for layer in self.layers:
#             x = layer(x, prototypes)

#         x = self.norm(x)
        
#         # Global average pooling (works for both patch and non-patch)
#         features = x.mean(dim=1)  # [B, D]
#         return features