# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# vit_pytorch: https://github.com/lucidrains/vit-pytorch
# --------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from models.vit import TransformerBlock


__all__ = ['ST_MEM_ViT', 'st_mem_vit_small', 'st_mem_vit_base']

# 替换原有的 to_patch_embedding 定义
# self.to_patch_embedding = PatchEmbeddingLayer(patch_size, width, stride=patch_size//2 if patch_size > 1 else 1)

# 需要在类中添加这个新的嵌入层类
class PatchEmbeddingLayer(nn.Module):
    def __init__(self, patch_size, embed_dim, stride=None, dropout=0.1):
        super(PatchEmbeddingLayer, self).__init__()  # 修正此处
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride if stride is not None else patch_size
        
        # 使用 ReplicationPad1d 处理不能整除的情况
        self.padding_layer = nn.ReplicationPad1d((0, patch_size - 1))
        
        # 线性投影层
        self.projection = nn.Linear(patch_size, embed_dim)
        
        # LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, channels, sequence_length)
        b, c, seq_len = x.shape
        
        # 添加 padding 保证可以进行 unfold 操作
        x = self.padding_layer(x)
        
        # 使用 unfold 进行 patch 切分
        x = x.unfold(-1, self.patch_size, self.stride)
        # x shape: (batch, channels, num_patches, patch_size)
        
        # 投影到 embedding 空间
        x = self.projection(x)
        # x shape: (batch, channels, num_patches, embed_dim)
        
        # 应用 LayerNorm 和 Dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x

class ST_MEM_ViT(nn.Module):
    def __init__(self,
                 seq_len: int,
                 patch_size: int,
                 num_leads: int,
                 num_classes: Optional[int] = None,
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        # 移除了必须整除的断言，因为我们现在可以处理任意长度
        # assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'num_classes': num_classes if num_classes is not None else 'None',
                           'width': width,
                           'depth': depth,
                           'mlp_dim': mlp_dim,
                           'heads': heads,
                           'dim_head': dim_head,
                           'qkv_bias': qkv_bias,
                           'drop_out_rate': drop_out_rate,
                           'attn_drop_out_rate': attn_drop_out_rate,
                           'drop_path_rate': drop_path_rate}
        self.width = width
        self.depth = depth

        # embedding layers
        # 修改为使用新的 PatchEmbeddingLayer
        self.to_patch_embedding = PatchEmbeddingLayer(patch_size, width, stride=patch_size)

        # 动态计算 num_patches
        # 由于我们现在支持任意长度，这里使用公式计算最大可能的 patches 数量
        num_patches = (seq_len + patch_size - 1) // patch_size  # 向上取整
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(width))
                                                for _ in range(num_leads))

        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(input_dim=width,
                                     output_dim=width,
                                     hidden_dim=mlp_dim,
                                     heads=heads,
                                     dim_head=dim_head,
                                     qkv_bias=qkv_bias,
                                     drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=drop_path_rate_list[i])
            self.add_module(f'block{i}', block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

        # classifier head
        self.head = nn.Identity() if num_classes is None else nn.Linear(width, num_classes)

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def forward_encoding(self, series):
        num_leads = series.shape[1]
        if num_leads > len(self.lead_embeddings):
            raise ValueError(f'Number of leads ({num_leads}) exceeds the number of lead embeddings')

        x = self.to_patch_embedding(series)
        b, _, n, _ = x.shape
        x = x + self.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        # lead indicating modules
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        lead_embeddings = torch.stack([lead_embedding for lead_embedding in self.lead_embeddings]).unsqueeze(0)
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n + 2, -1)
        x = x + lead_embeddings
        x = rearrange(x, 'b c n p -> b (c n) p')

        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        # remove SEP embeddings
        x = rearrange(x, 'b (c n) p -> b c n p', c=num_leads)
        x = x[:, :, 1:-1, :]

        x = torch.mean(x, dim=(1, 2))
        return self.norm(x)

    def forward(self, series):
        x = self.forward_encoding(series)
        return x

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


def st_mem_vit_small(num_leads, num_classes=None, seq_len=2250, patch_size=75, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=384,
                      depth=12,
                      heads=6,
                      mlp_dim=1536,
                      **kwargs)
    return ST_MEM_ViT(**model_args)


def st_mem_vit_base(num_leads, num_classes=None, seq_len=2250, patch_size=75, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=768,
                      depth=12,
                      heads=12,
                      mlp_dim=3072,
                      **kwargs)
    return ST_MEM_ViT(**model_args)
