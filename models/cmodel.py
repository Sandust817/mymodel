__all__ = ['PatchTST']

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from layers.pos_encoding import positional_encoding  # 确保该函数存在
from layers.basics import Transpose, get_activation_fn, SigmoidRange  # 确保基础层存在
from layers.attention import MultiheadAttention  # 确保注意力层存在
# from model.plus import MonaLayer
import math
from einops import rearrange, repeat, einsum

from layers.Embed import DataEmbedding
class ClassifierHead(nn.Module):
    pass

class MlpHeadV1(ClassifierHead):
    name = "mlp_v1"
    def __init__(self, pretrain_out_dim, class_n,Identity=False):
        super().__init__()
        if Identity is True:
            self.classifier = nn.Sequential(
                # nn.Linear(pretrain_out_dim, pretrain_out_dim),
                # nn.BatchNorm1d(pretrain_out_dim),  # 对应 [batch_size, pretrain_out_dim]
                # nn.ReLU(),
                # nn.Dropout(0.8),
                    

                # nn.Linear(pretrain_out_dim, pretrain_out_dim // 2),
                # nn.BatchNorm1d(pretrain_out_dim // 2),  # 对应 [batch_size, pretrain_out_dim//2]
                # nn.ReLU(),
                # nn.Dropout(0.3),

                # nn.Linear(pretrain_out_dim//2, class_n)
                nn.Identity(),
                nn.Linear(pretrain_out_dim, class_n)
            )
        else :
            self.classifier = nn.Sequential(
                nn.Linear(pretrain_out_dim, pretrain_out_dim),
                nn.BatchNorm1d(pretrain_out_dim),  # 对应 [batch_size, pretrain_out_dim]
                nn.ReLU(),
                nn.Dropout(0.8),
                    

                nn.Linear(pretrain_out_dim, pretrain_out_dim // 2),
                nn.BatchNorm1d(pretrain_out_dim // 2),  # 对应 [batch_size, pretrain_out_dim//2]
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(pretrain_out_dim//2, class_n)
            )
            
        self.apply(self._init_weights)

        self.fc1 = nn.Linear(pretrain_out_dim, pretrain_out_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = nn.Linear(pretrain_out_dim, class_n)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        assert len(x.shape) == 2, f"Expected input shape [batch_size, features], got {x.shape}"
        return self.classifier(x)

class ScaleNorm(nn.Module):
    """
    ScaleNorm归一化（数据层面）：对输入序列按通道维度计算均值和标准差，支持逆归一化
    公式：normalized = (x - mean) / (std + eps)
          denormalized = normalized * std + mean
    """
    def __init__(self, dim=-1, eps=1e-5):
        super().__init__()
        self.dim = dim  # 归一化维度（默认通道维度）
        self.eps = eps  # 避免分母为0的微小值

    def forward(self, x: Tensor, observed_mask: Optional[Tensor] = None):
        """
        归一化前向传播
        Args:
            x: 输入数据，形状 [bs x seq_len x n_vars]（PatchTST标准输入格式）
            observed_mask: 观测掩码（1=有效数据，0=缺失数据），默认全1（无缺失）
        Returns:
            normalized_x: 归一化后的数据，形状同x
            mean: 各通道均值，形状 [bs x 1 x n_vars]
            std: 各通道标准差，形状 [bs x 1 x n_vars]
        """
        # 处理观测掩码（默认全有效）
        if observed_mask is None:
            observed_mask = torch.ones_like(x)
        
        # 按通道计算均值（仅用有效数据）
        sum_x = (x * observed_mask).sum(dim=self.dim-1, keepdim=True)  # [bs x 1 x n_vars]
        count_x = observed_mask.sum(dim=self.dim-1, keepdim=True).clamp_min(1.0)  # 避免除以0
        mean = sum_x / count_x
        
        # 按通道计算标准差（仅用有效数据）
        var = (((x - mean) * observed_mask) ** 2).sum(dim=self.dim-1, keepdim=True) / count_x
        std = torch.sqrt(var + self.eps)  # 添加eps避免std=0
        
        # 归一化
        normalized_x = (x - mean) / std
        return normalized_x, mean, std

    def inverse(self, normalized_x: Tensor, mean: Tensor, std: Tensor):
        """
        逆归一化（恢复原始数据尺度）
        Args:
            normalized_x: 归一化后的数据，形状 [bs x seq_len x n_vars] 或 Patch维度
            mean: 归一化时的均值，形状 [bs x 1 x n_vars]
            std: 归一化时的标准差，形状 [bs x 1 x n_vars]
        Returns:
            denormalized_x: 逆归一化后的数据，形状同normalized_x
        """
        # 自动匹配维度（支持序列维度或Patch维度）
        if normalized_x.dim() == 4:  # Patch维度：[bs x n_vars x num_patch x patch_len]
            # 调整mean/std维度以匹配Patch数据：[bs x n_vars x 1 x 1]
            mean = mean.transpose(1, 2).unsqueeze(-1)
            std = std.transpose(1, 2).unsqueeze(-1)
        elif normalized_x.dim() == 3:  # 序列维度：[bs x seq_len x n_vars]
            # 保持mean/std维度：[bs x 1 x n_vars]
            pass
        else:
            raise ValueError(f"不支持的输入维度：{normalized_x.dim()}，仅支持3维（序列）或4维（Patch）")
        
        denormalized_x = normalized_x * std + mean
        return denormalized_x

def masked_mse_loss(preds, targets, mask):
    """
    preds:   [bs x n_vars x num_patch x patch_len]（不含CLS）
    targets: [bs x n_vars x num_patch x patch_len]（不含CLS）
    mask:    [bs x n_vars x num_patch]（不含CLS，1表示被掩码，0表示保留）
    """
    loss = (preds - targets) ** 2
    loss = loss.mean(dim=-1)  # 在patch_len维度求平均，得到[bs x n_vars x num_patch]
    loss = (loss * mask).sum() / (mask.sum() + 1e-10)  # 仅计算被掩码位置的损失
    return loss

    
class PatchTSTStdScaler(nn.Module):
    def __init__(self, dim=1, keepdim=True, minimum_scale=1e-5):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor):
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim).clamp_min(1.0)
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator
        
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars] 
    输出: [bs x n_vars x num_patch x patch_len]（不含CLS）, num_patch（不含CLS）
    """
    seq_len = xb.shape[1]
    # 计算有效patch数量（避免超出序列长度）
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len  # 从序列末尾截取有效长度，确保patch不越界
    
    xb = xb[:, s_begin:, :]  # [bs x tgt_len x n_vars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)  # [bs x num_patch x n_vars x patch_len]
    xb = xb.transpose(1, 2).contiguous()  # 调整维度为[bs x n_vars x num_patch x patch_len]
    return xb, num_patch


def random_masking(xb, mask_ratio):
    """
    自监督预训练的随机掩码：对不含CLS的patch进行掩码
    xb: [bs x n_vars x num_patch x patch_len]（不含CLS）
    输出: 掩码后patch、保留的patch、掩码矩阵、恢复索引
    """
    bs, n_vars, num_patch, patch_len = xb.shape
    x = xb.clone()
    
    # 计算需要保留的patch数量
    len_keep = int(num_patch * (1 - mask_ratio))
    
    # 生成随机噪声用于排序（决定哪些patch被保留）
    noise = torch.rand(bs, n_vars, num_patch, device=xb.device)  # [bs x n_vars x num_patch]
    ids_shuffle = torch.argsort(noise, dim=2)  # 升序排序：小值保留，大值掩码
    ids_restore = torch.argsort(ids_shuffle, dim=2)  # 用于恢复原始顺序
    
    # 保留前len_keep个patch
    ids_keep = ids_shuffle[:, :, :len_keep]  # [bs x n_vars x len_keep]
    x_kept = torch.gather(
        x, dim=2, 
        index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, patch_len)  # 扩展到patch_len维度
    )  # [bs x n_vars x len_keep x patch_len]
    
    # 被掩码的patch用0填充
    x_removed = torch.zeros(bs, n_vars, num_patch - len_keep, patch_len, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=2)  # [bs x n_vars x num_patch x patch_len]
    
    # 恢复原始patch顺序（确保掩码位置与输入对应）
    x_masked = torch.gather(
        x_, dim=2, 
        index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, patch_len)
    )  # [bs x n_vars x num_patch x patch_len]（不含CLS）
    
    # 生成掩码矩阵：1表示被掩码，0表示保留
    mask = torch.ones([bs, n_vars, num_patch], device=xb.device)
    mask[:, :, :len_keep] = 0
    mask = torch.gather(mask, dim=2, index=ids_restore)  # [bs x n_vars x num_patch]（不含CLS）
    
    return x_masked, x_kept, mask, ids_restore


class PretrainHead(nn.Module):
    """自监督预训练头：将编码器输出映射回patch_len维度，移除CLS token"""
    def __init__(self, d_model, patch_len, dropout, use_cls_token=False):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(d_model, patch_len)  # 从d_model映射到patch_len
        self.use_cls_token = use_cls_token

    def forward(self, x):
        """
        x: [bs x n_vars x (num_patch+1) x d_model]（含CLS）或 [bs x n_vars x num_patch x d_model]（不含）
        输出: [bs x n_vars x num_patch x patch_len]（不含CLS，与原始patch维度匹配）
        """
        x = self.linear(self.dropout(x))  # 映射到patch_len维度
        if self.use_cls_token:
            x = x[:, :, 1:, :]  # 移除CLS token（第0个位置）
        return x

class TSTEncoderLayer(nn.Module):
    """Transformer编码器单层：包含多头注意力和前馈网络"""
    def __init__(self, config):

        super().__init__()
        d_model = config.d_model
        n_heads = config.n_heads
        d_ff = getattr(config, 'd_ff', 256)
        norm = getattr(config, 'norm', 'BatchNorm')
        attn_dropout = getattr(config, 'attn_dropout', 0.)
        dropout = getattr(config, 'dropout', 0.)
        activation = getattr(config, 'act', 'gelu')
        res_attention = getattr(config, 'res_attention', False)
        pre_norm = getattr(config, 'pre_norm', False)
        store_attn = getattr(config, 'store_attn', False)
        bias = getattr(config, 'bias', True)
        
        assert d_model % n_heads == 0, f"d_model({d_model})必须被n_heads({n_heads})整除"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # 多头自注意力
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(
            d_model, n_heads, d_k, d_v, 
            attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention
        )
        self.dropout_attn = nn.Dropout(dropout)
        # self.mona = MonaLayer(channels=d_model, down_stride=2)
        # 归一化层（BatchNorm/LayerNorm）
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # 前馈网络
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        
        # 前馈网络归一化
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm  # 预归一化/后归一化
        self.store_attn = store_attn  # 是否保存注意力权重

    def forward(self, src: Tensor, prev: Optional[Tensor] = None):
        """
        src: [bs x seq_len x d_model]（seq_len含CLS时为num_patch+1）
        prev: 残差注意力的历史分数（可选）
        """
        # 注意力子层
        if self.pre_norm:
            src = self.norm_attn(src)  # 预归一化
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn  # 保存注意力权重
        # 残差连接 +  dropout
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)  # 后归一化
        # src2 = self.mona(src)
        # 前馈网络子层
        if self.pre_norm:
            src = self.norm_ffn(src)  # 预归一化
        src2 = self.ff(src)
        # 残差连接 + dropout
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)  # 后归一化

        if self.res_attention:
            return src, scores
        else:
            return src


class TSTEncoder(nn.Module):
    """Transformer编码器：堆叠多个TSTEncoderLayer"""
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        n_heads = config.n_heads
        d_ff = getattr(config, 'd_ff', None)
        norm = getattr(config, 'norm', 'BatchNorm')
        attn_dropout = getattr(config, 'attn_dropout', 0.)
        dropout = getattr(config, 'dropout', 0.)
        activation = getattr(config, 'act', 'gelu')
        res_attention = getattr(config, 'res_attention', False)
        e_layers = getattr(config, 'e_layers', 8)
        pre_norm = getattr(config, 'pre_norm', False)
        store_attn = getattr(config, 'store_attn', False)
        
        self.res_attention = res_attention
        # self.layers = nn.ModuleList([
        #     TSTEncoderLayer(
        #         type('Config', (), {
        #             'd_model': d_model,
        #             'n_heads': n_heads,
        #             'd_ff': d_ff,
        #             'norm': norm,
        #             'attn_dropout': attn_dropout,
        #             'dropout': dropout,
        #             'act': activation,
        #             'res_attention': res_attention,
        #             'pre_norm': pre_norm,
        #             'store_attn': store_attn,
        #             'bias': True
        #         })
        #     ) for _ in range(e_layers)
        # ])
        self.d_inner = config.d_model * 2
        self.dt_rank = math.ceil(config.d_model / 16)
        self.layers = nn.ModuleList([ResidualBlock(config, self.d_inner, self.dt_rank) for _ in range(config.e_layers)])

    def forward(self, src: Tensor):
        """
        src: [bs x seq_len x d_model]（seq_len含CLS时为num_patch+1）
        输出: [bs x seq_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, prev=scores)
        else:
            for mod in self.layers:
                output = mod(output)
        return output


class PatchTSTEncoder(nn.Module):
    """PatchTST编码器：包含patch嵌入、CLS token添加、位置编码、Transformer编码器"""
    def __init__(self, config):
        super().__init__()
        seq_len = config.seq_len
        num_class = config.num_class  # 输入变量数
        patch_len = 16 # config.patch_len  每个patch的长度
        stride = 8 # config.stride  patch滑动步长
        num_patch = (max(config.seq_len, 16) - 16) // 8 + 1 # 不含CLS的原始patch数量
        # stride = config.stride
        e_layers = getattr(config, 'e_layers', 3)
        d_model = config.d_model
        n_heads = getattr(config, 'n_heads', 16)
        shared_embedding = getattr(config, 'shared_embedding', True)
        d_ff = getattr(config, 'd_ff', 256)
        norm = getattr(config, 'norm', 'BatchNorm')
        attn_dropout = getattr(config, 'attn_dropout', 0.)
        dropout = getattr(config, 'dropout', 0.)
        act = getattr(config, 'act', 'gelu')
        res_attention = getattr(config, 'res_attention', False)
        pre_norm = getattr(config, 'pre_norm', False)
        store_attn = getattr(config, 'store_attn', False)
        pe = getattr(config, 'pe', 'sincos')
        learn_pe = getattr(config, 'learn_pe', True)
        use_cls_token = config.use_cls_token
        verbose = getattr(config, 'verbose', False)
        
        self.n_vars = num_class  # 变量数量
        self.num_patch = num_patch  # 含CLS的patch总数（num_patch = 原始num_patch + 1）
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.shared_embedding = shared_embedding  # 是否所有变量共享嵌入层
        self.use_cls_token = use_cls_token

        # 1. Patch嵌入层：将patch_len维度映射到d_model
        if not self.shared_embedding:
            # 每个变量单独的嵌入层
            self.W_P = nn.ModuleList([nn.Linear(patch_len, d_model) for _ in range(self.n_vars)])
        else:
            # 所有变量共享嵌入层
            self.W_P = nn.Linear(patch_len, d_model)

        # 2. CLS token初始化（可学习，维度与嵌入后一致）
        if self.use_cls_token:
            # 形状：[1, 1, 1, d_model]，方便后续扩展到[bs, n_vars, 1, d_model]
            self.cls_token = nn.Parameter(torch.randn(1, 1, 1, d_model))
            self.num_patch = num_patch + 1
            nn.init.trunc_normal_(self.cls_token, std=0.02)  # 符合Transformer初始化规范

        # 3. 位置编码（长度为含CLS的num_patch）
        self.W_pos = positional_encoding(pe, learn_pe, self.num_patch, d_model)
        self.dropout = nn.Dropout(dropout)  # 位置编码后的dropout
        # self.mona=MonaLayer(d_model)
        self.gelu=nn.GELU()



        self.encoder = TSTEncoder(config)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [bs x n_vars x num_patch_origin x patch_len]（num_patch_origin：不含CLS的原始patch数量）
        输出: [bs x n_vars x (num_patch_origin+1) x d_model]（含CLS）或 [bs x n_vars x num_patch_origin x d_model]（不含）
        """
        bs, n_vars, num_patch_origin, patch_len = x.shape

        # Step 1: Patch嵌入到d_model维度
        if not self.shared_embedding:
            x_embed_list = []
            for i in range(n_vars):
                # 每个变量单独嵌入：[bs x num_patch_origin x d_model]
                z = self.W_P[i](x[:, i, :, :])
                x_embed_list.append(z)
            x_embed = torch.stack(x_embed_list, dim=1)  # [bs x n_vars x num_patch_origin x d_model]
        else:
            # 共享嵌入：调整维度后批量处理
            x = x.permute(0, 2, 1, 3)  # [bs x num_patch_origin x n_vars x patch_len]
            x_embed = self.W_P(x)  # [bs x num_patch_origin x n_vars x d_model]
            x_embed = x_embed.permute(0, 2, 1, 3)  # [bs x n_vars x num_patch_origin x d_model]

        # Step 2: 添加CLS token（仅当use_cls_token=True）
        if self.use_cls_token:
            # 扩展CLS token到批次和变量维度：[bs x n_vars x 1 x d_model]
            cls_tokens = self.cls_token.expand(bs, n_vars, -1, -1)
            # 在num_patch维度（dim=2）的开头添加CLS
            x_embed = torch.cat([cls_tokens, x_embed], dim=2)  # [bs x n_vars x (num_patch_origin+1) x d_model]
            current_num_patch = num_patch_origin + 1
        else:
            current_num_patch = num_patch_origin

        # Step 3: 位置编码（调整维度为Transformer输入格式）
        # 合并bs和n_vars维度：[bs*n_vars x current_num_patch x d_model]
        u = x_embed.reshape(bs * n_vars, current_num_patch, self.d_model)
        # 获取匹配长度的位置编码（避免超出预定义长度）
        pos_embedding = self.W_pos[:current_num_patch, :]  # [current_num_patch x d_model]
        # 添加位置编码并dropout
        u = self.dropout(u + pos_embedding)

        # Step 4: Transformer编码器前向传播
        z_encoder = self.encoder(u)  # [bs*n_vars x current_num_patch x d_model]
        # tmp=z_encoder
        # z_encoder = self.gelu(self.mona(z_encoder))+tmp

        # Step 5: 恢复原始维度（拆分bs和n_vars）
        z = z_encoder.reshape(bs, n_vars, current_num_patch, self.d_model)  # [bs x n_vars x current_num_patch x d_model]

        return z

class Model(nn.Module):

    """
    PatchTST主模型（含CLS token支持）：
    - 新增超参数use_scalenorm：控制是否启用ScaleNorm归一化
    - 新增inverse_scale方法：支持逆归一化恢复原始数据
    """
    def __init__(self, config):
        super().__init__()
        # 自监督预训练参数
        self.task_name = config.task_name
        # self.mask_ratio = config.mask_rate
        config.use_cls_token = True
        config.head_dropout= 0.0
        config.shared_embedding=True
        config.verbose=False,
        self.use_cls_token = True
        
        # --------------------------- 新增：归一化控制超参数与模块 ---------------------------
        self.use_scalenorm = getattr(config, 'use_scalenorm', True)  # 超参数：是否启用ScaleNorm
        if self.use_scalenorm:
            self.scaler = ScaleNorm(dim=-1, eps=1e-5)  # 初始化ScaleNorm（按通道归一化）
        else:
            self.scaler = None  # 不启用归一化时，scaler设为None

        # 基础结构参数
        self.seq_len = config.seq_len
        self.n_vars = config.enc_in # 输入变量数
        self.patch_len = 16 # config.patch_len  每个patch的长度
        self.stride = 8 # config.stride  patch滑动步长
        self.num_patch_origin = (max(config.seq_len, 16) - 16) // 8 + 1 # 不含CLS的原始patch数量
        config.num_patch = self.num_patch_origin
        # 计算含CLS的patch总数（传给编码器）
        self.num_patch_with_cls = self.num_patch_origin + 1 if config.use_cls_token else self.num_patch_origin

        # 1. 编码器 backbone（含CLS token处理）

        self.backbone = PatchTSTEncoder(config)
        
        # 2. 自监督预训练头
        self.pretrain_head = PretrainHead(
            patch_len=self.patch_len,
            d_model=config.d_model,
            dropout=config.head_dropout,
            use_cls_token=config.use_cls_token
        )
        self.classification_head=MlpHeadV1(pretrain_out_dim=self.n_vars*config.d_model,class_n=config.num_class)
        # 3. 特征展平（用于下游任务特征提取）
        self.flatten = nn.Flatten(start_dim=1)

    # --------------------------- 核心修改3：前向传播集成归一化 ---------------------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
    def pretrain(self, z: Tensor,z_mark):
        """
        自监督预训练模式：输入序列 → （可选归一化）→ 掩码重建 → 损失计算
        z: [bs x n_vars x seq_len]（原始输入，变量在前，序列在后）
        返回: 含损失、重建结果、逆归一化所需参数的字典
        """
        # Step 1: 维度调整（适配ScaleNorm和Patch创建：[bs x seq_len x n_vars]）
        z_seq = z.transpose(1, 2)  # [bs x seq_len x n_vars]
        observed_mask = torch.ones_like(z_seq)  # 假设所有数据有效（实际场景可传入真实掩码）
        self.mean = None  # 保存均值（用于逆归一化）
        self.std = None   # 保存标准差（用于逆归一化）

        # Step 2: （可选）ScaleNorm归一化
        if self.use_scalenorm:
            scaled_z, self.mean, self.std = self.scaler(z_seq, observed_mask)
        else:
            scaled_z = z_seq  # 不归一化，直接使用原始数据

        # Step 3: 创建Patch（不含CLS）
        z_patch, _ = create_patch(scaled_z, self.patch_len, self.stride)  # [bs x n_vars x num_patch_origin x patch_len]
        original_patch = z_patch.clone()  # 保存原始Patch（归一化后）用于计算损失
        # z_patch=scaled_z
        # Step 4: 随机掩码（仅预训练阶段）
        if self.mask_ratio > 0:
            z_patch_masked, _, mask, _ = random_masking(z_patch, self.mask_ratio)
            self.mask = mask.bool()  # [bs x n_vars x num_patch_origin]（1=掩码，0=保留）
        else:
            z_patch_masked = z_patch
            self.mask = torch.zeros_like(z_patch[:, :, :, 0]).bool()  # 全0掩码（无掩码）
        
        # z_current = z_patch_masked
        # for block in self.backbone:
        #     z_current = block(z_current)  # 每个块接收上一个块的输出作为输入

        # # 最终输出即为经过所有残差块处理后的特征
        # z_backbone = z_current  # [bs x n_vars x num_patch_with_cls x d_model]
        z_backbone = self.backbone(z_patch_masked)  # [bs x n_vars x num_patch_with_cls x d_model]
        # Step 6: 重建被掩码的Patch
        reconstructed_patch = self.pretrain_head(z_backbone)  # [bs x n_vars x num_patch_origin x patch_len]

        # Step 7: 计算掩码MSE损失（基于归一化后的数据，避免尺度影响）
        loss = masked_mse_loss(reconstructed_patch, original_patch, self.mask)

        # Step 8: （可选）对重建结果进行逆归一化（恢复原始尺度，便于后续分析）

        # 返回结果（含逆归一化后的数据，便于下游分析）
        return {
            'loss': loss,
            'prediction_output': reconstructed_patch,  # 归一化后的重建Patch
            'hidden_states': z_backbone,  # 编码器输出（含CLS）
            'attentions': self.backbone.encoder.layers[0].attn if hasattr(self.backbone.encoder.layers[0], 'attn') else None,
            'mask': self.mask,  # 掩码矩阵
            'mean': self.mean,  # 归一化均值（用于逆归一化）
            'std': self.std,    # 归一化标准差（用于逆归一化）
            'patch_input': original_patch  # 归一化后的原始Patc
        }

    # --------------------------- 核心修改4：特征提取模式集成归一化 ---------------------------
    def classification(self, z: Tensor,z_mark):
        """
        特征提取模式：用于下游任务（分类/回归），输出CLS token特征或Patch平均特征
        z: [bs x n_vars x seq_len]
        输出: [bs x n_vars*d_model]（展平后的特征）
        """
        # Step 1: 维度调整 + （可选）归一化
        # z_seq = z.transpose(1, 2)  # [bs x seq_len x n_vars]
        z_seq = z
        observed_mask = torch.ones_like(z_seq)
        if self.use_scalenorm:
            scaled_z, _, _ = self.scaler(z_seq, observed_mask)
        else:
            scaled_z = z_seq

        # Step 2: 创建Patch
        z_patch, _ = create_patch(scaled_z, self.patch_len, self.stride)  # [bs x n_vars x num_patch_origin x patch_len]

        # Step 3: 编码器前向传播（含CLS token）
        z_backbone = self.backbone(z_patch)  # [bs x n_vars x num_patch_with_cls x d_model]

        # Step 4: 特征提取（CLS token或平均池化）
        if self.use_cls_token:
            feature = z_backbone[:, :, 0, :]  # [bs x n_vars x d_model]（取CLS token）
        else:
            feature = z_backbone.mean(dim=2)  # [bs x n_vars x d_model]（Patch平均）

        # Step 5: 特征展平（适配下游任务输入）
        feature_flatten = self.flatten(feature)  # [bs x n_vars*d_model]
        x=self.classification_head(feature_flatten)

        return x

    # --------------------------- 新增：独立逆归一化方法（外部调用） ---------------------------
    def inverse_scale(self, normalized_x: Tensor, mean: Tensor, std: Tensor):
        """
        外部调用的逆归一化接口（如对测试集结果逆归一化）
        Args:
            normalized_x: 归一化后的数据（序列/Patch维度）
            mean: 归一化时的均值（来自forward的返回值）
            std: 归一化时的标准差（来自forward的返回值）
        Returns:
            denormalized_x: 逆归一化后的数据
        """
        if not self.use_scalenorm:
            # warnings.warn("当前模型未启用ScaleNorm，逆归一化操作无效，直接返回输入数据")
            return normalized_x
        return self.scaler.inverse(normalized_x, mean, std)

class ResidualBlock(nn.Module):
    def __init__(self, config, d_inner, dt_rank):
        super(ResidualBlock, self).__init__()
        
        self.mixer = MambaBlock(config, d_inner, dt_rank)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

class MambaBlock(nn.Module):
    def __init__(self, config, d_inner, dt_rank):
        super(MambaBlock, self).__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels = self.d_inner,
            out_channels = self.d_inner,
            bias = True,
            kernel_size = config.d_conv,
            padding = config.d_conv - 1,
            groups = self.d_inner
        )

        # takes in x and outputs the input-specific delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + config.d_ff * 2, bias=False)

        # projects delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, config.d_ff + 1), "n -> d n", d=self.d_inner).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)

    def forward(self, x):
        """
        Figure 3 in Section 3.4 in the paper
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x) # [B, L, 2 * d_inner]
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output


    def ssm(self, x):
        """
        Algorithm 2 in Section 3.2 in the paper
        """
        
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float()) # [d_in, n]
        D = self.D.float() # [d_in]

        x_dbl = self.x_proj(x) # [B, L, d_rank + 2 * d_ff]
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1) # delta: [B, L, d_rank]; B, C: [B, L, n]
        delta = F.softplus(self.dt_proj(delta)) # [B, L, d_in]
        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n")) # A is discretized using zero-order hold (ZOH) discretization
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n") # B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors: "A is the more important term and the performance doesn't change much with the simplification on B"

        # selective scan, sequential instead of parallel
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1) # [B, L, d_in]
        y = y + u * D

        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


import types  # 导入types模块

def Cmodel_prefer_custom(args):
    # 构建统一的配置字典（内容不变）
    config_dict = {
        'target_dim': args.signal_length,
        'patch_len': args.patch_length,
        'stride': args.patch_stride,
        'num_patch': (max(args.signal_length, args.patch_length) - args.patch_length) // args.patch_stride + 1,
        'mask_ratio': args.mask_ratio,
        'use_cls_token': getattr(args, 'use_cls_token', True),
        'd_model': args.embed_dim,
        'head_dropout': getattr(args, 'dropout_rate', 0.0),
        'shared_embedding': True,
        'verbose': False,
        'e_layers': getattr(args, 'encoder_depth', 3),
        'n_heads': getattr(args, 'encoder_num_heads', 4),
        'd_ff': getattr(args, 'embed_dim', 128) * getattr(args, 'mlp_ratio', 2),
        'd_conv': getattr(args, 'd_conv', 4),
        # 'norm': getattr(args, 'all_encode_norm_layer', 'BatchNorm'),
        # 'attn_dropout': getattr(args, 'dropout_rate', 0.0),
        # 'dropout': getattr(args, 'dropout_rate', 0.0),
        # 'act': 'gelu',
        # 'res_attention': False,
        # 'pre_norm': True,
        # 'store_attn': False,
        # 'pe': getattr(args, 'positional_encoding_type', 'zeros'),
        # 'learn_pe': getattr(args, 'use_positional_encoding', True),

    }
    
    if hasattr(args, 'mask_type'):
        config_dict['mask_type'] = args.mask_type
    
    # 关键：将字典转换为可以用.访问的命名空间对象
    config = types.SimpleNamespace(**config_dict)
    
    # 保持原有的访问方式不变
    model = PatchTST(config)
    return model
    