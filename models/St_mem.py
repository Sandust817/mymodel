from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.st_mem_vit import ST_MEM_ViT, TransformerBlock


__all__ = ['ST_MEM', 'st_mem_vit_small_dec256d4b', 'st_mem_vit_base_dec256d4b']

class MlpHead(nn.Module):
    name = "mlp"
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

# 1. 定义支持去归一化的ScaleNorm类
class ScaleNorm(nn.Module):
    """ScaleNorm层实现，新增inverse方法用于去归一化"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim ** 0.5))
        self.last_norm = None  # 保存归一化时的norm值，用于去归一化
    
    def forward(self, x):
        self.last_norm = torch.norm(x, dim=-1, keepdim=True)  # 保存当前输入的norm
        return x / self.last_norm.clamp(min=self.eps) * self.g
    
    def inverse(self, x):
        """去归一化：恢复到原始数据尺度"""
        assert self.last_norm is not None, "需先调用forward进行归一化，再调用inverse"
        return x * self.last_norm.clamp(min=self.eps) / self.g


def get_1d_sincos_pos_embed(embed_dim: int,
                            grid_size: int,
                            temperature: float = 10000,
                            sep_embed: bool = False):
    """Positional embedding for 1D patches."""
    assert (embed_dim % 2) == 0, \
        'feature dimension must be multiple of 2 for sincos emb.'
    grid = torch.arange(grid_size, dtype=torch.float32)

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= (embed_dim / 2.)
    omega = 1. / (temperature ** omega)

    grid = grid.flatten()[:, None] * omega[None, :]
    pos_embed = torch.cat((grid.sin(), grid.cos()), dim=1)
    if sep_embed:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed, torch.zeros(1, embed_dim)],
                              dim=0)
    return pos_embed


# 修改 Model 类的 __init__ 方法
class Model(nn.Module):
    name = 'ST_MEM'

    def __init__(self, configs):
        # 从 configs (args) 对象中提取参数
        seq_len = configs.seq_len
        patch_size = configs.patch_len if hasattr(configs, 'patch_len') else 16  # 根据实际配置调整默认值
        num_leads = configs.enc_in
        embed_dim = configs.d_model
        depth = configs.e_layers
        num_heads = configs.n_heads
        decoder_embed_dim = 128  # 可以添加到 configs 中
        decoder_depth = 3        # 可以添加到 configs 中
        decoder_num_heads = 3    # 可以添加到 configs 中
        mlp_ratio = 2
        qkv_bias = True
        norm_layer = nn.LayerNorm
        num_classes = configs.num_class
        norm_pix_loss = False
        
        super().__init__()
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'embed_dim': embed_dim,
                           'depth': depth,
                           'num_heads': num_heads,
                           'decoder_embed_dim': decoder_embed_dim,
                           'decoder_depth': decoder_depth,
                           'decoder_num_heads': decoder_num_heads,
                           'mlp_ratio': mlp_ratio,
                           'qkv_bias': qkv_bias,
                           'norm_layer': str(norm_layer),
                           'norm_pix_loss': norm_pix_loss}
        self.task_name = configs.task_name
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.num_leads = num_leads
        self.num_classes = num_classes
        
        # 2. 新增：在__init__中初始化ScaleNorm（仅用于输入输出，不侵入encoder/decoder）
        self.input_output_norm = ScaleNorm(dim=seq_len)  # 输入输出共用一个归一化层，保证尺度一致
        
        # --------------------------------------------------------------------
        # MAE encoder specifics - 完全未修改
        self.encoder = ST_MEM_ViT(seq_len=seq_len,
                                  patch_size=patch_size,
                                  num_leads=num_leads,
                                  width=embed_dim,
                                  depth=depth,
                                  mlp_dim=mlp_ratio * embed_dim,
                                  heads=num_heads,
                                  qkv_bias=qkv_bias)
        self.to_patch_embedding = self.encoder.to_patch_embedding
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # MAE decoder specifics - 完全未修改
        self.to_decoder_embedding = nn.Linear(embed_dim, decoder_embed_dim)

        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 2, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([TransformerBlock(input_dim=decoder_embed_dim,
                                                              output_dim=decoder_embed_dim,
                                                              hidden_dim=decoder_embed_dim * mlp_ratio,
                                                              heads=decoder_num_heads,
                                                              dim_head=64,
                                                              qkv_bias=qkv_bias)
                                             for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, patch_size)
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()
        self.classification_head = MlpHead(embed_dim, self.num_classes)

    def initialize_weights(self):
        
        pos_embed = get_1d_sincos_pos_embed(self.encoder.pos_embedding.shape[-1],
                                            self.num_patches,
                                            sep_embed=True)
        if self.encoder.pos_embedding.shape[1] == pos_embed.shape[0]:
            self.encoder.pos_embedding.data.copy_(pos_embed.float().unsqueeze(0))
        else:
            # 如果维度不匹配，进行插值调整
            pos_embed_resized = F.interpolate(
                pos_embed.unsqueeze(0).unsqueeze(0), 
                size=(self.encoder.pos_embedding.shape[1], self.encoder.pos_embedding.shape[-1]), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0).squeeze(0)
            self.encoder.pos_embedding.data.copy_(pos_embed_resized.float().unsqueeze(0))
        self.encoder.pos_embedding.requires_grad = False

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    self.num_patches,
                                                    sep_embed=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))

        torch.nn.init.normal_(self.encoder.sep_embedding, std=.02)
        torch.nn.init.normal_(self.mask_embedding, std=.02)
        for i in range(self.num_leads):
            torch.nn.init.normal_(self.encoder.lead_embeddings[i], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, series):
        
        p = self.patch_size
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b c (n p) -> b c n p', p=p)
        return x

    def unpatchify(self, x):
        
        series = rearrange(x, 'b c n p -> b c (n p)')
        return series

    def random_masking(self, x, mask_ratio):
        
        b, num_leads, n, d = x.shape
        len_keep = int(n * (1 - mask_ratio))

        noise = torch.rand(b, num_leads, n, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=2)
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        ids_keep = ids_shuffle[:, :, :len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, d))

        mask = torch.ones([b, num_leads, n], device=x.device)
        mask[:, :, :len_keep] = 0
        mask = torch.gather(mask, dim=2, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.to_patch_embedding(x)
        b, _, n, _ = x.shape

        x = x + self.encoder.pos_embedding[:, 1:n + 1, :].unsqueeze(1)

        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = torch.zeros([b, self.num_leads, n], device=x.device)
            ids_restore = torch.arange(n, device=x.device).unsqueeze(0).repeat(b, self.num_leads, 1)

        sep_embedding = self.encoder.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, self.num_leads, -1, -1) + self.encoder.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        #?
        lead_embeddings = torch.stack([self.encoder.lead_embeddings[i] for i in range(self.num_leads)]).unsqueeze(0)
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, x.shape[2], -1)
        x = x + lead_embeddings

        x = rearrange(x, 'b c n p -> b (c n) p')
        for i in range(self.encoder.depth):
            x = getattr(self.encoder, f'block{i}')(x)
        x = self.encoder.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.to_decoder_embedding(x)

        x = rearrange(x, 'b (c n) p -> b c n p', c=self.num_leads)
        b, _, n_masked_with_sep, d = x.shape
        n = ids_restore.shape[2]
        mask_embeddings = self.mask_embedding.unsqueeze(1)
        mask_embeddings = mask_embeddings.repeat(b, self.num_leads, n + 2 - n_masked_with_sep, 1)

        x_wo_sep = torch.cat([x[:, :, 1:-1, :], mask_embeddings], dim=2)
        x_wo_sep = torch.gather(x_wo_sep, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, d))

        x_wo_sep = x_wo_sep + self.decoder_pos_embed[:, 1:n + 1, :].unsqueeze(1)
        left_sep = x[:, :, :1, :] + self.decoder_pos_embed[:, :1, :].unsqueeze(1)
        right_sep = x[:, :, -1:, :] + self.decoder_pos_embed[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x_wo_sep, right_sep], dim=2)

        x_decoded = []
        for i in range(self.num_leads):
            x_lead = x[:, i, :, :]
            for block in self.decoder_blocks:
                x_lead = block(x_lead)
            x_lead = self.decoder_norm(x_lead)
            x_lead = self.decoder_head(x_lead)
            x_decoded.append(x_lead[:, 1:-1, :])
        x = torch.stack(x_decoded, dim=1)

        return x

    def forward_loss(self, series, pred, mask):
        # 5. 损失计算：用归一化后的输入计算目标，保证与模型内部处理的输入尺度一致
        series_normed = self.input_output_norm(series)  # 与输入到encoder的尺度相同
        target = self.patchify(series_normed)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    # 在 ST_MEM 类中添加以下方法
    def classification(self, x_enc, x_mark_enc=None):
        """
        用于分类任务的前向传播
        x_enc: [B, L, D] - 输入序列
        x_mark_enc: [B, L, D] - 时间标记（可选）
        """

        
        # 转换维度以适应模型输入 [B, D, L]
        x_enc = x_enc.permute(0, 2, 1)
        
        # 使用forward_feature提取特征
        feature = self.forward_feature(x_enc, use_gap=False)  # [B, D]
        
        # 添加分类头
        output = self.classification_head(feature)

        
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        统一的前向传播接口，根据task_name选择不同的任务
        x_enc: [B, L, D] - 编码器输入序列
        x_mark_enc: [B, L, D] - 编码器时间标记
        x_dec: [B, L, D] - 解码器输入序列（预测任务时使用）
        x_mark_dec: [B, L, D] - 解码器时间标记
        mask: [B, L] - 掩码
        """
        
        if self.task_name == 'reconstruction':
            # 重构任务保持原有逻辑
            result = super().forward(x_enc.permute(0, 2, 1), mask_ratio=0.75)
            return result
        
        elif self.task_name == 'classification':
            x_enc=self.input_output_norm(x_enc)
            # 分类任务使用新的分类方法
            return self.classification(x_enc, x_mark_enc)
        
        else:
            raise ValueError(f"Unsupported task name: {task_name}")
    
    def forward_feature(self, x, use_gap=False):
        # 特征提取时也保持输入归一化，与训练时一致
        x_normed = self.input_output_norm(x)
        latent, _, ids_restore = self.forward_encoder(x_normed, mask_ratio=0.0)

        if not use_gap:
            B, _, D = latent.shape
            cls_tokens = latent[:, :self.num_leads, :]
            feature = cls_tokens.mean(dim=1)
        else:
            feature = latent.mean(dim=1)
            
        return feature

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str


def st_mem_vit_small_dec256d4b(**kwargs):
    model = ST_MEM(embed_dim=384,
                   depth=12,
                   num_heads=6,
                   decoder_embed_dim=256,
                   decoder_depth=4,
                   decoder_num_heads=4,
                   mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6),** kwargs)
    return model


def st_mem_vit_base_dec256d4b(**kwargs):
    model = ST_MEM(embed_dim=768,
                   depth=12,
                   num_heads=12,
                   decoder_embed_dim=256,
                   decoder_depth=4,
                   decoder_num_heads=4,
                   mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6),** kwargs)
    return model

def stmem_prefer_custom(args):
    model = ST_MEM(
        seq_len = args.signal_length,num_leads=args.num_input_channels,patch_size=args.patch_length,
    )
    return model