from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import repeat, rearrange
import math

from layers.mona import MonaFeatureExtractor

class ContextPrototypes(nn.Module):
    def __init__(self, num_proto, dim):
        super().__init__()
        directions = F.normalize(torch.randn(num_proto, dim), dim=-1)
        # 初始化幅值（例如 0.5）
        magnitudes = torch.full((num_proto, 1), 1.0)
        prototypes = directions * magnitudes
        self.prototypes = nn.Parameter(prototypes)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMSNorm 公式: x * w / sqrt(mean(x^2) + eps)
        norm_x = torch.mean(x.pow(2), dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

class ScoreAggregation(nn.Module):
    def __init__(self, n_prototypes_total: int, num_classes: int, temperature: float = 0.05):
        super().__init__()
        assert n_prototypes_total % num_classes == 0
        self.k = n_prototypes_total // num_classes
        self.temperature = temperature

    def forward(self, prototype_logits: torch.Tensor) -> torch.Tensor:
        B, K = prototype_logits.shape
        logits = prototype_logits.view(B, -1, self.k)  # [B, C, k]
        return torch.logsumexp(logits / self.temperature, dim=-1)  # [B, C]
# ----------------------------
# 2. 增强型原型损失函数
# ----------------------------
class IntegratedPrototypeLoss(nn.Module):
    def __init__(self, n_prototypes_total: int, num_classes: int, alpha: float = 0.5, beta: float = 0.1, temperature: float = 0.05):
        super().__init__()
        self.k = n_prototypes_total // num_classes
        self.num_classes = num_classes
        self.alpha = alpha  # 子任务（原型分类）权重
        self.beta = beta    # 多样性权重
        self.ce = nn.CrossEntropyLoss()
        self.temperature = temperature

    def diversity_loss(self, prototypes: Tensor) -> Tensor:
        """计算类内原型多样性损失: 鼓励同类原型正交"""
        K, D = prototypes.shape
        # 归一化原型向量
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        # 变性为 [C, k, D]
        proto_reshaped = proto_norm.view(self.num_classes, self.k, D)
        # 计算每类内部的相似度矩阵 [C, k, k]
        sim_matrix = torch.bmm(proto_reshaped, proto_reshaped.transpose(1, 2))
        
        identity = torch.eye(self.k, device=prototypes.device).unsqueeze(0)
        # 惩罚非对角线（不同原型）之间的相似度
        return F.mse_loss(sim_matrix, identity.expand(self.num_classes, -1, -1))

    def forward(self, aggregated_logits, prototype_logits, prototypes, targets):
        # 1. 主任务：聚合后的类别预测损失 [B, C]
        main_loss = self.ce(aggregated_logits, targets)
        
        # 2. 子任务：原型级别的分类损失 [B, K] -> [B*k, C]
        B = targets.shape[0]
        # 将每个样本对应的 k 个原型都分配同样的标签
        expanded_targets = targets.repeat_interleave(self.k) 
        # 重新排列 prototype_logits 以匹配扩展后的标签
        # prototype_logits: [B, C*k] -> [B, C, k] -> [B, k, C] -> [B*k, C]
        sub_proto_logits = (prototype_logits / self.temperature).view(B, self.num_classes, self.k).transpose(1, 2).reshape(-1, self.num_classes)
        proto_ce_loss = self.ce(sub_proto_logits, expanded_targets)
        
        # 3. 多样性约束
        div_loss = self.diversity_loss(prototypes)
        
        return main_loss + self.alpha * proto_ce_loss + self.beta * div_loss

# ----------------------------
# 3. 改进的注意力模块 (使用 RMSNorm)
# ----------------------------
class FFTFrequencyWeight(nn.Module):
    """
    Apply learnable or fixed weights in frequency domain.
    """
    def __init__(self, seq_len: int, weight_type: str = "learnable"):
        super().__init__()
        self.seq_len = seq_len
        self.weight_type = weight_type

        if weight_type == "learnable":
            # Learnable weight for each frequency bin
            self.weight = nn.Parameter(torch.ones(seq_len // 2 + 1))
        elif weight_type == "lowpass":
            freqs = torch.arange(seq_len // 2 + 1) / (seq_len // 2 + 1)
            self.register_buffer("weight", 1.0 / (1.0 + 10 * freqs))
        elif weight_type == "highpass":
            # Fixed high-pass filter
            freqs = torch.arange(seq_len // 2 + 1) / (seq_len // 2 + 1)
            self.register_buffer("weight", freqs)
        else:  # "none"
            self.register_buffer("weight", torch.ones(seq_len // 2 + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        returns: [B, C, L] with frequency-weighted signal
        """
        B, C, L = x.shape
        assert L == self.seq_len, f"Expected seq_len={self.seq_len}, got {L}"

        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)  # [B, C, L//2 + 1]

        # Apply weight
        if self.weight_type == "learnable":
            weight = F.softplus(self.weight)  # Ensure positive
        else:
            weight = self.weight

        x_fft_weighted = x_fft * weight.unsqueeze(0).unsqueeze(0)  # [B, C, L//2+1]

        # IFFT
        x_weighted = torch.fft.irfft(x_fft_weighted, n=L, dim=-1)  # [B, C, L]
        return x_weighted

class PrototypeCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_proto: int, dropout: float = 0.2):
        super().__init__()
        self.context_proto = ContextPrototypes(num_proto, embed_dim)
        
        # 替换为 RMSNorm
        self.norm_x = RMSNorm(embed_dim)
        self.norm_p = RMSNorm(embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        x_norm = self.norm_x(x)
        proto = self.norm_p(self.context_proto.prototypes)
        proto = proto.unsqueeze(0).expand(B, -1, -1)

        attn_out, _ = self.cross_attn(query=x_norm, key=proto, value=proto)
        return x + self.dropout(attn_out)

class PrototypeBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_proto, dropout=0.2, ffn_ratio=4):
        super().__init__()
        self.proto_attn = PrototypeCrossAttention(embed_dim, num_heads, num_proto, dropout)
        # 替换为 RMSNorm
        self.norm = RMSNorm(embed_dim)

        hidden_dim = embed_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.proto_attn(x)
        x = x + self.ffn(self.norm(x))
        return x

class TransformerBackbone(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.enc_in = config.d_model
        self.seq_len = config.seq_len
        self.embed_dim = config.d_model
        self.num_heads = config.n_heads
        self.num_layers = config.e_layers
        self.dropout = config.dropout
        self.proto=config.optimizing_prototypes

        self.use_patch = getattr(config, "use_patch", 0)
        self.patch_size = getattr(config, "patch_size", 16)

        # FFT frequency weighting
        self.use_fft_weight = getattr(config, "use_fft_weight", True)
        self.fft_weight_type = getattr(config, "fft_weight_type", "learnable")
        if self.use_fft_weight:
            self.fft_weight = FFTFrequencyWeight(self.seq_len, self.fft_weight_type)

        # Patching
        if self.use_patch == 1:
            self.patch_embed = PatchEmbedding(patch_size=self.patch_size, embed_dim=self.embed_dim)
            transformer_seq_len = (self.seq_len + self.patch_size - 1) // self.patch_size
        elif self.use_patch == 2:
            self.patch_embed = ViTPatchEmbedding(
                seq_len=self.seq_len,
                patch_size=self.patch_size,
                in_channels=self.enc_in,
                embed_dim=self.embed_dim,
            )
            transformer_seq_len = self.patch_embed.n_patches
        else:
            self.input_proj = nn.Linear(self.enc_in, self.embed_dim)
            transformer_seq_len = self.seq_len

        # Positional encoding
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.zeros(1, transformer_seq_len, self.embed_dim))
            nn.init.normal_(self.pos_encoding, std=0.02)

        # [CLS] token
        self.use_cls_token = True
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)
            transformer_seq_len += 1

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.embed_dim)

        # ✅ Cross attention to prototypes
        self.proto_attn = PrototypeBlock(embed_dim=self.embed_dim, num_heads=self.num_heads,num_proto=config.K*config.num_class,dropout=self.dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        if self.use_fft_weight:
            x = self.fft_weight(x)

        # Patching
        if self.use_patch != 0:
            x = self.patch_embed(x)
        else:
            x = x.transpose(1, 2)
            x = self.input_proj(x)

        if self.use_pos_encoding:
            x = x + self.pos_encoding

        if self.use_cls_token:
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
            x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoder
        x = self.transformer(x)
        x = self.norm(x)

        # # ✅ Cross-attention with prototypes
        if(self.proto):
            x = self.proto_attn(x)
            # x = self.proto_attn(x)

        # Extract CLS and aggregate
        cls_tokens = x[:, 0]
        if self.use_patch == 1:
            cls_tokens = cls_tokens.view(B, C, self.embed_dim)
            cls_final = cls_tokens.mean(dim=1)
        else:
            cls_final = cls_tokens

        return cls_final

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc_in = config.enc_in
        self.seq_len = config.seq_len
        self.num_class = config.num_class
        self.k = getattr(config, "k", 5)
        self.n_prototypes_total = self.k * self.num_class
        self.embed_dim = config.d_model
        self.temperature = getattr(config, "temperature", 1)
        self.gamma_u = 0.9
        self.gamma_d = getattr(config, "gamma", 0.999)
        self.backbone = TransformerBackbone(config)
        self.optimizing_prototypes = config.optimizing_prototypes
        self.feature_extractor=MonaFeatureExtractor(self.enc_in,self.embed_dim)
        self.dB=1

        # 原型参数初始化
        directions = F.normalize(torch.randn(self.n_prototypes_total, self.embed_dim), dim=-1)
        magnitudes = torch.randn(self.n_prototypes_total, 1) * 0.2 + 1.0
        self.prototypes = nn.Parameter(directions * magnitudes, requires_grad=False)

        if self.optimizing_prototypes:
            self.classifier = ScoreAggregation(self.n_prototypes_total, self.num_class)
            # 初始化集成 Loss 模块
            self.criterion = IntegratedPrototypeLoss(
                self.n_prototypes_total, 
                self.num_class, 
                alpha=getattr(config, "proto_alpha", 0.5),
                beta=getattr(config, "proto_beta", 0.1)
            )
        else:
            self.classifier = nn.Linear(self.embed_dim, self.num_class)

    def forward(self, x,_, labels,_a):
        """
        返回: 
        如果 training: 返回 (aggregated_logits, prototype_logits) 
        如果 eval: 只返回 aggregated_logits
        """
        x1 = self.feature_extractor(x.transpose(1, 2))
        x = x1
        features = self.backbone(x, self.prototypes) # [B, D]

        if self.optimizing_prototypes:
            # 计算余弦相似度得分 (Prototype Logits)
            features_norm = F.normalize(features, p=2, dim=-1)
            prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
            proto_logits = torch.mm(features_norm, prototypes_norm.t()) # [B, K]

            # 聚合得分 (Aggregated Logits)
            class_logits = self.classifier(proto_logits)

            if self.training and labels is not None:
                self._update_prototypes(features, labels.long().squeeze(-1))
                # 训练模式下返回所有必要信息用于计算 Loss
                return class_logits, proto_logits
            
            return class_logits
        else:
            return self.classifier(features)

    def get_loss(self, forward_output, targets):
        """
        在训练脚本中调用: 
        output = model(x, labels)
        loss = model.get_loss(output, labels)
        """
        if isinstance(forward_output, tuple):
            aggregated_logits, prototype_logits = forward_output
            return self.criterion(aggregated_logits, prototype_logits, self.prototypes, targets)
        else:
            # 基础分类损失
            return F.cross_entropy(forward_output, targets)

    # ... _update_prototypes 函数保持不变 ...
    @torch.no_grad()
    def _update_prototypes(self, features, labels, eps=1e-6):
        B, D = features.shape
        gamma = 0.9 + (0.999 - 0.9) * math.exp(-self.dB / 100)
        # gamma=0.999
        self.dB+=1

        # 归一化方向 
        features_dir = F.normalize(features, dim=-1)
        proto_dir = F.normalize(self.prototypes.data, dim=-1)

        # 获取每个样本的真实类别
        labels = labels.long().view(-1)  # [B]

        # 初始化新原型
        proto_new = torch.zeros_like(self.prototypes.data)

        for c in range(self.num_class):
            # 找出属于类别 c 的样本
            mask = (labels == c)
            if mask.sum() == 0:
                proto_new[c * self.k : (c+1) * self.k] = self.prototypes.data[c * self.k : (c+1) * self.k]
                continue

            # 这些样本只与类别 c 的原型交互
            feat_subset = features_dir[mask]  # [N_c, D]
            proto_subset_dir = proto_dir[c * self.k : (c+1) * self.k]  # [k, D]

            # 计算分配权重 Q: [N_c, k]
            
            Q = torch.softmax(feat_subset @ proto_subset_dir.t(), dim=1)  # 可加温度

            # 用原始 features（非归一化）更新
            feat_raw_subset = features[mask]  # [N_c, D]
            # 计算每个原型的总接收权重（列和）
            weight_sum = Q.sum(dim=0, keepdim=True)  # [1, k]

            # 加权平均：避免幅值放大
            proto_updated = (Q.t() @ feat_raw_subset) / (weight_sum.t() + 1e-8)  # [k, D]

            proto_new[c * self.k : (c+1) * self.k] = proto_updated

        # EMA update
        self.prototypes.data.copy_(gamma * self.prototypes.data + (1 - gamma) * proto_new)