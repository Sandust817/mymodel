from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import repeat, rearrange
import math

from layers.mona import MonaFeatureExtractor


# ------------------------------------------------
# Prototype Cross-Attention
# ------------------------------------------------
class PrototypeCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, prototypes: Tensor) -> Tensor:
        residual = x.clone()
        x_norm = self.norm(x)
        proto = prototypes.unsqueeze(0).expand(x.size(0), -1, -1)
        attn_output, _ = self.cross_attn(x_norm, proto, proto)
        attn_output = self.norm(attn_output)
        return residual + self.dropout(attn_output)


# ------------------------------------------------
# Prototype Score Aggregation
# ------------------------------------------------
class ScoreAggregation(nn.Module):
    def __init__(self, n_prototypes_total: int, num_classes: int, init_val: float = 0.2, temperature: float = 0.1):
        super().__init__()
        assert n_prototypes_total % num_classes == 0
        self.k = n_prototypes_total // num_classes

        init_weights = torch.zeros(num_classes, self.k)
        init_weights[:, 0] = 2.0
        init_weights[:, 1:] = -2.0
        self.weights = nn.Parameter(init_weights)
        self.temperature = temperature

    def forward(self, prototype_logits: Tensor) -> Tensor:
        B = prototype_logits.size(0)
        num_classes = self.weights.size(0)

        grouped = prototype_logits.reshape(B, num_classes, self.k)
        w = F.softmax(self.weights / self.temperature, dim=-1) * self.k
        return (grouped * w).sum(dim=-1)


# ------------------------------------------------
# FFT Frequency Weighting
# ------------------------------------------------
class FFTFrequencyWeight(nn.Module):
    def __init__(self, seq_len: int, weight_type: str = "learnable"):
        super().__init__()
        self.seq_len = seq_len
        self.weight_type = weight_type

        if weight_type == "learnable":
            self.weight = nn.Parameter(torch.ones(seq_len // 2 + 1))
        elif weight_type == "lowpass":
            freqs = torch.arange(seq_len // 2 + 1) / (seq_len // 2 + 1)
            self.register_buffer("weight", 1.0 / (1.0 + 10 * freqs))
        elif weight_type == "highpass":
            freqs = torch.arange(seq_len // 2 + 1) / (seq_len // 2 + 1)
            self.register_buffer("weight", freqs)
        else:
            self.register_buffer("weight", torch.ones(seq_len // 2 + 1))

    def forward(self, x: Tensor) -> Tensor:
        B, C, L = x.shape
        x_fft = torch.fft.rfft(x, dim=-1)

        weight = F.softplus(self.weight) if self.weight_type == "learnable" else self.weight
        x_fft = x_fft * weight.unsqueeze(0).unsqueeze(0)

        return torch.fft.irfft(x_fft, n=L, dim=-1)


# ------------------------------------------------
# Patch Embedding
# ------------------------------------------------
class ViTPatchEmbedding(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.n_patches = math.ceil(seq_len / patch_size)

    def forward(self, x: Tensor) -> Tensor:
        B, C, L = x.shape
        pad_len = (self.patch_size - L % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        x = self.proj(x)
        return x.transpose(1, 2)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, C, L = x.shape
        if L % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - L % self.patch_size))
            L = x.size(-1)

        n_patches = L // self.patch_size
        x = x.view(B, C, n_patches, self.patch_size)
        x = x.view(B * C, n_patches, self.patch_size)
        return self.proj(x)


# ------------------------------------------------
# Transformer Backbone
# ------------------------------------------------
class TransformerBackbone(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.enc_in = config.d_model
        self.seq_len = config.seq_len
        self.embed_dim = config.d_model
        self.num_heads = config.n_heads
        self.num_layers = config.e_layers
        self.dropout = config.dropout

        self.use_patch = getattr(config, "use_patch", 0)
        self.patch_size = getattr(config, "patch_size", 16)

        self.use_fft_weight = getattr(config, "use_fft_weight", True)
        if self.use_fft_weight:
            self.fft_weight = FFTFrequencyWeight(self.seq_len, getattr(config, "fft_weight_type", "learnable"))

        if self.use_patch == 1:
            self.patch_embed = PatchEmbedding(self.patch_size, self.embed_dim)
            seq_len = (self.seq_len + self.patch_size - 1) // self.patch_size
        elif self.use_patch == 2:
            self.patch_embed = ViTPatchEmbedding(self.seq_len, self.patch_size, self.enc_in, self.embed_dim)
            seq_len = self.patch_embed.n_patches
        else:
            self.input_proj = nn.Linear(self.enc_in, self.embed_dim)
            seq_len = self.seq_len

        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, self.embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.selu=nn.SELU()
        nn.init.normal_(self.cls_token, std=0.02)
        seq_len += 1

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.proto_attn = PrototypeCrossAttention(self.embed_dim, self.num_heads)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: Tensor, prototypes: Tensor) -> Tensor:
        if self.use_fft_weight:
            x = self.fft_weight(x)

        if self.use_patch != 0:
            x = self.patch_embed(x)
        else:
            x = self.input_proj(x.transpose(1, 2))

        x = x + self.pos_encoding
        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.size(0))
        x = torch.cat([cls, x], dim=1)

        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
        x1=self.norm(x)
        for i in range(self.num_layers//2+1):
            x = self.proto_attn(x, prototypes)
        # x=x+self.selu(x1)
        cls_final = x[:, 0]
        return cls_final


# ------------------------------------------------
# Main Model
# ------------------------------------------------
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enc_in = config.enc_in
        self.seq_len = config.seq_len
        self.num_class = config.num_class
        self.embed_dim = config.d_model
        self.temperature = getattr(config, "temperature", 0.2)
        self.gamma = getattr(config, "gamma", 0.95)
        self.k = getattr(config, "k", 5)

        self.n_prototypes_total = min(self.k * self.num_class, 200)

        self.feature_extractor = MonaFeatureExtractor(self.enc_in, self.embed_dim)
        self.use_scalenorm = getattr(config, "use_scalenorm", False)

        self.backbone = TransformerBackbone(config)

        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes_total, self.embed_dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

        self.classifier = ScoreAggregation(self.n_prototypes_total, self.num_class)
        self.optimizing_prototypes = True
        self.dT = 1

    def forward(self, x: Tensor, _a=None, labels: Optional[Tensor] = None, _b=None, return_sim_maps: bool = False):
        x1 = self.feature_extractor(x.transpose(1, 2))
        x = x1.transpose(1, 2)

        if self.use_scalenorm:
            scaled_x, _, _ = self.scaler(x, torch.ones_like(x))
        else:
            scaled_x = x

        if scaled_x.shape[1] != self.backbone.enc_in:
            x = scaled_x.transpose(1, 2)

        features = self.backbone(x, self.prototypes)
        features = F.normalize(features, p=2, dim=-1)
        prot_norm = F.normalize(self.prototypes, p=2, dim=-1)

        logits = torch.mm(features, prot_norm.t())

        if self.training and self.optimizing_prototypes:
            self._update_prototypes(features)

        return self.classifier(logits)

    @torch.no_grad()
    def _update_prototypes(self, feats, sinkhorn_iters=5, temperature=0.13, eps=1e-6):
        prot_old = F.normalize(self.prototypes.data, dim=-1)

        logits = feats @ prot_old.t()
        grouped = logits.view(-1, self.num_class, self.k)
        Q = F.softmax(grouped / 0.05, dim=-1).view(-1, self.num_class * self.k)
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)

        for _ in range(sinkhorn_iters):
            Q = Q / (Q.sum(dim=1, keepdim=True) + eps)

        mass = Q.sum(0).unsqueeze(1) + eps
        proto_new = (Q.T @ feats) / mass

        if self.dT % 100 == 0:
            print(Q.mean(0))
        self.dT += 1

        updated = self.gamma * prot_old + (1 - self.gamma) * proto_new

        for i in range(updated.size(0)):
            for j in range(i):
                proj = (updated[i] @ updated[j]) * updated[j]
                updated[i] -= proj

        updated = F.normalize(updated, dim=-1)
        self.prototypes.data.copy_(updated)

    def diversity_loss(self):
        loss = 0.0
        for c in range(self.num_class):
            start = c * self.k
            end = start + self.k
            proto = self.prototypes[start:end]
            sim = F.cosine_similarity(proto.unsqueeze(1), proto.unsqueeze(0), dim=-1)
            loss += (sim - torch.eye(self.k, device=sim.device)).pow(2).mean()
        return loss
