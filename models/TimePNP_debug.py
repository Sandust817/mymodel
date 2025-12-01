from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import repeat, rearrange
import math

from layers.mona import MonaFeatureExtractor

class GatedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, gate_type="scalar"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for attention weights (standard in Transformer)
        self.attn_dropout = nn.Dropout(dropout)
        # Optional: dropout on output (also common)
        self.out_dropout = nn.Dropout(dropout)

        # Gate parameters
        if gate_type == "scalar":
            self.gate = nn.Parameter(torch.zeros(num_heads))  # (H,)
        else:
            # For vector gate per head (less common, but supported)
            self.gate = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        self.dropout_p = dropout

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        B, N_q, _ = query.shape
        N_k = key.shape[1]

        # Project queries, keys, values
        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_q, D)
        k = self.k_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, N_k, D)
        v = self.v_proj(value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N_k, D)

        # Scaled dot-product
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N_q, N_k)

        # Apply gating BEFORE softmax (key idea from NeurIPS 2025)
        if self.gate.dim() == 1:  # scalar per head
            gate = self.gate.view(1, -1, 1, 1)  # (1, H, 1, 1)
        else:
            # If using vector gate, broadcasting to (1, H, 1, D) doesn't directly apply to (N_q, N_k)
            # So scalar gate is strongly recommended
            gate = self.gate.view(1, self.num_heads, 1, self.head_dim)
            # In this case, you'd need to rethink gating strategy — usually not done
            # For simplicity and correctness, we assume scalar gate
            gate = gate.sum(dim=-1, keepdim=True)  # fallback, but not standard

        gated_logits = attn_logits * torch.sigmoid(gate)

        # Optional: apply attention mask (e.g., for causal attention)
        if attn_mask is not None:
            gated_logits = gated_logits + attn_mask

        # Softmax + Dropout on attention weights
        attn_weights = torch.softmax(gated_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights, v)  # (B, H, N_q, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)

        if need_weights:
            return attn_output, attn_weights.mean(dim=1)  # average over heads
        else:
            return attn_output

class PrototypeCrossAttention(nn.Module):
    """
    Prototype Cross-Attention with Transformer-style residual connection and LayerNorm.
    Structure: x + Dropout(Attention(LayerNorm(x), prototypes))
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # LayerNorm before attention (like norm_first=True in Transformer)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Multihead cross-attention: queries from x, keys/values from prototypes
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Dropout for residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] - input sequence (e.g., transformer output)
            prototypes: [K, D] - prototype matrix (normalized)
        Returns:
            x_out: [B, L, D] - residual output
        """
        # Save original x for residual connection
        residual = x
        
        # LayerNorm -> Cross-Attention -> Dropout -> Residual
        # x_norm = self.norm(x)  # [B, L, D]
        prototypes_batch = prototypes.unsqueeze(0).expand(x.size(0), -1, -1)  # [B, K, D]
        
        # Cross-attention: x queries attend to prototypes
        attn_output, _ = self.cross_attn(
            query=x,
            key=prototypes_batch,
            value=prototypes_batch
        )  # [B, L, D]
        attn_output=self.norm(attn_output)
        # Residual connection
        x_out = residual + attn_output
        return x_out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, norm_first=True):
        super().__init__()
        self.norm_first = norm_first

        # Attention block
        self.self_attn = GatedMultiheadAttention(d_model, nhead,dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN block
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()  # 或 nn.ReLU()，但 GELU 更常见于 Transformer

    def forward(self, src):
        # Pre-norm or Post-norm
        if self.norm_first:
            # Pre-normalization (used in modern Transformers like ViT, BERT, etc.)
            x = src + self._sa_block(self.norm1(src))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(src + self._sa_block(src))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x):
        x = self.self_attn(x,x,x)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output





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
        self.use_cls =config.use_cls

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

        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout,
            norm_first=True,
        )
        self.proto_encoder= PrototypeCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x: Tensor, prototypes: Tensor) -> Tensor:
        if self.use_fft_weight:
            x = self.fft_weight(x)

        if self.use_patch != 0:
            x = self.patch_embed(x)
        else:
            x = self.input_proj(x.transpose(1, 2))

        x = x + self.pos_encoding
        if(self.use_cls) :
            cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.size(0))
            x = torch.cat([cls, x], dim=1)
        
        x=self.transformer(x)
        x=self.proto_encoder(x,prototypes)

        if(self.use_cls):
            x = x[:, 0]
        return x


# ------------------------------------------------
# Main Model
# ------------------------------------------------
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enc_in = config.enc_in
        self.seq_len = config.seq_len
        self.num_class = getattr(config, "num_class", 2)
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
        if config.task_name=="anomaly_detection":
            self.classifier = nn.Linear(self.embed_dim, self.enc_in, bias=True)
        else:
            self.classifier = ScoreAggregation(self.n_prototypes_total, self.num_class)
        self.optimizing_prototypes = True
        self.dT = 1
        # self.ad=nn.Linear(self.)

    def forward(self, x: Tensor, _a=None, labels: Optional[Tensor] = None, _b=None,anomaly_mode=False, return_sim_maps: bool = False):
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


        if self.training and self.optimizing_prototypes:
            if anomaly_mode :
                features=features.mean(dim=1)
            self._update_prototypes(features)
        if anomaly_mode :
            return self.classifier(features)
        else:
            logits = torch.mm(features, prot_norm.t())
            return self.classifier(logits)

    @torch.no_grad()
    def _update_prototypes(self, feats, sinkhorn_iters=5, eps=1e-6):
        # feats assumed normalized already
        prot_old = F.normalize(self.prototypes.data, dim=-1)

        # scale to amplify differences
        scale = math.sqrt(self.embed_dim)
        logits = feats @ prot_old.t() * scale   # [B, K]

        grouped = logits.view(-1, self.num_class, self.k)
        group_temperature = 0.005
        Q = F.softmax(grouped / group_temperature, dim=-1).view(-1, self.num_class * self.k)

        # sinkhorn (row + col)
        for _ in range(sinkhorn_iters):
            Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
            Q = Q / (Q.sum(dim=0, keepdim=True) + eps)

        mass = Q.sum(0).unsqueeze(1) + eps
        proto_new = (Q.T @ feats) / mass

        if self.dT % 10 == 0:
            print("logits stats: mean %.4f std %.4f min %.4f max %.4f" %
                (logits.mean().item(), logits.std().item(), logits.min().item(), logits.max().item()))
            print("Q.mean:", Q.mean(0))

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
