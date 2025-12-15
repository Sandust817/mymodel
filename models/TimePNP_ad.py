from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import repeat, rearrange
import math

from layers.mona import MonaFeatureExtractor

# --- keep FFTFrequencyWeight, ViTPatchEmbedding, PatchEmbedding, TransformerBackbone mostly the same ---
# (I reuse your original implementations; omit here to avoid redundancy in this message.
#  Assume they are identical to what you pasted, except TransformerBackbone no longer calls proto_attn.)
# --------------------------------------------------------------------

# energy and prototype utilities
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

class ScoreAggregation(nn.Module):
    def __init__(
        self,
        n_prototypes_total: int,
        num_classes: int,
        temperature: float = 0.2,  # 用于 logits 锐化（可选）
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_prototypes_total = n_prototypes_total
        self.temperature = temperature

        # 可学习映射矩阵: [K, C]
        # 初始化为 Xavier（或 small random）
        self.prototype_to_class = nn.Parameter(
            torch.randn(n_prototypes_total, num_classes) * 0.1
        )

    def forward(self, prototype_logits: torch.Tensor) -> torch.Tensor:
        """
        prototype_logits: [B, K] (cosine similarities ∈ [-1, 1])
        Returns: [B, num_classes]
        """
        B, K = prototype_logits.shape
        assert K == self.n_prototypes_total, f"Expected {self.n_prototypes_total}, got {K}"

        # 可选：锐化 logits（增强判别力）
        logits_sharpened = prototype_logits / self.temperature  # [B, K]

        # 映射到类别 logits: [B, K] @ [K, C] = [B, C]
        class_logits = logits_sharpened @ self.prototype_to_class  # [B, C]

        return class_logits
    
# ----------------------------
# FFT Frequency Weighting Module
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


class ViTPatchEmbedding(nn.Module):
    """
    Standard ViT-style Patch Embedding for 1D sequences.
    Input:  [B, C, L]
    Output: [B, n_patches, embed_dim]
    """
    def __init__(self, seq_len: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # Conv1d acts as patchify + linear projection
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # compute number of patches after padding
        self.n_patches = math.ceil(seq_len / patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        returns: [B, n_patches, embed_dim]
        """
        B, C, L = x.shape
        pad_len = (self.patch_size - L % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode='constant', value=0)
        x = self.proj(x)               # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)          # [B, n_patches, embed_dim]
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        """
        Channel-independent patching with padding support.
        Input:  [B, C, L]  (L can be any positive integer)
        Output: [B*C, n_patches, embed_dim]
        """
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        returns: [B*C, n_patches, embed_dim]
        """
        B, C, L = x.shape
        patch_size = self.patch_size

        # Compute required padding
        if L % patch_size != 0:
            pad_len = patch_size - (L % patch_size)
            # Pad on the right (temporal dimension)
            x = F.pad(x, (0, pad_len), mode='constant', value=0)  # [B, C, L + pad_len]
            new_L = L + pad_len
        else:
            new_L = L

        n_patches = new_L // patch_size

        # Reshape to [B, C, n_patches, patch_size]
        x = x.view(B, C, n_patches, patch_size)

        # Flatten batch and channel: [B*C, n_patches, patch_size]
        x = x.view(B * C, n_patches, patch_size)

        # Project each patch to embedding
        x = self.proj(x)  # [B*C, n_patches, embed_dim]
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
        if(config.task_name=='anomaly_detection'):
            self.use_cls_token = False
        else:
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
        self.proto_attn = PrototypeCrossAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

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
        # if(self.proto):
        #     x = self.proto_attn(x, prototypes)
            # x = self.proto_attn(x, prototypes)

        # Extract CLS and aggregate
        if self.use_cls_token:
            x = x[:, 0]
        if self.use_patch == 1:
            cls_tokens = x.view(B, C, self.embed_dim)
            cls_final = cls_tokens.mean(dim=1)
        else:
            cls_final = x

        return cls_final

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc_in = config.enc_in
        self.seq_len = config.seq_len
        self.num_class = getattr(config, "num_class", None)
        self.embed_dim = config.d_model
        self.temperature = getattr(config, "temperature", 0.1)  # for energy
        self.gamma_proto = getattr(config, "proto_ema_gamma", 0.999)
        self.k = getattr(config, "k", 5)
        self.n_prototypes_total = getattr(config, "n_prototypes_total", 80)

        # Feature extractor (you had MonaFeatureExtractor)
        self.feature_extractor = MonaFeatureExtractor(self.enc_in, self.embed_dim)

        # Backbone - reuse TransformerBackbone but we removed proto_attn call
        self.backbone = TransformerBackbone(config)

        # prototypes
        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes_total, self.embed_dim))
        nn.init.xavier_normal_(self.prototypes)
        # normalize initially (always keep prototypes normalized for cosine usage)
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

        # Regularization weights
        self.optimizing_prototypes=config.optimizing_prototypes
        self.use_sinkhorn = getattr(config, "use_sinkhorn", True)
        self.sinkhorn_iters = getattr(config, "sinkhorn_iters", 3)
        self.sinkhorn_eps = getattr(config, "sinkhorn_eps", 0.05)
        self.classifier = nn.Linear(self.embed_dim, self.enc_in, bias=True)

        # for numerical stability
        self.eps = 1e-6

    def forward(self, x, _a=None, labels: Optional[Tensor] = None, _b=None, return_sim_maps: bool = False):
        x1 = self.feature_extractor(x.transpose(1, 2))
        x = x1.transpose(1, 2)


        scaled_x = x

        if scaled_x.shape[1] != self.backbone.enc_in:
            x = scaled_x.transpose(1, 2)

        features = self.backbone(x, self.prototypes)
        features = F.normalize(features, p=2, dim=-1)

        if self.training and self.optimizing_prototypes:
            feature=features.mean(dim=1)
            self._update_prototypes(feature)
        return self.classifier(features)

    @torch.no_grad()
    def _update_prototypes(self, features: torch.Tensor):
        """
        Update prototypes with features (features should be normalized already)
        features: [M, D]  (M may be < B)
        Algorithm:
            - Option A: simple soft-assignment (softmax over cos sims) -> weighted sum
            - Option B: Sinkhorn balanced assignment (optional)
            - EMA update with gamma_proto
            - Normalize prototypes after update
        """
        M, D = features.shape
        K = self.n_prototypes_total
        proto_dir = F.normalize(self.prototypes.data, dim=-1)  # [K, D]
        logits = features @ proto_dir.t()  # [M, K]

        # soft assignment
        if self.use_sinkhorn and M >= K:
            # Sinkhorn-like balanced assignment over the selected subset
            Q = torch.exp(logits / (self.sinkhorn_eps + 1e-8))
            Q = Q / (Q.sum() + 1e-8)
            r = torch.ones(M, device=Q.device) / M
            c = torch.ones(K, device=Q.device) / K
            for _ in range(self.sinkhorn_iters):
                Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
                Q = Q * r.unsqueeze(1)
                Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12)
                Q = Q * c.unsqueeze(0)
            # Now Q is [M, K]
        else:
            # simple softmax along K
            Q = F.softmax(logits, dim=1)  # [M, K]

        weight_sum = Q.sum(dim=0, keepdim=True).t()  # [K, 1]
        proto_new = (Q.t() @ features)  # [K, D]
        denom = weight_sum + self.eps
        proto_new = proto_new / denom  # [K, D] (if some proto got zero mass, remains small)

        # EMA update on prototypes
        gamma = self.gamma_proto
        updated = gamma * self.prototypes.data + (1.0 - gamma) * proto_new
        # Normalize updated prototypes to unit vectors (since we compare by cosine)
        updated = F.normalize(updated, dim=-1)
        self.prototypes.data.copy_(updated)

    # (Optional) helper to compute anomaly score directly (energy)
    def score(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad():
            return self.forward(x, labels=None)
