from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import repeat, rearrange
import math

from layers.mona import MonaFeatureExtractor

class PrototypeCrossAttention(nn.Module):
    """
    Prototype Cross-Attention with Transformer-style residual connection and LayerNorm.
    Structure: x + Dropout(Attention(LayerNorm(x), prototypes))
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
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

    def __init__(self, n_prototypes_total: int,num_classes: int,  init_val: float = 0.2, temperature: float = 0.1):
        super().__init__()
        # 核心：计算每个类别对应的原型数量（确保 n_prototypes_total 是 num_classes 的整数倍）
        assert n_prototypes_total % num_classes == 0, "原型总数必须是类别数的整数倍"
        self.k = n_prototypes_total // num_classes  # 每个类别对应 k 个原型
        
        # 可学习权重：[num_classes, k]，每个类别下的 k 个原型各有一个权重
        self.weights = nn.Parameter(
            torch.full((num_classes, self.k), init_val, dtype=torch.float32)
        )
        self.temperature = temperature  # 权重归一化的温度系数，控制权重差异度

    def forward(self, prototype_logits: torch.Tensor) -> torch.Tensor:

        num_classes = self.weights.shape[0]
        B, _ = prototype_logits.shape
        

        prototype_logits_grouped = prototype_logits.reshape(B, num_classes, self.k)

        normalized_weights = F.softmax(self.weights / self.temperature, dim=-1) * self.k

        class_logits = (prototype_logits_grouped * normalized_weights).sum(dim=-1)
        
        return class_logits
    
def diversity_loss(self):
    loss = 0
    for c in range(self.num_class):
        start = c * self.k
        end = start + self.k
        proto_class = self.prototypes[start:end]  # [k, D]
        sim = torch.mm(proto_class, proto_class.t())  # [k, k]
        loss += (sim - torch.eye(self.k, device=sim.device)).pow(2).mean()
    return loss
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


# ----------------------------
# Transformer Backbone with Channel-Independent Patching
# ----------------------------
class TransformerBackbone(nn.Module):
    def __init__(self, config: Dict[str, Any], prototypes: Optional[nn.Parameter] = None):
        super().__init__()
        self.enc_in = config.d_model
        self.seq_len = config.seq_len
        self.embed_dim = config.d_model
        self.num_heads = config.n_heads
        self.num_layers = config.e_layers
        self.dropout = config.dropout

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
        x = self.proto_attn(x, prototypes)

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
        self.embed_dim = config.d_model
        self.temperature = getattr(config, "temperature", 0.2)
        self.gamma = getattr(config, "gamma", 0.999)
        self.k = getattr(config, "k", 5)
        self.n_prototypes_total = min(self.k * self.num_class, 200) 
        self.feature_extractor=MonaFeatureExtractor(self.enc_in,self.embed_dim)


        # New: Scalable loss config
        self.use_scalenorm = getattr(config, "use_scalenorm", False)
        self.scalable_alpha = getattr(config, "scalable_alpha", 1.0)
        self.scalable_gamma = getattr(config, "scalable_gamma", 2.0)

        # Backbone
        self.backbone =TransformerBackbone(config)

        # Prototypes
        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes_total, self.embed_dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)
        self.relu=nn.ReLU()

        self.optimizing_prototypes = True
        self.classifier=ScoreAggregation(self.n_prototypes_total,self.num_class)

    def forward(
        self,
        x: torch.Tensor,
        _a=None,
        labels: Optional[torch.Tensor] = None,
        _b=None,
        return_sim_maps: bool = False,
    ):
        #input=[B,L,C]
        x1 = self.feature_extractor(x.transpose(1, 2))
        x = x1.transpose(1, 2)

        observed_mask = torch.ones_like(x)  # 假设所有数据有效（实际场景可传入真实掩码）

        # Step 2: （可选）ScaleNorm归一化
        if self.use_scalenorm:
            scaled_x, _, _ = self.scaler(x, observed_mask)
        else:
            scaled_x = x  # 不归一化，直接使用原始数据
            
        if scaled_x.shape[1] != self.backbone.enc_in:
            x = scaled_x.transpose(1, 2)  # to [B, C, L]
        
        features = self.backbone(x,self.prototypes)  # [B, D]
        features = F.normalize(features, p=2, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)

        logits = torch.mm(features, prototypes_norm.t())  # [B, K]

        if self.training and self.optimizing_prototypes and labels is not None:
            # 注意：labels 在 forward 中是 Optional[Tensor]
            # 确保 labels 是 [B] 形状
            label_tensor = labels.long().squeeze(-1)  # [B]
            self._update_prototypes(features, label_tensor)
        class_logits = self.classifier(logits)  # [B, num_class]
        return class_logits

    @torch.no_grad()
    def _update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        labels = labels.to(device)
        B, D = features.shape
        k = self.k
        one_hot = F.one_hot(labels, num_classes=self.num_class).float()  # [B, C]
        rep = repeat(one_hot, 'b c -> b (c k)', k=k)  # [B, K]
        numerator = rep.t() @ features  # [K, D]
        counts = rep.sum(dim=0)         # [K]
        mask = counts > 0
        counts_masked = counts[mask].unsqueeze(1)  # [k_pos,1]
        new_protos = torch.zeros_like(self.prototypes, device=device)  # [K, D]
        new_protos[mask] = numerator[mask] / counts_masked
        # EMA update only for masked positions:
        old = self.prototypes.data.clone()
        updated = old
        updated[mask] = self.gamma * old[mask] + (1.0 - self.gamma) * new_protos[mask]
        self.prototypes.data = F.normalize(updated, p=2, dim=-1)

    def diversity_loss(self):
        """计算类内原型多样性损失（鼓励同类原型彼此不同）"""
        loss = 0.0
        for c in range(self.num_class):
            start = c * self.k
            end = start + self.k
            proto_class = self.prototypes[start:end]  # [k, D]
            # 计算余弦相似度矩阵 [k, k]
            sim = torch.mm(proto_class, proto_class.t())  # 已归一化，即余弦相似度
            # 目标：非对角线元素趋近于0，对角线为1
            target = torch.eye(self.k, device=sim.device)
            loss += F.mse_loss(sim, target)
        return loss / self.num_class  # 平均每类损失


        