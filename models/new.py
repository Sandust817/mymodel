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


class TransformerBackboneNoPatch(nn.Module):
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


# ----------------------------
# Main Model with Prototype Interaction
# ----------------------------
class ProtoTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_class = config.num_class
        self.embed_dim = config.d_model
        self.k = getattr(config, "k", 5)
        self.gamma = getattr(config, "gamma", 0.99)
        self.n_prototypes_total = self.k * self.num_class

        # Backbone: transformer without patch
        self.backbone = TransformerBackboneNoPatch(
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
            seq_len=config.seq_len,
        )

        # Prototypes (non-parameter, updated by EMA)
        self.register_buffer("prototypes", torch.zeros(self.n_prototypes_total, self.embed_dim))

        # Prototype cross-attention
        self.proto_attn = PrototypeCrossAttention(self.embed_dim, num_heads=config.n_heads)

        # Classifier
        self.fc = nn.Linear(self.embed_dim, self.num_class)

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        """
        x: [B, C, L]
        """
        B = x.size(0)

        # 1️⃣ Transformer backbone
        feats = self.backbone(x)  # [B, L, D]
        feats = self.proto_attn(feats, self.prototypes)  # [B, L, D]

        # 2️⃣ Global pooling
        pooled = feats.mean(dim=1)  # [B, D]
        pooled = F.normalize(pooled, p=2, dim=-1)

        # 3️⃣ Prototype similarity (optional auxiliary signal)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        proto_logits = torch.mm(pooled, proto_norm.t())  # [B, K]
        class_logits = self.fc(pooled)

        # 4️⃣ Online update (if training)
        if self.training and labels is not None:
            self._update_prototypes(pooled, labels)

        return class_logits, proto_logits

    @torch.no_grad()
    def _update_prototypes(self, features, labels):
        """
        EMA update for prototypes per class
        """
        device = features.device
        one_hot = F.one_hot(labels, num_classes=self.num_class).float()
        rep = repeat(one_hot, "b c -> b (c k)", k=self.k)
        numerator = rep.t() @ features
        counts = rep.sum(dim=0)
        mask = counts > 0
        counts_masked = counts[mask].unsqueeze(1)
        new_protos = torch.zeros_like(self.prototypes, device=device)
        new_protos[mask] = numerator[mask] / counts_masked
        old = self.prototypes.clone()
        updated = old.clone()
        updated[mask] = self.gamma * old[mask] + (1 - self.gamma) * new_protos[mask]
        self.prototypes.copy_(F.normalize(updated, p=2, dim=-1))
