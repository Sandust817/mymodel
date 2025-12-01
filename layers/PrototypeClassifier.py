# prototype_head.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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



class PrototypeHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        k: int = 5,
        ema_m: float = 0.99,
        sinkhorn_iters: int = 3,
        sinkhorn_temp: float = 0.05,
        agg_temperature: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.k = k
        self.C = num_classes
        self.K_total = num_classes * k

        self.ema_m = ema_m
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temp = sinkhorn_temp

        # prototypes
        self.prototypes = nn.Parameter(torch.randn(self.K_total, embed_dim))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

        self.aggregator = ScoreAggregation(
            n_prototypes_total=self.K_total,
            num_classes=self.C,
            temperature=agg_temperature
        )


    @torch.no_grad()
    def _ema_update_from_soft_assign(self, feats: torch.Tensor, eps: float = 1e-6):
        """EMA update prototypes."""
        feats =F.normalize(feats, dim=-1)
        prot_old = F.normalize(self.prototypes.data, dim=-1)

        logits = feats @ prot_old.t()
        grouped = logits.view(-1, self.num_classes, self.k)
        Q = F.softmax(grouped / 0.05, dim=-1).view(-1, self.num_classes * self.k)
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)

        for _ in range(self.sinkhorn_iters):
            Q = Q / (Q.sum(dim=1, keepdim=True) + eps)

        mass = Q.sum(0).unsqueeze(1) + eps
        proto_new = (Q.T @ feats) / mass

        updated = self.ema_m * self.prototypes.data + (1.0 - self.ema_m) * proto_new
        for i in range(updated.size(0)):
            for j in range(i):
                proj = (updated[i] @ updated[j]) * updated[j]
                updated[i] -= proj
        updated = F.normalize(updated, dim=-1)
        self.prototypes.data.copy_(updated)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pretraining mode (train): update prototypes
        Non-pretraining mode (eval): do NOT update prototypes
        """
        # Normalize features & prototypes
        feat_norm = F.normalize(features, dim=-1)                   # [B, D]
        prototypes_norm = F.normalize(self.prototypes, dim=-1)      # [K, D]

        # ------------ (1) Compute logits ------------ #
        logits = torch.mm(feat_norm, prototypes_norm.t())          # [B, K]
        class_logits = self.aggregator(logits)         # [B, C]

        # ------------ (2) Update prototypes only in training mode ------------ #
        if self.training:
            with torch.no_grad():
                self._ema_update_from_soft_assign(prototypes_norm)

        return class_logits


    def diversity_loss(self):
        loss = 0.0
        for c in range(self.num_classes):
            start = c * self.k
            end = start + self.k
            proto = self.prototypes[start:end]
            sim = F.cosine_similarity(proto.unsqueeze(1), proto.unsqueeze(0), dim=-1)
            loss += (sim - torch.eye(self.k, device=sim.device)).pow(2).mean()
        return loss
