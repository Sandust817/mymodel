from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange


# ----------------------------
# 1. Score Aggregation Head
# ----------------------------
class ScoreAggregation(nn.Module):
    """Weighted aggregation over K prototypes per class."""
    def __init__(self, n_classes: int, n_prototypes: int, init_val: float = 0.5):
        super().__init__()
        self.weights = nn.Parameter(torch.full((n_classes, n_prototypes), init_val))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_classes, n_prototypes]
        returns: [B, n_classes]
        """
        weights = F.softmax(self.weights, dim=-1)  # [C, K]
        x = x * weights.unsqueeze(0)  # [B, C, K]
        return x.sum(dim=-1)  # [B, C]


# ----------------------------
# 2. TimePNP Model
# ----------------------------
class TimePNP(nn.Module):
    """
    Prototype-based Neural Part discovery for Time Series Classification.
    
    Input:  (B, C, L)
    Output: logits [B, n_classes], sim_maps [B, n_classes, K, L] (optional)
    """
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int,
        n_prototypes: int = 5,
        prototype_dim: Optional[int] = None,
        use_embedding_space: bool = True,
        gamma: float = 0.999,
        temperature: float = 0.2,
        sa_init: float = 0.5,
        normalize_prototypes: bool = True,
        optimizing_prototypes: bool = True,
    ):
        """
        Args:
            backbone: Feature extractor, input (B, C, L) → output (B, D) or (B, C, L)
            n_classes: Number of classes
            n_prototypes: Number of prototypes per class
            prototype_dim: Dimension of prototype space (if None, inferred from backbone)
            use_embedding_space: If True, prototypes live in backbone's output embedding space.
                                 If False, prototypes are full time series (C, L).
            gamma: Momentum update coefficient for prototypes
            temperature: Temperature for similarity logits
            sa_init: Initial value for score aggregation weights
            normalize_prototypes: Whether to L2-normalize prototypes
            optimizing_prototypes: Whether to update prototypes during training
        """
        super().__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.use_embedding_space = use_embedding_space
        self.gamma = gamma
        self.temperature = temperature
        self.normalize_prototypes = normalize_prototypes
        self.optimizing_prototypes = optimizing_prototypes

        # Determine prototype dimension and shape
        if use_embedding_space:
            # Prototypes in embedding space: [C, K, D]
            if prototype_dim is None:
                raise ValueError("prototype_dim must be specified when use_embedding_space=True")
            self.prototype_shape = (n_classes, n_prototypes, prototype_dim)
            self.prototypes = nn.Parameter(torch.randn(*self.prototype_shape))
        else:
            # Prototypes as full time series: [C, K, C_in, L]
            # We'll infer C_in and L during first forward pass
            self.prototypes = None  # Will be initialized in forward
            self.register_buffer("prototype_C", torch.tensor(0))
            self.register_buffer("prototype_L", torch.tensor(0))

        nn.init.trunc_normal_(self.prototypes if use_embedding_space else torch.tensor(0), std=0.02)

        self.classifier = ScoreAggregation(n_classes, n_prototypes, init_val=sa_init)

    def _init_time_prototypes(self, C_in: int, L: int):
        """Initialize time-series prototypes on first forward pass."""
        if not self.use_embedding_space and self.prototypes is None:
            self.prototype_C.copy_(torch.tensor(C_in))
            self.prototype_L.copy_(torch.tensor(L))
            self.prototypes = nn.Parameter(torch.randn(self.n_classes, self.n_prototypes, C_in, L))
            nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_sim_maps: bool = False,
    ) -> dict:
        """
        Args:
            x: [B, C, L] - input time series
            labels: [B,] - image-level labels (required for prototype update)
            return_sim_maps: whether to return similarity maps for visualization
        
        Returns:
            dict with keys:
                - "logits": [B, n_classes]
                - "similarity": [B, n_classes, K] (or [B, n_classes, K, L] if use_embedding_space=False)
                - (if return_sim_maps and not use_embedding_space) "sim_maps": [B, n_classes, K, L]
        """
        B, C_in, L = x.shape

        # Extract features
        features = self.backbone(x)  # Could be [B, D] or [B, C_out, L_out]

        if self.use_embedding_space:
            # Assume backbone outputs [B, D]
            if features.ndim != 2:
                # Global average pooling if needed
                if features.ndim == 3:
                    features = features.mean(dim=-1)  # [B, D]
                else:
                    raise ValueError(f"Unexpected backbone output shape: {features.shape}")
            D = features.shape[-1]

            # Normalize
            features = F.normalize(features, p=2, dim=-1)  # [B, D]
            prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)  # [C, K, D]

            # Compute similarity: [B, C, K]
            sim = torch.einsum("bd,ckd->bck", features, prototypes_norm)  # cosine similarity
            sim_maps = None

        else:
            # Prototypes are full time series
            self._init_time_prototypes(C_in, L)
            assert self.prototypes is not None

            # Normalize along channel and time? Or just flatten?
            # Here we normalize each (C, L) as a vector
            x_flat = rearrange(x, "b c l -> b (c l)")  # [B, C*L]
            proto_flat = rearrange(self.prototypes, "c k c_in l -> c k (c_in l)")  # [C, K, C*L]

            x_flat = F.normalize(x_flat, p=2, dim=-1)
            proto_flat = F.normalize(proto_flat, p=2, dim=-1)

            # Similarity per prototype: [B, C, K]
            sim = torch.einsum("bm,ckm->bck", x_flat, proto_flat)

            # For visualization: compute similarity at each time step (optional)
            if return_sim_maps:
                # Compute local similarity: [B, C, K, L]
                x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)  # normalize over channels
                proto_norm = F.normalize(self.prototypes, p=2, dim=2, eps=1e-8)  # [C, K, C, L]
                sim_maps = torch.einsum("bcl,ckcl->bckl", x_norm, proto_norm)
            else:
                sim_maps = None

        # Classification
        logits = self.classifier(sim) / self.temperature  # [B, C]

        outputs = {
            "logits": logits,
            "similarity": sim,
        }
        if sim_maps is not None:
            outputs["sim_maps"] = sim_maps

        # Online prototype update (only during training with labels)
        if self.training and labels is not None and self.optimizing_prototypes:
            with torch.no_grad():
                self._update_prototypes(features if self.use_embedding_space else x, labels, sim)

        return outputs

    @torch.no_grad()
    def _update_prototypes(self, features: torch.Tensor, labels: torch.Tensor, sim: torch.Tensor):
        """
        Momentum update of prototypes using hard assignment.
        """
        B = features.shape[0]
        device = features.device

        # Hard assignment: for each sample, assign to the best prototype of its true class
        # sim: [B, C, K]
        sim_true_class = sim[torch.arange(B, device=device), labels]  # [B, K]
        best_proto_idx = sim_true_class.argmax(dim=-1)  # [B]

        # Accumulate features per (class, prototype)
        for b in range(B):
            c = labels[b].item()
            k = best_proto_idx[b].item()
            feat = features[b:b+1]  # [1, ...]

            if self.use_embedding_space:
                old_proto = self.prototypes[c, k]  # [D]
                new_proto = self.gamma * old_proto + (1 - self.gamma) * feat.squeeze(0)
                self.prototypes.data[c, k].copy_(new_proto)
            else:
                old_proto = self.prototypes[c, k]  # [C, L]
                new_proto = self.gamma * old_proto + (1 - self.gamma) * feat.squeeze(0)
                self.prototypes.data[c, k].copy_(new_proto)

        if self.normalize_prototypes:
            self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=-1 if self.use_embedding_space else (2, 3))


# ----------------------------
# 3. Example Backbone: Simple TCN or MLP
# ----------------------------
class SimpleMLPBackbone(nn.Module):
    """A simple backbone that outputs an embedding."""
    def __init__(self, input_channels: int, seq_len: int, hidden_dim: int = 128, embed_dim: int = 64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)  # [B, C*L]
        return self.mlp(x)   # [B, embed_dim]


# ----------------------------
# 4. Usage Example
# ----------------------------
if __name__ == "__main__":
    B, C, L = 16, 3, 100
    n_classes = 10
    n_prototypes = 5
    embed_dim = 64

    # Dummy data
    x = torch.randn(B, C, L)
    labels = torch.randint(0, n_classes, (B,))

    # Backbone
    backbone = SimpleMLPBackbone(C, L, embed_dim=embed_dim)

    # TimePNP model
    model = TimePNP(
        backbone=backbone,
        n_classes=n_classes,
        n_prototypes=n_prototypes,
        prototype_dim=embed_dim,
        use_embedding_space=True,
        gamma=0.99,
        temperature=0.2,
        optimizing_prototypes=True
    )

    # Forward
    outputs = model(x, labels=labels, return_sim_maps=False)
    print("Logits shape:", outputs["logits"].shape)        # [16, 10]
    print("Similarity shape:", outputs["similarity"].shape) # [16, 10, 5]

    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs["logits"], labels)
    print("Loss:", loss.item())