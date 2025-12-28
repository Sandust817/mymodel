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
        self.k_levels = getattr(config, "k_levels",[3,4])  # 从config读取不定长数组，也可直接赋值如 [2,3]
        self.counts = [self.num_class * k for k in self.k_levels]  
        self.weights=[0.1,0.2]

        # 2. 动态生成对应层数的原型参数（核心优化：适配不定长k_levels）
        self.prototype_layers = nn.ParameterList()  # 使用nn.ParameterList管理多个原型层
        for count in self.counts:
            # 按count数量初始化原型，保持原有参数配置（randn、requires_grad=False）
            prototype = nn.Parameter(
                torch.randn(count, self.embed_dim),
                requires_grad=False
            )
            self.prototype_layers.append(prototype)

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

    def forward(self, x, _, labels, _a):
        """
        前向传播：
        - 训练模式：返回 (最终分类响应, 总损失)
        - 测试模式：仅返回 最终分类响应
        无字典返回，直接返回计算好的损失
        """
        # 1. 特征提取
        x = self.feature_extractor(x.transpose(1, 2))
        top_prototypes = self.prototype_layers[-1]
        features = self.backbone(x, top_prototypes)  # [B, D]

        # 2. 多阶段推理：获取所有层级的logits和响应
        stage_logits = []
        stage_responses = []
        for layer_idx in range(len(self.prototype_layers)):
            cur_prototypes = self.prototype_layers[layer_idx]
            cur_k = self.k_levels[layer_idx]
            cur_logits, cur_resp = self._get_stage_logits(features, cur_prototypes, cur_k)
            stage_logits.append(cur_logits)
            stage_responses.append(cur_resp)
        top_response=stage_responses[-1]
        # 3. 训练模式：计算总损失 + 更新原型
        if self.training and labels is not None:
            targets = labels.long().squeeze(-1)
            self._update_all_hierarchy(features, targets)


            # 返回：(最终分类响应, 总损失) （无字典，直接返回计算好的loss）
            top_response = stage_responses[-1]
            return top_response, self.get_loss(stage_responses,stage_logits,targets)
        return top_response

    def get_loss(self, stage_responses,stage_logits, targets):

        total_loss =0  # 初始化总损失

        # 逐层级计算损失 + 加权求和（类似 weights[0]*loss1 + weights[1]*loss2 + ...）
        for layer_idx in range(len(self.prototype_layers)):
            cur_weight = self.weights[layer_idx]
            cur_logits = stage_logits[layer_idx]
            cur_resp = stage_responses[layer_idx]
            # 计算当前层级损失（调用损失函数）
            criterion = nn.CrossEntropyLoss()
            cur_loss = criterion(cur_resp,  targets)
            # 加权累加总损失
            total_loss += cur_weight * cur_loss

        # 更新所有层级原型

        return total_loss

    def _get_stage_logits(self, features, prototypes, k_per_class):
        """
        计算特征与原型的相似度，并按类别分组聚合
        """
        # 归一化计算余弦相似度
        feat_norm = F.normalize(features, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        logits = torch.mm(feat_norm, proto_norm.t()) / self.temperature # [B, num_class * k]
        
        # 模拟“筛选”过程：LogSumExp 聚合到类级别
        B = logits.shape[0]
        grouped_logits = logits.view(B, self.num_class, k_per_class)
        # 使用 LogSumExp 代表该类在该阶段的综合响应
        class_response = torch.logsumexp(grouped_logits, dim=-1) # [B, num_class]
        
        return logits, class_response

    @torch.no_grad()
    def _update_prototypes(self, prototypes, features, labels, k, gamma):
        """
        单一层级原型的纯特征驱动更新（无层级约束，底层原型专用/上层原型自身更新值计算）
        参数：
            prototypes: 当前层原型 [count, D]
            features: 输入特征 [B, D]
            labels: 样本标签 [B]
            k: 当前层每个类的原型数
            gamma: EMA更新系数
        返回：
            proto_new: 当前层仅由特征计算的新原型 [count, D]
            proto_class_centers: 当前层原型的类内均值 [num_class, D]（用于约束上层）
        """
        B, D = features.shape
        # 归一化方向（保持你原始逻辑）
        features_dir = F.normalize(features, dim=-1)
        proto_dir = F.normalize(prototypes.data, dim=-1)
        labels = labels.long().view(-1)  # [B]

        # 初始化新原型
        proto_new = torch.zeros_like(prototypes.data)

        for c in range(self.num_class):
            # 找出类别c的样本
            mask = (labels == c)
            if mask.sum() == 0:
                # 无样本时保持原原型
                proto_new[c * k : (c+1) * k] = prototypes.data[c * k : (c+1) * k]
                continue

            # 类别c的样本和原型子集
            feat_subset = features_dir[mask]  # [N_c, D]
            proto_subset_dir = proto_dir[c * k : (c+1) * k]  # [k, D]
            feat_raw_subset = features[mask]  # [N_c, D]（原始特征，非归一化）

            # 计算分配权重Q（保持你原始逻辑）
            Q = torch.softmax(feat_subset @ proto_subset_dir.t(), dim=1)  # [N_c, k]
            weight_sum = Q.sum(dim=0, keepdim=True)  # [1, k]

            # 加权平均更新（避免幅值放大，保持你原始逻辑）
            proto_updated = (Q.t() @ feat_raw_subset) / (weight_sum.t() + 1e-8)  # [k, D]
            proto_new[c * k : (c+1) * k] = proto_updated

        # 计算当前层原型的类内均值（每个类对应k个原型的均值，用于约束上层）
        proto_class_centers = prototypes.data.view(self.num_class, k, D).mean(dim=1)  # [num_class, D]

        return proto_new, proto_class_centers

    @torch.no_grad()
    def _update_all_hierarchy(self, features, labels,update_alpha=0.5):
        """
        层级依赖式原型更新：
        1.  先更新最底层（第0层）：仅由原始特征驱动，无上层约束
        2.  再更新上层（第1层→最后一层）：新原型 = alpha*自身特征更新值 + (1-alpha)*底层原型约束值
        3.  所有层级最终执行EMA平滑更新
        """
        # 1. 计算EMA更新系数（保持你原始逻辑）
        gamma = 0.9 + (0.999 - 0.9) * math.exp(-self.dB / 100)
        self.dB += 1

        # 存储每层原型的类内均值（用于约束上一层）
        layer_class_centers = []
        num_layers = len(self.prototype_layers)

        # 2. 先更新最底层（第0层）：纯特征驱动，无层级约束
        bottom_layer_idx = 0
        bottom_prototypes = self.prototype_layers[bottom_layer_idx]
        bottom_k = self.k_levels[bottom_layer_idx]
        # 计算底层纯特征驱动的新原型 + 类内均值
        bottom_proto_new_feat, bottom_proto_centers = self._update_prototypes(
            bottom_prototypes, features, labels, bottom_k, gamma
        )
        # 底层执行EMA更新（无混合，直接用特征更新值）
        bottom_prototypes.data.copy_(
            gamma * bottom_prototypes.data + (1 - gamma) * bottom_proto_new_feat
        )
        # 存储底层类内均值，用于约束第1层
        layer_class_centers.append(bottom_proto_centers)

        # 3. 迭代更新上层（第1层 → 最后一层）：层级依赖混合更新
        for layer_idx in range(1, num_layers):
            # 当前层核心参数
            cur_prototypes = self.prototype_layers[layer_idx]
            cur_k = self.k_levels[layer_idx]
            cur_count = self.counts[layer_idx]

            # 底层参数（上一层，即layer_idx-1层）
            prev_layer_centers = layer_class_centers[layer_idx - 1]  # [num_class, D]（上一层类内均值）

            # 步骤1：计算当前层自身的特征驱动更新值（纯特征更新，无约束）
            cur_proto_new_feat, cur_proto_centers = self._update_prototypes(
                cur_prototypes, features, labels, cur_k, gamma
            )

            # 步骤2：计算底层原型对当前层的约束值（将上一层类内均值扩展为当前层原型维度）
            # 上一层每个类的均值 → 扩展为当前层每个类的k个原型（与当前层原型结构匹配）
            prev_layer_constraint = prev_layer_centers.repeat_interleave(cur_k, dim=0)  # [cur_count, D]

            # 步骤3：混合更新值 = alpha*自身特征更新值 + (1-alpha)*底层约束值
            cur_proto_new_combined = update_alpha * cur_proto_new_feat + \
                                     (1 - update_alpha) * prev_layer_constraint

            # 步骤4：当前层执行EMA平滑更新
            cur_prototypes.data.copy_(
                gamma * cur_prototypes.data + (1 - gamma) * cur_proto_new_combined
            )

            # 步骤5：存储当前层类内均值，用于约束下一层（若存在）
            layer_class_centers.append(cur_proto_centers)