import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EnhancedInceptionModule(nn.Module):
    """增强版Inception模块：多分支+注意力融合+模块内残差"""
    def __init__(
        self,
        in_channels: int,
        branch_out: int,  # 单分支输出通道
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.branch_out = branch_out
        self.total_out = 4 * branch_out  # 4分支拼接总通道
        
        # 1. 动态瓶颈层：输入通道>8时降维（减少计算），否则保持
        self.bottleneck = nn.Conv1d(
            in_channels, 
            max(branch_out, in_channels // 2),  # 动态调整瓶颈维度
            kernel_size=1,
            padding="same",
            padding_mode=padding_mode
        ) if in_channels > 8 else nn.Identity()
        self.bottleneck_channels = max(branch_out, in_channels // 2) if in_channels > 8 else in_channels
        
        # 2. 多尺度卷积分支（kernel_size递进：3, 7, 11 → 捕捉不同尺度）
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                self.bottleneck_channels, branch_out, kernel_size=3, 
                padding="same", padding_mode=padding_mode
            ),
            nn.Conv1d(
                self.bottleneck_channels, branch_out, kernel_size=7, 
                padding="same", padding_mode=padding_mode
            ),
            nn.Conv1d(
                self.bottleneck_channels, branch_out, kernel_size=11, 
                padding="same", padding_mode=padding_mode
            )
        ])
        
        # 3. 池化分支（带可变形池化增强鲁棒性）
        self.pool_branch = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, padding=1, stride=1),  # 平均池化保留更多信息
            nn.Conv1d(
                in_channels, branch_out, kernel_size=1, 
                padding="same", padding_mode=padding_mode
            )
        )
        
        # 4. 分支注意力：动态加权不同分支的重要性
        self.branch_attn = nn.Sequential(
            nn.Conv1d(self.total_out, 4, kernel_size=1),  # 4分支→4个权重
            nn.Softmax(dim=1)  # 沿通道维度归一化
        )
        
        # 5. 模块内残差调整（确保输入输出通道匹配）
        self.residual_adjust = nn.Conv1d(
            in_channels, self.total_out, kernel_size=1, padding="same"
        ) if in_channels != self.total_out else nn.Identity()
        
        # 6. 激活与归一化
        self.norm_act = nn.Sequential(
            nn.BatchNorm1d(self.total_out),
            nn.GELU()  # GELU比ReLU更平滑，适合深层网络
        )

    def forward(self, x: Tensor) -> Tensor:
        # 输入：[B, in_channels, L]
        residual = self.residual_adjust(x)  # 模块内残差（通道适配）
        
        # 瓶颈层降维
        x_bn = self.bottleneck(x)
        
        # 多分支特征提取
        conv3 = self.conv_branches[0](x_bn)
        conv7 = self.conv_branches[1](x_bn)
        conv11 = self.conv_branches[2](x_bn)
        pool = self.pool_branch(x)
        
        # 分支拼接：[B, 4*branch_out, L]
        x_cat = torch.cat([conv3, conv7, conv11, pool], dim=1)
        
        # 分支注意力加权：每个分支特征乘以其权重
        attn = self.branch_attn(x_cat)  # [B, 4, L]
        attn = attn.unsqueeze(2).repeat(1, 1, self.branch_out, 1)  # [B, 4, branch_out, L]
        attn = attn.reshape(x_cat.shape)  # [B, 4*branch_out, L]
        x_attn = x_cat * attn  # 加权融合
        
        # 残差连接+归一化激活
        out = self.norm_act(x_attn + residual)
        return out  # [B, total_out, L]


class EnhancedInceptionBlock(nn.Module):
    """增强版Inception块：多模块串联+跨层残差"""
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,  # 单分支输出通道
        out_channels: int,  # 块最终输出通道
        n_modules: int = 3,  # 每个块包含3个模块（比原版更深）
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1. 主路径：n_modules个增强版InceptionModule串联
        modules = []
        for i in range(n_modules):
            module_in = in_channels if i == 0 else 4 * mid_channels  # 前一模块输出为4*mid_channels
            modules.append(
                EnhancedInceptionModule(
                    in_channels=module_in,
                    branch_out=mid_channels,
                    padding_mode=padding_mode
                )
            )
        self.main_path = nn.Sequential(*modules)
        
        # 2. 跨层残差路径（适配输入到输出通道）
        self.cross_residual = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=1, 
                padding="same", padding_mode=padding_mode
            ),
            nn.BatchNorm1d(out_channels)
        )
        
        # 3. 最终通道调整（确保输出为out_channels）
        self.final_adjust = nn.Conv1d(
            4 * mid_channels, out_channels, kernel_size=1, padding="same"
        ) if 4 * mid_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # 主路径特征提取
        x_main = self.main_path(x)  # [B, 4*mid_channels, L]
        x_main = self.final_adjust(x_main)  # [B, out_channels, L]
        
        # 跨层残差连接
        x_res = self.cross_residual(x)  # [B, out_channels, L]
        
        # 残差融合+激活
        return F.gelu(x_main + x_res)  # [B, out_channels, L]


class MonaFeatureExtractor(nn.Module):
    """增强版特征提取器：结合Inception优势+动态适配"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
        min_len: int = 21,  # 最小长度（根据最大卷积核11+安全余量设置）
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.min_len = min_len
        
        # 特征提取器：2个增强版InceptionBlock串联（先扩张再压缩）
        self.extractor = nn.Sequential(
            # 第1块：输入→4*out_channels（特征扩张，捕捉更多细节）
            EnhancedInceptionBlock(
                in_channels=in_channels,
                mid_channels=out_channels,
                out_channels=4 * out_channels,  # 扩张4倍
                padding_mode=padding_mode
            ),
            # 第2块：4*out_channels→out_channels（特征压缩，聚焦关键信息）
            EnhancedInceptionBlock(
                in_channels=4 * out_channels,
                mid_channels=out_channels,
                out_channels=out_channels,  # 最终输出
                padding_mode=padding_mode
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        # 输入：[B, in_channels, L]
        # 动态padding：确保长度≥min_len（避免卷积核过大导致的信息丢失）
        if x.shape[-1] < self.min_len:
            x = self._dynamic_pad(x, self.min_len)
        
        # 特征提取
        return self.extractor(x)  # 输出：[B, out_channels, L]

    def _dynamic_pad(self, x: Tensor, min_len: int) -> Tensor:
        """动态补零：优先在两端均匀补零，保持时序对称性"""
        pad_len = min_len - x.shape[-1]
        pad_left = pad_len // 2
        pad_right = pad_len - pad_left
        return F.pad(x, (pad_left, pad_right), mode="replicate")  # 复制填充比零填充更优


# 测试代码（验证维度正确性）
if __name__ == "__main__":
    # 模拟输入：[B=2, in_channels=1, L=50]（单通道时序数据）
    x = torch.randn(2, 1, 50)
    extractor = EnhancedInceptionFeatureExtractor(in_channels=1, out_channels=32)
    out = extractor(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")  # 预期：[2, 32, 50]