"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import torch
import torch.nn.functional as F
from torch import nn

# from models.common import manual_pad

def manual_pad(x: torch.Tensor, min_length: int) -> torch.Tensor:
    """
    Manual padding function that pads x to a minimum length with replicate padding.
    PyTorch padding complains if x is too short relative to the desired pad size, hence this function.

    :param x: Input tensor to be padded.
    :param min_length: Length to which the tensor will be padded.
    :return: Padded tensor of length min_length.
    """
    # Calculate amount of padding required
    pad_amount = min_length - x.shape[-1]
    # Split either side
    pad_left = pad_amount // 2
    pad_right = pad_amount - pad_left
    # Pad left (replicate first value)
    # print(x.shape)
    # print(pad_left)
    # pad_x = F.pad(x, [pad_left, 0], mode="constant", value=x[:, :, 0].item())
    pad_x = F.pad(x, [pad_left, 0], mode="constant", value=0.)
    # Pad right (replicate last value)
    # pad_x = F.pad(pad_x, [0, pad_right], mode="constant", value=x[:, :, -1].item())
    pad_x = F.pad(pad_x, [0, pad_right], mode="constant", value=0.)
    # print(pad_x.shape)
    return pad_x


class InceptionTimeFeatureExtractor(nn.Module):
    """
    顶层特征提取器：输入原始时序数据，输出最终高维特征
    in_channels: 输入通道数（原始数据通道，如单导联心电为1）
    out_channels: 最终输出通道数（默认32）
    """
    def __init__(
        self,
        in_channels: int,  # 明确命名为in_channels（输入通道）
        out_channels: int = 128,  # 明确命名为out_channels（最终输出通道）
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # instance_encoder：2个InceptionBlock串联，实现“特征提取→维度压缩”
        self.instance_encoder = nn.Sequential(
            # 第1个InceptionBlock：输入=in_channels，输出=4*out_channels（4个分支拼接）
            InceptionBlock(
                in_channels=in_channels,
                mid_channels=out_channels,  # 单分支输出通道（中间维度）
                padding_mode=padding_mode
            ),
            # 第2个InceptionBlock：输入=4*out_channels，输出=out_channels（最终目标维度）
            InceptionBlock(
                in_channels=4 * out_channels,
                mid_channels=out_channels,
                out_channels=out_channels,  # 新增out_channels，明确最终输出
                padding_mode=padding_mode
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, L] → 输出: [B, out_channels, L]
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = self.manual_pad(x, min_len)  # 补充manual_pad实现（避免报错）
            return self.instance_encoder(padded_x)
    
    def manual_pad(self, x: torch.Tensor, min_len: int) -> torch.Tensor:
        """补零到最小长度，保证replicate padding正常运行"""
        pad_len = min_len - x.shape[-1]
        return F.pad(x, (0, pad_len), mode="constant", value=0)


class InceptionBlock(nn.Module):
    """
    中层块：含残差连接的Inception模块组
    in_channels: 块输入通道数
    mid_channels: 内部InceptionModule单分支输出通道（中间维度）
    out_channels: 块最终输出通道数（默认4*mid_channels，即不压缩）
    """
    def __init__(
        self,
        in_channels: int,  # 输入通道
        mid_channels: int = 32,  # 内部单分支中间通道
        out_channels: int = None,  # 块输出通道（可选，默认4*mid_channels）
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
        n_modules: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        # 默认输出通道：4个分支拼接（mid_channels*4），若指定则用指定值
        self.out_channels = out_channels if out_channels is not None else 4 * mid_channels

        # 1. 主路径：n_modules个InceptionModule串联
        inception_modules = []
        for i in range(n_modules):
            module_in = in_channels if i == 0 else 4 * mid_channels  # 前一模块输出为4*mid_channels
            inception_modules.append(
                InceptionModule(
                    in_channels=module_in,  # 模块输入通道
                    branch_out=mid_channels,  # 模块内单分支输出通道
                    bottleneck_channels=bottleneck_channels,
                    padding_mode=padding_mode,
                ),
            )
        self.inception_modules = nn.Sequential(*inception_modules)

        # 2. 残差路径：确保输入输出通道匹配（最终与主路径相加）
        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,  # 残差输入=块输入
                out_channels=self.out_channels,  # 残差输出=块输出（保证能与主路径相加）
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=self.out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主路径输出：[B, 4*mid_channels, L]
        x_main = self.inception_modules(x)
        # 残差路径输出：[B, out_channels, L]
        x_res = self.residual(x)
        # 若主路径与残差通道数不同，用1×1卷积调整（新增兼容逻辑）
        if x_main.shape[1] != x_res.shape[1]:
            x_main = nn.Conv1d(
                in_channels=x_main.shape[1],
                out_channels=x_res.shape[1],
                kernel_size=1,
                padding="same"
            ).to(x.device)(x_main)
        # 输出：[B, out_channels, L]
        return F.relu(x_main + x_res)


class InceptionModule(nn.Module):
    """
    底层模块：单步多尺度特征捕捉（4分支并行）
    in_channels: 模块输入通道数
    branch_out: 每个分支的输出通道数
    total_out: 模块总输出通道数（固定=4*branch_out，4个分支拼接）
    """
    def __init__(
        self,
        in_channels: int,  # 模块输入通道
        branch_out: int = 32,  # 单分支输出通道
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.branch_out = branch_out
        self.total_out = 4 * branch_out  # 总输出=4分支拼接（固定）

        # 1. 瓶颈层：降维（减少计算量，仅当输入通道>1时使用）
        self.bottleneck: nn.Module
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,  # 瓶颈输入=模块输入
                out_channels=bottleneck_channels,  # 瓶颈输出=指定维度
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            )
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = 1  # 输入通道=1时，瓶颈输出=1

        # 2. 3个卷积分支（不同 kernel_size 捕捉多尺度特征）
        self.conv_branches = nn.ModuleList()  # 用conv_branches明确“分支”含义
        for kernel_size in [10, 20, 40]:
            self.conv_branches.append(
                nn.Conv1d(
                    in_channels=bottleneck_channels,  # 分支输入=瓶颈输出
                    out_channels=branch_out,  # 分支输出=branch_out
                    kernel_size=kernel_size,
                    padding="same",
                    padding_mode=padding_mode,
                )
            )

        # 3. 池化分支（带1×1卷积调整通道）
        self.pool_branch = nn.Sequential(  # 用pool_branch明确“池化分支”
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            nn.Conv1d(
                in_channels=in_channels,  # 池化分支输入=模块输入
                out_channels=branch_out,  # 池化分支输出=branch_out
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
        )

        # 4. 激活层：批量归一化+ReLU
        self.activation = nn.Sequential(
            nn.BatchNorm1d(num_features=self.total_out),  # 输入=模块总输出通道
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入：[B, in_channels, L]
        # 1. 瓶颈层降维：[B, in_channels, L] → [B, bottleneck_channels, L]
        x_bottleneck = self.bottleneck(x)
        
        # 2. 4分支并行计算（每个分支输出：[B, branch_out, L]）
        branch1 = self.conv_branches[0](x_bottleneck)  # kernel=10
        branch2 = self.conv_branches[1](x_bottleneck)  # kernel=20
        branch3 = self.conv_branches[2](x_bottleneck)  # kernel=40
        branch4 = self.pool_branch(x)                  # 池化分支
        
        # 3. 分支拼接：[B, 4*branch_out, L]（total_out）
        x_cat = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        # 4. 激活层：[B, total_out, L] → [B, total_out, L]
        x_out = self.activation(x_cat)
        
        # 输出：[B, total_out, L]（total_out=4*branch_out）
        return x_out