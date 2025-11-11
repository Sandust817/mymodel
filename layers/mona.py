import torch
import torch.nn.functional as F
from torch import nn

def manual_pad(x: torch.Tensor, min_length: int) -> torch.Tensor:
    pad_amount = min_length - x.shape[-1]
    pad_left = pad_amount // 2
    pad_right = pad_amount - pad_left
    pad_x = F.pad(x, [pad_left, 0], mode="constant", value=0.)
    pad_x = F.pad(pad_x, [0, pad_right], mode="constant", value=0.)
    return pad_x
def frequency_enhance(x):
    freq = torch.fft.rfft(x, dim=-1)
    mag = freq.abs()
    phase = freq.angle()
    enhanced = torch.fft.irfft(mag * 0.5 + torch.exp(1j * phase), n=x.shape[-1])
    return enhanced.real


class MonaFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,  # 最终输出通道数
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 添加可学习权重：控制频率增强的贡献度（初始为0.5，通过sigmoid限制在[0,1]）
        self.freq_weight = nn.Parameter(torch.tensor(0.8))  # 初始权重0.5
        
        # 3层InceptionBlock串联，确保通道对齐
        self.instance_encoder = nn.Sequential(
            # 第1层：输入=in_channels，输出=4*out_channels
            InceptionBlock(
                in_channels=in_channels,
                mid_channels=out_channels,
                padding_mode=padding_mode
            ),
            InceptionBlock(
                in_channels=4 * out_channels,
                mid_channels=out_channels,
                padding_mode=padding_mode
            ),
            # 第3层：输入=4*out_channels，输出=out_channels
            InceptionBlock(
                in_channels=4 * out_channels,
                mid_channels=out_channels,
                out_channels=out_channels,
                padding_mode=padding_mode
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 假设输入x的shape: [B, C, L]（批量，通道，时序长度）
        min_len = 21
        # 处理短序列（padding到最小长度）
        if x.shape[-1] < min_len:
            x = self.manual_pad(x, min_len)
        
        # 计算频率增强后的特征
        x_enhanced = frequency_enhance(x)  # shape与x一致：[B, C, L]
        
        # 可学习权重融合：原始输入与增强输入按权重混合
        # weight = torch.sigmoid(self.freq_weight)  # 将权重限制在[0, 1]
        # weight=self.freq_weight
        x_merged = x  # 动态融合
        
        # 送入编码器提取特征
        return self.instance_encoder(x_merged)
    
    def manual_pad(self, x: torch.Tensor, min_len: int) -> torch.Tensor:
        pad_len = min_len - x.shape[-1]
        return F.pad(x, (0, pad_len), mode="constant", value=0)


# 以下InceptionBlock和InceptionModule代码与原版本一致，无需修改
class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 32,
        out_channels: int = None,
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
        n_modules: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.total_module_out = 4 * mid_channels  # 固定4*mid_channels（与InceptionModule一致）
        self.out_channels = out_channels if out_channels is not None else self.total_module_out

        inception_modules = []
        for i in range(n_modules):
            module_in = in_channels if i == 0 else self.total_module_out
            inception_modules.append(
                InceptionModule(
                    in_channels=module_in,
                    total_out=self.total_module_out,
                    bottleneck_channels=bottleneck_channels,
                    padding_mode=padding_mode,
                ),
            )
        self.inception_modules = nn.Sequential(*inception_modules)

        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=self.out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_main = self.inception_modules(x)
        x_res = self.residual(x)
        if x_main.shape[1] != x_res.shape[1]:
            x_main = nn.Conv1d(
                in_channels=x_main.shape[1],
                out_channels=x_res.shape[1],
                kernel_size=1,
                padding="same"
            ).to(x.device)(x_main)
        return F.relu(x_main + x_res)


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        total_out: int,
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.total_out = total_out
        self.n_branches = 5
        self.single_branch_out = (self.total_out + self.n_branches - 1) // self.n_branches

        self.bottleneck: nn.Module
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            )
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = 1

        self.conv_branches = nn.ModuleList()
        for kernel_size in [10, 20, 30, 40]:
            self.conv_branches.append(
                nn.Conv1d(
                    in_channels=bottleneck_channels,
                    out_channels=self.single_branch_out,
                    kernel_size=kernel_size,
                    padding="same",
                    padding_mode=padding_mode,
                )
            )

        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.single_branch_out,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
        )

        self.channel_clip = nn.Conv1d(
            in_channels=self.n_branches * self.single_branch_out,
            out_channels=self.total_out,
            kernel_size=1,
            padding="same",
            padding_mode=padding_mode
        )

        self.activation = nn.Sequential(
            nn.BatchNorm1d(num_features=self.total_out),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.bottleneck(x)
        branch1 = self.conv_branches[0](x_bottleneck)
        branch2 = self.conv_branches[1](x_bottleneck)
        branch3 = self.conv_branches[2](x_bottleneck)
        branch4 = self.conv_branches[3](x_bottleneck)
        branch5 = self.pool_branch(x)
        x_cat = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)
        x_clip = self.channel_clip(x_cat)
        x_out = self.activation(x_clip)
        return x_out