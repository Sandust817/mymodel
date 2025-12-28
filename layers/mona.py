import torch
import torch.nn.functional as F
from torch import nn
# from fastai.basics import *

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
        # self.freq_weight = nn.Parameter(torch.tensor(0.5))  # 初始权重0.5
        
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
        if x.shape[2] < min_len:
            x = self.manual_pad(x, min_len)
        
        
        # 送入编码器提取特征
        return self.instance_encoder(x)
        # x_trans = x.permute(0, 2, 1)  # [B, L, C]
        # x_linear = self.linner(x_trans)  # [B, L, out_channels]
        # x_linear = x_linear.permute(0, 2, 1)  # [B, out_channels, L]
        
        # # 实例编码器输出
        # x_inception = self.instance_encoder(x)  # [B, out_channels, L]
        
        # # 修复：扩展self.a的维度，支持广播（[out_channels] → [1, out_channels, 1]）
        # a_expanded = self.a.unsqueeze(0).unsqueeze(-1)  # [1, out_channels, 1]
        # out = a_expanded*x_linear + x_inception  # [B, out_channels, L]
        # return out
    
    def manual_pad(self, x: torch.Tensor, min_len: int) -> torch.Tensor:
        pad_len = min_len - x.shape[2]
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
        self.total_module_out =mid_channels*4  # 固定4*mid_channels（与InceptionModule一致）
        self.out_channels = out_channels if out_channels is not None else self.total_module_out

        inception_modules = []
        for i in range(n_modules):
            module_in = in_channels if i == 0 else self.total_module_out
            inception_modules.append(
                InceptionModule(
                    ni=module_in,
                    nf=self.mid_channels
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

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()  # 关键：调用父类的__init__方法
        self.dim = dim

    def forward(self, *x):
        # 修正：torch.cat的参数是列表，*x会解包，所以需要用list(x)
        return torch.cat(list(x), dim=self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'

class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        super().__init__()  # 确保父类初始化
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 1 else False

        # 初始化bottleneck层（关键：先调用super().__init__再赋值）
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False) if bottleneck else noop
        self.convs = nn.ModuleList([
            nn.Conv1d(nf if bottleneck else ni, nf, k, padding="same", bias=False) 
            for k in ks
        ])
        self.maxconvpool = nn.Sequential(*[
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(ni, nf, 1, bias=False)
        ])
        self.concat = Concat()
        self.bn = nn.InstanceNorm1d(nf * 4)
        self.act = nn.GELU()  # Raw is ReLU

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        # 拼接卷积结果和池化结果
        conv_outs = [l(x) for l in self.convs]
        pool_out = self.maxconvpool(input_tensor)
        x = self.concat(*(conv_outs + [pool_out]))
        return self.act(self.bn(x))
        
# class InceptionModule(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         total_out: int,
#         bottleneck_channels: int = 32,
#         padding_mode: str = "replicate",
#     ) -> None:
#         super().__init__()
#         self.in_channels = in_channels
#         self.total_out = total_out
#         self.n_branches = 5
#         self.single_branch_out = (self.total_out + self.n_branches - 1) // self.n_branches

#         self.bottleneck: nn.Module
#         if in_channels > 1:
#             self.bottleneck = nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=bottleneck_channels,
#                 kernel_size=1,
#                 padding="same",
#                 padding_mode=padding_mode,
#             )
#         else:
#             self.bottleneck = nn.Identity()
#             bottleneck_channels = 1

#         self.conv_branches = nn.ModuleList()
#         for kernel_size in [10, 20, 30, 40]:
#             self.conv_branches.append(
#                 nn.Conv1d(
#                     in_channels=bottleneck_channels,
#                     out_channels=self.single_branch_out,
#                     kernel_size=kernel_size,
#                     padding="same",
#                     padding_mode=padding_mode,
#                 )
#             )

#         self.pool_branch = nn.Sequential(
#             nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
#             nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=self.single_branch_out,
#                 kernel_size=1,
#                 padding="same",
#                 padding_mode=padding_mode,
#             ),
#         )

#         self.channel_clip = nn.Conv1d(
#             in_channels=self.n_branches * self.single_branch_out,
#             out_channels=self.total_out,
#             kernel_size=1,
#             padding="same",
#             padding_mode=padding_mode
#         )

#         self.activation = nn.Sequential(
#             nn.BatchNorm1d(num_features=self.total_out),
#             nn.ReLU()
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_bottleneck = self.bottleneck(x)
#         branch1 = self.conv_branches[0](x_bottleneck)
#         branch2 = self.conv_branches[1](x_bottleneck)
#         branch3 = self.conv_branches[2](x_bottleneck)
#         branch4 = self.conv_branches[3](x_bottleneck)
#         branch5 = self.pool_branch(x)
#         x_cat = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)
#         x_clip = self.channel_clip(x_cat)
#         x_out = self.activation(x_clip)
#         return x_out