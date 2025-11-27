import torch
print(torch.version.cuda)        # 编译时使用的 CUDA 版本
print(torch.cuda.is_available()) # 是否能实际使用 GPU