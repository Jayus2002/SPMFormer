import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
    论文地址：https://arxiv.org/pdf/2405.14343
    论文题目：Efficient Visual State Space Model for Image Deblurring （CVPR 2025）
    中文题目：SEM-Net：基于空间增强型状态空间模型的高效像素级图像修复建模方法（CVPR 2025）
    讲解视频：https://www.bilibili.com/video/BV1fh3NzdEUu/
        高效判别频域模块（Efficient discriminative frequency domain-based FFN，EDFFN）：
            实际意义：①频域计算量大的问题：在网络中间阶段进行频域处理，此时特征通道数是输入的数倍，执行快速傅里叶变换（FFT）的计算量随通道数呈线性增长。
                     ②局部特征局限性：传统 SSM（如 Mamba）将mamba主要聚焦于捕获全局长距离依赖，缺乏对局部特征处理，但局部纹理、边缘等细节信息同样关键。
            实现方式：①频域转换：快速傅里叶变换（FFT）从空间域转至频域。
                    ②频域筛选：用可学习量化矩阵 W，自适应保留有用的频率信息（重点保留高频细节）。
                    ③频域逆变换：经逆傅里叶变换（IFFT）转回空间域，与原特征融合后输出。
"""
class EDFFN(nn.Module):
    def __init__(self, dim, patch_size, network_depth, ffn_expansion_factor=4, bias=True,):
        super(EDFFN, self).__init__()
        # 计算隐藏层的特征维度，通常是输入维度的若干倍
        hidden_features = int(dim * ffn_expansion_factor)
        # 保存patch大小，用于后续分块处理
        self.patch_size = patch_size
        self.dim = dim
        self.network_depth = network_depth
        # 第一个1x1卷积层，用于提升特征维度，输出维度是隐藏层维度的两倍
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 深度可分离卷积，对每个通道单独处理，进一步提取特征
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        # 可学习的FFT参数，用于频域操作
        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        # 第二个1x1卷积层，用于将特征维度降回输入维度
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # 通过第一个卷积层提升特征维度【提升维度】
        x = self.project_in(x)
        # 经过深度可分离卷积后，将输出分成两部分
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 对第一部分应用GELU激活函数，然后与第二部分相乘
        x = F.gelu(x1) * x2
        # 通过第二个卷积层降低特征维度【降低维度】
        x = self.project_out(x)

        # 将特征图按指定patch大小进行分块重组
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)
        # 对分块后的特征图进行二维快速傅里叶变换，转换到频域
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        # 在频域中应用可学习的参数，对频域特征进行调整
        x_patch_fft = x_patch_fft * self.fft
        # 进行二维逆快速傅里叶变换，将特征从频域转回空间域
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

        # 将分块的特征图重新组合成完整的特征图
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,patch2=self.patch_size)
        return x
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None):
#         super().__init__()
#         # 输出特征维度，默认为输入特征维度
#         out_features = out_features or in_features
#         # 隐藏层特征维度，默认为输入特征维度
#         hidden_features = hidden_features or in_features

#         # 记录网络深度
#         # self.network_depth = network_depth

#         # 定义多层感知机的网络结构
#         self.mlp = nn.Sequential(
#             nn.Conv2d(in_features, hidden_features, 1),
#             nn.ReLU(True),
#             nn.Conv2d(hidden_features, out_features, 1)
#         )

#         # 对网络中的模块进行权重初始化
#         # self.apply(self._init_weights)

#     # def _init_weights(self, m):
#     #     if isinstance(m, nn.Conv2d):
#     #         # 计算增益系数
#     #         gain = (8 * self.network_depth) ** (-1 / 4)
#     #         # 计算输入和输出的扇入和扇出
#     #         fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
#     #         # 计算标准差
#     #         std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
#     #         # 初始化权重为截断正态分布
#     #         trunc_normal_(m.weight, std=std)
#     #         if m.bias is not None:
#     #             # 初始化偏置为0
#     #             nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # 前向传播，将输入通过多层感知机
#         return self.mlp(x)


if __name__ == "__main__":
    x = torch.randn(1, 32, 64, 64) # H 和 W 一定要能被patch_size整除
    model = EDFFN(dim=32,patch_size=8,network_depth=4)
    # model = Mlp(in_features=32, hidden_features=32*4)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")