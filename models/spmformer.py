import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# import netron
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_

import sys
sys.path.append('/home/benchunlei/dl/SPMFormer/test')
# from wavelets_test import HeightWidthDiagonalFeatureProcessor
from conv_test import HeightWidthFeatureDepthwiseConv


"""
在trasform块中加小波变换残差    
"""
# 代码功能相关注释：此部分导入了代码所需的各种库，包括PyTorch相关库、数学库和初始化函数等
# 下面的注释记录了不同版本的基础子融合相关信息
##########################
# base:sub-fusion
# v2.0 ->pe1+pe -> mask_2 + pe ->mask2,pe1
# v1.0 ->pe1 (+0.04) -> pe(K,B)
#
#
########################
# 定义Revised LayerNorm类，用于实现修订后的层归一化操作
class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        # 定义归一化所需的epsilon值
        self.eps = eps
        # 定义是否分离梯度的标志
        self.detach_grad = detach_grad

        # 定义可训练的权重参数
        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        # 定义可训练的偏置参数
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        # 定义两个卷积层，用于元信息处理
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        # 初始化meta1的权重为截断正态分布
        trunc_normal_(self.meta1.weight, std=.02)
        # 初始化meta1的偏置为1
        nn.init.constant_(self.meta1.bias, 1)

        # 初始化meta2的权重为截断正态分布
        trunc_normal_(self.meta2.weight, std=.02)
        # 初始化meta2的偏置为0
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        # 计算输入的均值
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        # 计算输入的标准差
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        # 对输入进行归一化处理
        normalized_input = (input - mean) / std

        if self.detach_grad:
            # 如果需要分离梯度，则对标准差和均值分离梯度后输入到卷积层
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            # 否则直接输入到卷积层
            rescale, rebias = self.meta1(std), self.meta2(mean)

        # 对归一化后的输入进行缩放和平移
        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias

# 定义多层感知机类
class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        # 输出特征维度，默认为输入特征维度
        out_features = out_features or in_features
        # 隐藏层特征维度，默认为输入特征维度
        hidden_features = hidden_features or in_features

        # 记录网络深度
        self.network_depth = network_depth

        # 定义多层感知机的网络结构
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        # 对网络中的模块进行权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # 计算增益系数
            gain = (8 * self.network_depth) ** (-1 / 4)
            # 计算输入和输出的扇入和扇出
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            # 计算标准差
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            # 初始化权重为截断正态分布
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                # 初始化偏置为0
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 前向传播，将输入通过多层感知机
        return self.mlp(x)

# 将输入特征图划分为多个窗口
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows

# 将划分的窗口重新合并为特征图
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# 计算窗口内的相对位置
def get_relative_positions(window_size):
    # 生成高度方向的坐标
    coords_h = torch.arange(window_size)
    # 生成宽度方向的坐标
    coords_w = torch.arange(window_size)

    # 生成坐标网格
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    # 展平坐标
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    # 计算相对位置
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    # 调整相对位置的维度
    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    # 对相对位置取对数并添加符号
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log

# 定义窗口注意力类
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        # 特征维度
        self.dim = dim
        # 窗口大小
        self.window_size = window_size  # Wh, Ww
        # 注意力头的数量
        self.num_heads = num_heads
        # 每个注意力头的维度
        head_dim = dim // num_heads
        # 缩放因子
        self.scale = head_dim ** -0.5

        # 计算窗口内的相对位置
        relative_positions = get_relative_positions(self.window_size)
        # 将相对位置注册为缓冲区
        self.register_buffer("relative_positions", relative_positions)
        # 定义元信息处理的网络结构
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        # 定义softmax激活函数
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape

        # 调整qkv的维度
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        # 分离q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # 对q进行缩放
        q = q * self.scale
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1))

        # 计算相对位置偏置
        relative_position_bias = self.meta(self.relative_positions)
        # 调整相对位置偏置的维度
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # 将相对位置偏置添加到注意力分数中
        attn = attn + relative_position_bias.unsqueeze(0)

        # 对注意力分数进行softmax操作
        attn = self.softmax(attn)

        # 计算注意力输出
        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return x

# 定义注意力类
class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        # 特征维度
        self.dim = dim
        # 每个注意力头的维度
        self.head_dim = int(dim // num_heads)
        # 注意力头的数量
        self.num_heads = num_heads

        # 窗口大小
        self.window_size = window_size
        # 窗口移动的大小
        self.shift_size = shift_size

        # 网络深度
        self.network_depth = network_depth
        # 是否使用注意力机制的标志
        self.use_attn = use_attn
        # 卷积类型
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            # 如果卷积类型为'Conv'，定义卷积网络结构
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            # 如果卷积类型为'DWConv'，定义深度可分离卷积
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            # 如果卷积类型为'DWConv'或使用注意力机制，定义V和投影层
            self.V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            # 如果使用注意力机制，定义QK卷积层和窗口注意力模块
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        # 对网络中的模块进行权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                # 如果是QK卷积层，计算扇入和扇出并初始化权重
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                # 否则，根据网络深度计算增益系数并初始化权重
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                # 初始化偏置为0
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        # 计算需要填充的高度
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        # 计算需要填充的宽度
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            # 如果需要移动窗口，进行相应的填充
            x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
                      mode='reflect')
        else:
            # 否则，进行简单的填充
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape

        if self.conv_type == 'DWConv' or self.use_attn:
            # 如果卷积类型为'DWConv'或使用注意力机制，计算V
            V = self.V(X)

        if self.use_attn:
            # 如果使用注意力机制，计算QK并拼接QKV
            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            # 进行窗口移动和填充
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # 划分窗口
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

            # 计算窗口注意力
            attn_windows = self.attn(qkv)

            # 合并窗口
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # 反向移动窗口
            out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
            attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                # 如果卷积类型为'Conv'或'DWConv'，计算卷积输出并与注意力输出相加后投影
                conv_out = self.conv(V)
                out = self.proj(conv_out + attn_out)
            else:
                # 否则，直接对注意力输出进行投影
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                # 如果不使用注意力机制且卷积类型为'Conv'，直接进行卷积操作
                out = self.conv(X)  # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                # 如果不使用注意力机制且卷积类型为'DWConv'，对卷积输出进行投影
                out = self.proj(self.conv(V))

        return out

# 定义Transformer块类
class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.dim =dim
        # 是否使用注意力机制的标志
        self.use_attn = use_attn
        
        # 是否对MLP进行归一化的标志
        self.mlp_norm = mlp_norm

        # 定义第一个归一化层，如果不使用注意力机制则为恒等映射
        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        # 定义注意力模块
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        # 定义第二个归一化层，如果使用注意力机制且需要对MLP进行归一化则为归一化层，否则为恒等映射
        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        # 定义多层感知机模块
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))
        """
        加入小波变换残差
        """
        # self.wavelet = HeightWidthDiagonalFeatureProcessor(input_channel_count=dim, output_channel_count=dim)

        self.hwconv = HeightWidthFeatureDepthwiseConv(dim) 
    def forward(self, x):
        # 保存输入的副本
        identity = x
        # print("x:",x.shape)
        # wavelet = self.wavelet(x)
        # print("wavalet:",wavelet.shape)
        hwconv = self.hwconv(x)
        # print("X:",x.shape)
        # print("hwconv:",hwconv.shape)
        if self.use_attn:
            # 如果使用注意力机制，对输入进行归一化
            x, rescale, rebias = self.norm1(x)
        # 计算注意力输出
        x = self.attn(x)
        if self.use_attn:
            # 如果使用注意力机制，对注意力输出进行缩放和平移
            x = x * rescale + rebias
        # 残差连接
        x = identity + x + hwconv

        # 保存输入的副本
        identity = x
        if self.use_attn and self.mlp_norm:
            # 如果使用注意力机制且需要对MLP进行归一化，对输入进行归一化
            x, rescale, rebias = self.norm2(x)
        # 计算多层感知机输出
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm:
            # 如果使用注意力机制且需要对MLP进行归一化，对多层感知机输出进行缩放和平移
            x = x * rescale + rebias
        # 残差连接
        x = identity + x
        return x

# 定义基础层类
class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        # 特征维度
        self.dim = dim
        # 块的数量
        self.depth = depth

        # 计算使用注意力机制的块的数量
        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            # 如果注意力机制在最后部分使用，生成相应的标志列表
            use_attns = [i >= depth - attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            # 如果注意力机制在最开始部分使用，生成相应的标志列表
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            # 如果注意力机制在中间部分使用，生成相应的标志列表
            use_attns = [i >= (depth - attn_depth) // 2 and i < (depth + attn_depth) // 2 for i in range(depth)]
        # print("use_attns:",use_attns)
        # 构建Transformer块列表
        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        # 依次通过每个Transformer块
        for blk in self.blocks:
            x = blk(x)
        return x

# 定义补丁嵌入类，用于将图像划分为补丁并嵌入到特征空间
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        # 输入通道数
        self.in_chans = in_chans
        # 嵌入维度
        self.embed_dim = embed_dim

        if kernel_size is None:
            # 如果未指定卷积核大小，使用补丁大小作为卷积核大小
            kernel_size = patch_size

        # 定义卷积层，用于补丁嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        # 前向传播，将输入通过卷积层进行补丁嵌入
        x = self.proj(x)
        return x

# 定义补丁反嵌入类，用于将特征空间的特征反嵌入为图像

"""
nn.Conv2d：将嵌入特征映射到更高维度的空间
输入通道：embed_dim
输出通道：out_chans * patch_size ** 2（如3×4²=48）
使用反射填充(padding_mode='reflect')保持空间尺寸
nn.PixelShuffle：通过像素重排操作将特征图上采样
将通道维度重排为空间维度，实现上采样
输入：形状为 (B, embed_dim, H, W) 的特征图
经过卷积后：形状变为 (B, out_chans*patch_size², H, W)
经过PixelShuffle后：形状变为 (B, out_chans, H*patch_size, W*patch_size)
最终输出尺寸是输入尺寸的 patch_size 倍
"""
class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        # 输出通道数
        self.out_chans = out_chans
        # 嵌入维度
        self.embed_dim = embed_dim

        if kernel_size is None:
            # 如果未指定卷积核大小，使用1作为卷积核大小
            kernel_size = 1

        # 定义反嵌入的网络结构
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        # 前向传播，将输入通过反嵌入网络
        x = self.proj(x)
        return x

# 定义SK融合类，用于特征融合
class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        # 特征融合的分支数量
        self.height = height
        # 计算中间层的维度
        d = max(int(dim / reduction), 4)

        # 定义自适应平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义元信息处理的网络结构
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        # 定义softmax激活函数
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        # 拼接输入特征
        in_feats = torch.cat(in_feats, dim=1)
        # 调整输入特征的维度
        in_feats = in_feats.view(B, self.height, C, H, W)

        # 计算特征的总和
        feats_sum = torch.sum(in_feats, dim=1)
        # 计算注意力分数
        attn = self.mlp(self.avg_pool(feats_sum))
        # 对注意力分数进行softmax操作
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        # 根据注意力分数融合特征
        out = torch.sum(in_feats * attn, dim=1)
        return out

# 以下是被注释掉的另一个SKFusion类的定义
# class SKFusion(nn.Module):
#     def __init__(self, dim, height = 2, reduction=8):
#         super(SKFusion, self).__init__()
#
#         self.height = height
#         self.dim = dim
#         d = max(int(dim/reduction), 4)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, d, 1, bias=False),
#             nn.ReLU(True),
#             #nn.LeakyReLU(0.1,inplace=True),
#             nn.Conv2d(d, dim*height, 1, bias=False)
#         )
#
#         self.softmax = nn.Softmax(dim=1)
#         self.conv = nn.Conv2d(dim, dim*height, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, in_feats):
#         B, C, H, W = in_feats[0].shape
#         x = in_feats[1]
#
#         in_feats = torch.cat(in_feats, dim=1) #[b,2c,h,w]
#         in_feats = in_feats.view(B, self.height, C, H, W)
#
#         feats_sum = torch.sum(in_feats, dim=1)
#         attn = self.mlp(self.avg_pool(feats_sum))
#         attn = self.softmax(attn.view(B, self.height, C, 1, 1))
#
#         feat = torch.sum(in_feats*attn, dim=1)
#
#         #####是否将第一个当作变量B，K
#         feat = self.conv(feat)
#         K, B = torch.split(feat, (self.dim, self.dim), dim=1)
#         out = K * x + B + x
#
#         return out

# 定义位置编码类
class PE(nn.Module):
    def __init__(self):
        super(PE, self).__init__()
        # 对网络中的模块进行权重初始化
        self.apply(self._init_weights)
        # 定义第一个多层感知机模块
        self.mlp1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(6, 9, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(3, 9, 1)
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=True)

        # 定义第二个多层感知机模块
        self.mlp2 = nn.Sequential(
            # nn.BatchNorm2d(num_features=32),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(9, 12, kernel_size=1),
            nn.Conv2d(12, 18, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(18, 24, kernel_size=1, bias=True),
            # nn.Conv2d(8, 3, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))
        # 以下是被注释掉的归一化层定义
        # self.bn1 = nn.BatchNorm2d(3)
        # self.bn2 = nn.BatchNorm2d(24)
        # 记录网络深度
        self.network_depth = 2
        # 对网络中的模块进行权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # 计算增益系数
            gain = (8 * self.network_depth) ** (-1 / 4)
            # 计算输入和输出的扇入和扇出
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            # 计算标准差
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            # 初始化权重为截断正态分布
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                # 初始化偏置为0
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存输入的副本
        x_c = x
        # 输入通过第一个多层感知机模块
        x = self.mlp1(x)  # + self.conv1(x_c)
        # 以下是被注释掉的归一化操作
        # x = self.bn2(x)
        # 输入通过第二个多层感知机模块并加上第二个卷积层的输出
        x = self.mlp2(x) + self.conv2(x_c)
        # print("经过PE后的尺寸",x.shape)
        return x

# 以下是被注释掉的PE48类的定义
# class PE48(nn.Module):
#     def __init__(self):
#         super(PE48, self).__init__()
#         self.apply(self._init_weights)
#         self.mlp1 = nn.Sequential(
#             nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.Conv2d(9, 16, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
# 
#         #self.conv1 = nn.Conv2d(3, 9, 1)
#         self.conv2 = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=True)
# 
#         self.mlp2 = nn.Sequential(
#             # nn.BatchNorm2d(num_features=32),
#             # nn.LeakyReLU(0.2,inplace=True),
#             nn.Conv2d(16, 24, kernel_size=1),
#             nn.Conv2d(24, 36, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.Conv2d(36, 48, kernel_size=1, bias=True),
#             # nn.Conv2d(8, 3, 1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True))
#         # self.bn1 = nn.BatchNorm2d(3)
#         # self.bn2 = nn.BatchNorm2d(24)
#         self.network_depth = 2
#         self.apply(self._init_weights)
# 
#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             gain = (8 * self.network_depth) ** (-1 / 4)
#             fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
#             std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
#             trunc_normal_(m.weight, std=std)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
# 
#     def forward(self, x):
#         x_c = x
#         x = self.mlp1(x)  # + self.conv1(x_c)
#         # x = self.bn2(x)
#         x = self.mlp2(x) + self.conv2(x_c)
#         return x

# 定义SPMFormer类，为主模型类
class SPMFormer(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(SPMFormer, self).__init__()
        
        # 设置补丁大小
        self.patch_size = 4
        # 设置窗口大小
        self.window_size = window_size
        # 记录多层感知机的比例
        self.mlp_ratios = mlp_ratios

        # 定义补丁嵌入模块，将图像划分为补丁并嵌入到特征空间
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # 定义第一个基础层
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        # 定义第一个补丁合并模块
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        # 定义第一个跳跃连接的卷积层
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        # 定义第二个基础层
        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], window_size=window_size,
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])

        # 定义第二个补丁合并模块
        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        # 定义第二个跳跃连接的卷积层
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        # 定义第三个基础层
        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2], window_size=window_size,
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])

        # 定义第一个补丁拆分模块
        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        # 确保特征维度一致
        assert embed_dims[1] == embed_dims[3]
        # 定义第一个特征融合模块
        self.fusion1 = SKFusion(embed_dims[3])

        # 定义第四个基础层
        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], window_size=window_size,
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        # 定义第二个补丁拆分模块
        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=24, embed_dim=embed_dims[3])

        # 以下是被注释掉的维度检查
        #assert embed_dims[0] == embed_dims[4]
        # 定义第二个特征融合模块
        self.fusion2 = SKFusion(24)

        # 以下是被注释掉的第五个基础层定义
        # self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
        #                          num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
        #                          norm_layer=norm_layer[4], window_size=window_size,
        #                          attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        # 定义补丁反嵌入模块，将特征空间的特征反嵌入为图像
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=24, kernel_size=3)

        # 定义位置编码模块
        self.pe = PE()
        # 以下是被注释掉的其他位置编码模块定义
        # self.pe2 = PE()
        # self.pe3 = PE()
        # 定义位置编码反嵌入模块
        self.unpe = PatchUnEmbed(
            patch_size=1, out_chans=6, embed_dim=24, kernel_size=3)
        # 以下是被注释掉的其他位置编码反嵌入模块定义
        # self.unpe2 = PatchUnEmbed(
        #     patch_size=1, out_chans=2, embed_dim=24, kernel_size=3)
        # self.unpe3 = PatchUnEmbed(
        #     patch_size=1, out_chans=2, embed_dim=24, kernel_size=3)

    def check_image_size(self, x):
        # 此函数用于检查图像大小并进行填充，确保图像尺寸能被补丁大小整除
        # NOTE: for I2I test
        _, _, h, w = x.size()
        # 计算需要填充的高度
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        # 计算需要填充的宽度
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        # 对图像进行填充
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        # 此函数用于计算模型的特征输出
        #x = x * x_mask.float()
        # 保存输入的副本
        xc = x
        
        # 计算位置编码
        pe = self.pe(x)
        # 以下是被注释掉的其他位置编码计算
        # g_pe = self.pe1(x)
        # r_pe = self.pe1(x)

        #
        #pe = torch.cat((b_pe,g_pe,r_pe), dim=1)
        # 对位置编码进行反嵌入
        paras = self.unpe(pe)
        # print("paras的尺寸",paras.shape)
        # 分离参数K和B
        K, B = torch.split(paras, (3,3), dim = 1)
        # 对输入进行缩放和平移
        xc = K * xc + B
        # 对输入进行补丁嵌入
        x = self.patch_embed(xc)
        #x = K * xc + B  # 卷积

        # 通过第一个基础层
        x = self.layer1(x)
        # 保存第一个基础层的输出作为跳跃连接
        skip1 = x

        # 对第一个基础层的输出进行补丁合并
        x = self.patch_merge1(x)
        # 通过第二个基础层
        x = self.layer2(x)
        # 保存第二个基础层的输出作为跳跃连接
        skip2 = x

        # 对第二个基础层的输出进行补丁合并
        x = self.patch_merge2(x)
        # 通过第三个基础层
        x = self.layer3(x)
        # 对第三个基础层的输出进行补丁拆分
        x = self.patch_split1(x)

        # 进行特征融合
        x = self.fusion1([x, self.skip2(skip2)]) + x
        # 通过第四个基础层
        x = self.layer4(x)
        # 对第四个基础层的输出进行补丁拆分
        x = self.patch_split2(x)

        # 进行特征融合
        x = self.fusion2([x, self.skip1(skip1)]) + x
        # 以下是被注释掉的通过第五个基础层
        # x = self.layer5(x)
        # 对特征进行补丁反嵌入
        x = self.patch_unembed(x)
        return x, xc

    def forward(self, x):
        # 保存输入图像的高度和宽度
        H, W = x.shape[2:]
        # 检查图像大小并进行填充
        x = self.check_image_size(x)  # b,g,r
        # 以下是被注释掉的掩码相关操作
        #x_mask = torch.ones_like(x)
        #x_mask[:, 2, :, :] = 2.0
        # x_mask = self.check_image_size(x_mask)

        # 计算模型的特征输出
        feat, xc = self.forward_features(x)
        # 分离特征中的K和B
        K, B = torch.split(feat, (1, 3), dim=1)

        # 对输入进行缩放和平移
        x = K * xc + B + xc
        # print("K,B,xc的尺寸",K.shape,B.shape,xc.shape)
        # 裁剪输出到原始图像大小
        x = x[:, :, :H, :W]
        return x
# 定义一个小版本的SPMFormer模型
def spmformer_t():
    return SPMFormer(
        embed_dims=[24, 24, 24, 24],
        mlp_ratios=[2., 2., 2., 2.],
        depths=[2, 2, 2, 2],
        num_heads=[2, 2, 2, 1],
        attn_ratio=[0, 1 / 2, 1, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv'])

# 定义一个大版本的SPMFormer模型
def spmformer_b():
    return SPMFormer(
        embed_dims=[24, 48, 96, 48],
        mlp_ratios=[2., 4., 4., 2.],
        depths=[8, 8, 8, 8],
        num_heads=[2, 4, 6, 2],
        attn_ratio=[1 / 4, 1 / 2, 3 / 4, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv'])

if __name__ == '__main__':
    model = spmformer_b()
    input = torch.ones((1,3,256,256))
    torch.onnx.export(model, input, f='SPM.onnx')
    # print(model)cle