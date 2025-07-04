import torch
import torch.nn as nn

"""
    通过深度分离卷积提取条形（水平、垂直）、方形特征
        写作思路与代码讲解：https://www.bilibili.com/video/BV1FddyYzEW6/
        作用位置：在特征输入之前加入，或者残差。
        主要功能：①借助深度分离卷积把输入特征分解成低频分量与水平、垂直、方形的特征。 
                ②通过方形卷积核、水平条形卷积核、垂直条形卷积核捕捉输入特征图的空间局部特征、水平长距离依赖和垂直长距离依赖。
"""

class HeightWidthFeatureDepthwiseConv(nn.Module):
    def __init__(self, group_channels, square_kernel=3, band_kernel=11):
        super().__init__()

        """
            kernel_size : 指定了卷积核（或滤波器）的大小。它决定了每次卷积操作时覆盖输入张量的区域大小。
                          如果是一个整数，则表示方形卷积核；
                          如果是元组形式如(height, width)，则分别指定高度和宽度方向上的卷积核大小。
            padding    : 在输入张量边缘周围添加额外的零值，以控制输出张量的空间维度。它可以避免由于卷积操作导致的边界信息丢失。
                          如果是一个整数，则在每个边界的四周均匀地添加相同数量的零；
                          对于二维卷积，常见的简化形式为(height_padding, width_padding)。
        """
        # 深度可分离卷积层，处理方形区域，使用group_channels个分组进行卷积
        self.square_depthwise_conv = nn.Conv2d(group_channels, group_channels, kernel_size=square_kernel,
                                               padding=square_kernel // 2, groups=group_channels)

        # 深度可分离卷积层，处理水平 W 方向上的条形区域，使用group_channels个分组进行卷积
        self.horizontal_band_depthwise_conv = nn.Conv2d(group_channels, group_channels,
                                                        kernel_size=(1, band_kernel),
                                                        padding=(0, band_kernel // 2), groups=group_channels)
        # 深度可分离卷积层，处理竖直 H 方向上的条形区域，使用group_channels个分组进行卷积
        self.vertical_band_depthwise_conv = nn.Conv2d(group_channels, group_channels,
                                                      kernel_size=(band_kernel, 1),
                                                      padding=(band_kernel // 2, 0), groups=group_channels)

        # 定义一个Sequential容器，按顺序包含卷积层、批量归一化层和激活函数
        self.convolution_batchnorm_activation = nn.Sequential(
            # 添加2D卷积层，输入通道数是原通道数的4倍（因为经过DWT后会产生4个子带），输出通道数为group_channels
            nn.Conv2d(group_channels * 4, group_channels, kernel_size=1, stride=1),
            # 批量归一化层，对group_channels个特征图进行归一化
            nn.BatchNorm2d(group_channels),
            # ReLU激活函数，inplace=True表示直接在输入数据上进行修改以节省内存
            nn.ReLU(inplace=True),
        )

    # 定义前向传播方法，接收输入张量作为参数
    def forward(self, input_tensor):
        # 保存原始输入张量
        original_input = input_tensor
        # 对输入张量进行方形卷积操作
        square_convolved = self.square_depthwise_conv(input_tensor)
        # 对输入张量进行水平条形卷积操作
        horizontal_convolved = self.horizontal_band_depthwise_conv(input_tensor)
        # 对输入张量进行垂直条形卷积操作
        vertical_convolved = self.vertical_band_depthwise_conv(input_tensor)

        # 沿着通道维度（dim=1）将原始输入、方形卷积结果、水平条形卷积结果和垂直条形卷积结果拼接起来
        merged_features = torch.cat((original_input, square_convolved, horizontal_convolved, vertical_convolved), dim=1)

        # 维度还原①：可参考 000代码 https://www.bilibili.com/video/BV1vfZnYTEE2/
        output_tensor = self.convolution_batchnorm_activation(merged_features)

        return output_tensor

# if __name__ == '__main__':
#     feature_conv_model = HeightWidthFeatureDepthwiseConv(32)
#     input_data = torch.randn(1, 32, 50, 50)
#     output_data = feature_conv_model(input_data)
#     print(f'Input size: {input_data.size()}')
#     print(f'Output size: {output_data.size()}')
#     print("微信公众号、B站、CSDN同号")
#     print("布尔大学士 提醒您：微创新·代码无误")