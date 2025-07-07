import torch
import torch.nn as nn
from torch.nn import init

"""
    给特征增加位置水平、垂直编码信息：（位置特征、位置编码、通过位置信息引导特征学习
        写作思路与代码讲解：https://www.bilibili.com/video/BV1nRLczbEfm/
        作用位置：在特征输入之前加入。
        主要功能：在深度学习中，位置编码为模型提供特征元素位置的信息，有助于模型更好地理解和处理图像数据。
"""

class PositionEncodingModule(nn.Module):
    def __init__(self, channels, direction, window_size):
        super().__init__()
        self.direction = direction  # 操作方向，'H'表示水平方向，'W'表示垂直方向
        self.channels = channels  # 输入特征通道数
        self.window_size = window_size  # 全局窗口大小

        # 初始化位置编码参数
        if self.direction == 'H':  # 水平方向位置编码
            self.pos_encoding = nn.Parameter(torch.randn(1, channels, window_size, 1))
        elif self.direction == 'W':  # 垂直方向位置编码
            self.pos_encoding = nn.Parameter(torch.randn(1, channels, 1, window_size))

        # 使用截断正态分布初始化参数 30 === 90+  ===85
        init.trunc_normal_(self.pos_encoding, std=0.02)

    def forward(self, feature_map):
        # 扩展位置编码维度以匹配输入特征尺寸
        pos_enc_expanded = self.pos_encoding.expand(1, self.channels, self.window_size, self.window_size)
        # 将位置编码添加到特征图上
        return feature_map + pos_enc_expanded

# if __name__ == '__main__':
#     pos_enc_module = PositionEncodingModule(channels=32, direction='W', window_size=50)
#     input_tensor = torch.rand(1, 32, 50, 50)
#     output_tensor = pos_enc_module(input_tensor)
#     print(f'Input shape: {input_tensor.size()}')
#     print(f'Output shape: {output_tensor.size()}')
#     print("微信公众号、B站、CSDN同号")
#     print("布尔大学士 提醒您：微创新·代码无误")