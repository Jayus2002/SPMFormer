import re
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torchvision.models import ResNet50_Weights

class UnderwaterRestoration(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享特征提取
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
        # self.network_depth = 2
        # 对网络中的模块进行权重初始化
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Conv2d):
    #         # 计算增益系数
    #         gain = (8 * self.network_depth) ** (-1 / 4)
    #         # 计算输入和输出的扇入和扇出
    #         fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
    #         # 计算标准差
    #         std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    #         # 初始化权重为截断正态分布
    #         trunc_normal_(m.weight, std=std)
    #         if m.bias is not None:
    #             # 初始化偏置为0
    #             nn.init.constant_(m.bias, 0)

        # 透射率估计分支
        self.t_branch = nn.Sequential(
            nn.Conv2d(24, 128, 3, padding=1),
            nn.ReLU(),
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid()  # 输出范围[0,1]
        )
        
        # 环境光估计分支
        self.B_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局信息
            nn.Conv2d(24, 3, 1),    # 输出RGB值
            nn.Sigmoid()             # 限制范围
        )
        
    def forward(self, x):
        x_c = x
        x = self.mlp1(x)
        x = self.mlp2(x) + self.conv2(x_c)
        print(x.shape)
        t = self.t_branch(x)
        B = self.B_branch(x)
        return t, B

if __name__ == '__main__':
    model = UnderwaterRestoration()
    input_tensor = torch.rand(1, 3, 256, 256)
    t, B = model(input_tensor)
    # features = model(input_tensor)
    print(f'Input shape: {input_tensor.size()}')
    print(f'Output shape T: {t.size()}')
    print(f'Output shape B: {B.size()}')
    # print(features.shape)