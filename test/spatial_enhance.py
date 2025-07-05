import torch
import torch.nn as nn

"""
    简单实现特征图的自适应对比度增强：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1jGKqzaExJ/
        作用位置：任何单一输出特征后，或者任何即插即用模块中。
        主要功能：对输入特征图进行自适应对比度增强，通过计算每个通道方差调整特征响应，增强局部细节同时保留全局信息，解决图像对比度低、细节模糊特点。
                1、对噪声主导的低对比度区域进行抑制
                2、对包含重要信息的高对比度区域进行保留
        代码使用方式与写作思路请务必看视频~  
"""

class SpatialVarianceModulation(nn.Module):
    def __init__(self, eps: float = 1e-4):
        """
        :param eps: 数值稳定项，防止除零错误（默认值：1e-4）
        """
        super(SpatialVarianceModulation, self).__init__()
        self.activation = nn.Sigmoid()  # 激活函数：将权重映射压缩到(0,1)范围
        self.eps = eps  # 数值稳定性常数，避免方差为零的情况

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature_map.size()
        #===== 该模块通过计算输入特征图的局部方差来动态调整每个位置的响应强度 =====
        # 计算空间维度（H×W）上的均值
        spatial_mean = feature_map.mean(dim=[2, 3], keepdim=True)
        # 计算特征值与均值的偏差（中心化处理）【绝对距离 ====> 相对距离】，然后求平方
        squared_deviation = (feature_map - spatial_mean).pow(2)
        # ===== 自适应权重生成 =====
        # 计算空间方差的无偏估计：Σ(偏差²)/(H×W-1)
        spatial_variance = squared_deviation.sum(dim=[2, 3], keepdim=True) / (height * width - 1)
        # 核心调制公式：对权重进行Sigmoid激活，权重映射到(0,1)区间
        # 数学形式：权重 = (偏差²)/(4*方差) + 0.5
        modulation_coeff = squared_deviation / (4 * (spatial_variance + self.eps)) + 0.5
        weight = self.activation(modulation_coeff)

        # ===== 特征调制 =====
        # 原始特征与权重图逐元素相乘
        # weight ≈ 0.5 ，小幅增强
        # weight ≈ 1 ，保留原始值
        # 效果：增强高方差区域（边缘/纹理），抑制低方差区域（平滑背景）
        modulated_feature = feature_map * weight
        return modulated_feature

# if __name__ == "__main__":
#     x = torch.randn(1, 32, 50, 50)
#     modulator = SpatialVarianceModulation()
#     output = modulator(x)
#     print(f'输入特征尺寸: {x.size()}')
#     print(f'输出特征尺寸: {output.size()}')
#     print("微信公众号、B站、CSDN同号")
#     print("布尔大学士 提醒您：微创新·代码无误")