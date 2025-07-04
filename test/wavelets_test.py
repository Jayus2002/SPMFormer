import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# pip install pytorch_wavelets==1.3.0
# pip install PyWavelets

"""
    通过小波变换提取水平、垂直、对角线的特征（多频域特征融合）
        写作思路与代码讲解：https://www.bilibili.com/video/BV1cPR1YgEBo/
        作用位置：在特征输入之前加入，或者残差。
        主要功能：①借助离散小波变换（DWT）把输入特征分解成低频分量与水平、垂直、对角线方向的高频分量。
                ②通过解耦水平/垂直/对角特征，克服传统卷积核各方向响应的特征局限性。
        代码使用方式与写作思路请务必看视频~
"""

class HeightWidthDiagonalFeatureProcessor(nn.Module):
    def __init__(self, input_channel_count, output_channel_count):
        super(HeightWidthDiagonalFeatureProcessor, self).__init__()

        # 定义离散小波变换(DWT)前向操作，参数J表示分解级别为1，mode设置边界处理方式为零填充，wave指定使用Haar小波
        self.discrete_wavelet_transform = DWTForward(J=1, mode='zero', wave='haar')

        # 定义卷积-批归一化-激活层序列
        self.convolution_batchnorm_activation = nn.Sequential(
            # 添加2D卷积层，输入通道数是原通道数的4倍（因为经过DWT后会产生4个子带），输出通道数为output_channel_count
            nn.Conv2d(input_channel_count * 4, output_channel_count, kernel_size=1, stride=1),
            # 批量归一化层，对output_channel_count个特征图进行归一化
            nn.BatchNorm2d(output_channel_count),
            # ReLU激活函数，inplace=True表示直接在输入数据上进行修改以节省内存
            nn.ReLU(inplace=True),
        )
        self.feature_fuser = nn.Linear(4 * input_channel_count, input_channel_count)  # 定义融合层，用于融合来自不同方向的信息

    def forward(self, input_tensor):
        # 得到低频分量low_frequency_component和高频分量high_frequency_components
        low_frequency_component, high_frequency_components = self.discrete_wavelet_transform(input_tensor)

        # 从高频分量·提取出水平细节系数
        horizontal_detail_coefficient = high_frequency_components[0][:, :, 0, :, :]
        # 从高频分量·提取出垂直细节系数
        vertical_detail_coefficient = high_frequency_components[0][:, :, 1, :, :]
        # 从高频分量·提取出对角线细节系数
        diagonal_detail_coefficient = high_frequency_components[0][:, :, 2, :, :]

        # 将低频分量与三个方向的高频分量沿着通道维度拼接
        merged_features = torch.cat([low_frequency_component,
                                     horizontal_detail_coefficient,
                                     vertical_detail_coefficient,
                                     diagonal_detail_coefficient], dim=1)

        # 维度还原①：可参考 000代码 https://www.bilibili.com/video/BV1vfZnYTEE2/
        output_tensor = self.convolution_batchnorm_activation(merged_features)
        return output_tensor

# if __name__ == '__main__':
#     feature_processor = HeightWidthDiagonalFeatureProcessor(input_channel_count=32, output_channel_count=32)
#     input_tensor = torch.rand(1, 32, 256, 256)
#     output_tensor = feature_processor(input_tensor)
#     print(f'Input size: {input_tensor.size()}')
#     print(f'Output size: {output_tensor.size()}')
#     print("微信公众号、B站、CSDN同号")
#     print("布尔大学士 提醒您：微创新·代码无误")