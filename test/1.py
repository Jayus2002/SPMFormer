import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class HighFrequencyGenerator(nn.Module):
    def __init__(self, in_channels: int, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
        self.dct_matrix = self._get_dct_matrix(block_size).to(torch.float32)
        
        # 高通滤波器参数（可学习）
        self.hpf = nn.Parameter(torch.ones(block_size, block_size) * 0.1)
        nn.init.dirac_(self.hpf)  # 初始化为中心高频增强
        
        # 双路径处理
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _get_dct_matrix(self, N: float) -> Tensor:
        """生成DCT变换矩阵"""
        i, j = torch.meshgrid(torch.arange(N), torch.arange(N))
        alpha = torch.where(i == 0, 1.0, 2.0).sqrt() / N.sqrt()
        return alpha * torch.cos((2 * j + 1) * i * torch.pi / (2 * N))

    def _block_dct(self, x: Tensor) -> Tensor:
        """分块DCT变换"""
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        x = F.unfold(x, kernel_size=self.block_size, stride=self.block_size)
        x = x.transpose(1, 2)  # [B*C, num_blocks, block_size**2]
        
        # 应用DCT变换
        x = torch.matmul(x, self.dct_matrix.t().view(-1))
        x = x.view(B, C, -1, self.block_size, self.block_size)
        
        # 高通滤波
        x = x * self.hpf.unsqueeze(0).unsqueeze(0)
        return x

    def _block_idct(self, x: Tensor, original_size) -> Tensor:
        """分块逆DCT变换"""
        B, C, _, H, W = x.shape
        x = torch.matmul(x.view(-1, self.block_size**2), self.dct_matrix)
        x = x.view(B, C, -1, self.block_size**2)
        x = x.transpose(1, 2)
        x = F.fold(x, output_size=original_size[-2:], 
                  kernel_size=self.block_size, stride=self.block_size)
        return x.view(B, C, *original_size[-2:])

    def forward(self, x: Tensor) -> Tensor:
        # 频域处理路径
        original_size = x.shape
        freq = self._block_dct(x)
        hf_feat = self._block_idct(freq, original_size)
        
        # 通道权重路径
        gap = self.gap(hf_feat).squeeze(-1).squeeze(-1)
        gmp = self.gmp(hf_feat).squeeze(-1).squeeze(-1)
        channel_weight = self.channel_fc(gap + gmp).unsqueeze(-1).unsqueeze(-1)
        
        # 空间注意力路径
        spatial_weight = self.spatial_conv(hf_feat)
        
        # 特征融合
        return x * channel_weight * spatial_weight

if __name__ == '__main__':
    model = HighFrequencyGenerator(in_channels=64)
    x = torch.randn(2, 64, 128, 128)  # 示例输入
    output = model(x)  # 输出高频增强特征
    print("output:", output.shape)
