from traceback import print_tb
from torchvision import transforms
import torch
import torch_dct as dct
import cv2
import numpy as np
from PIL import Image

img_size = 256
img_path = "/home/benchunlei/dl/SPMFormer/test/643_img_.png"

img = cv2.imread(img_path)
# print(img.shape)
img = cv2.resize(img, (img_size, img_size))
# 移除转换为灰度图的步骤，保留三个通道
transf = transforms.ToTensor()
img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)


 


dct2d= dct.dct_2d(img_tensor, norm='ortho')
print(dct2d.shape)  

# 定义一个尺寸为256*256的掩码矩阵
mask = torch.ones((img_size, img_size))
# 将左上方6个元素设置为0
mask[:50, :51] = 0
# 对x进行掩码
dct2d = dct2d * mask

x = dct.idct_2d(dct2d, norm='ortho')

# 将结果转换为numpy数组并转换为合适的数据类型
# 修改为使用逆DCT变换后的结果x
x_np = x.detach().permute(1, 2, 0).numpy()  # 调整维度为 (H, W, C)
x_np = np.clip(x_np, 0, 1)  # 裁剪数值范围到 0-1
x_np = (x_np * 255).astype('uint8')  # 转换为 0-255 的整数

cv2.imwrite('reconstructed_image.png', x_np)



