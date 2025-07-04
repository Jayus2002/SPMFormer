# 导入操作系统相关功能模块，用于文件和目录操作
import os
# 导入命令行参数解析模块，方便用户通过命令行传递参数
import argparse
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入PyTorch的神经网络函数模块
import torch.nn.functional as F
# 从pytorch_msssim库中导入结构相似性指数（SSIM）计算函数
from pytorch_msssim import ssim
# 从torch.utils.data模块中导入数据加载器类，用于批量加载数据
from torch.utils.data import DataLoader
# 从collections模块中导入有序字典类
from collections import OrderedDict

# 从utils模块中导入平均指标计算类、图像写入函数和通道顺序转换函数
from myutils import AverageMeter, write_img, chw_to_hwc
# 从datasets.loader模块中导入成对数据加载器类
from datasets.loader import PairLoader
# 导入models模块中的所有内容
from models import *
# 导入NumPy科学计算库
import numpy as np
# 导入OpenCV计算机视觉库
import cv2

# import swanlab



# swanlab.init(
#     # 设置项目名
#     project="SPMFormer",
#     # 设置实验名
#     experiment=args.exp,
#     config=setting,
# )  


# 注释掉的导入语句，用于从skimage.metrics模块中导入峰值信噪比计算函数
# from skimage.metrics import peak_signal_noise_ratio
# 注释掉的导入语句，用于从skimage.metrics模块中导入结构相似性计算函数
# from skimage.metrics import structural_similarity

# 创建命令行参数解析器对象
parser = argparse.ArgumentParser()
# 添加模型名称参数，默认值为'spmformer-b'
parser.add_argument('--model', default='spmformer-b', type=str, help='模型名称')
# 添加数据加载器的工作线程数参数，默认值为4
parser.add_argument('--num_workers', default=4, type=int, help='数据加载器的工作线程数')
# 添加数据集路径参数，默认值为'./data/'
parser.add_argument('--data_dir', default='./dataset/', type=str, help='数据集路径')
# 添加模型保存路径参数，默认值为'./saved_models/'
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='模型保存路径')
# 添加结果保存路径参数，默认值为'./results/'
parser.add_argument('--result_dir', default='./results/', type=str, help='结果保存路径')
# 添加数据集名称参数，默认值为'LSUI'
parser.add_argument('--dataset', default='UIEB', type=str, help='数据集名称')
# 添加实验设置参数，默认值为'lsui'
parser.add_argument('--exp', default='uieb', type=str, help='实验设置')
# 解析命令行参数
args = parser.parse_args()

# 定义函数，用于处理模型的状态字典，去除键名中的前缀
def single(save_dir):
    # 加载保存的模型状态字典
    state_dict = torch.load(save_dir)['state_dict']
    # 创建一个有序字典，用于存储处理后的状态字典
    new_state_dict = OrderedDict()

    # 遍历原始状态字典的键值对
    for k, v in state_dict.items():
        # 去除键名的前7个字符
        name = k[7:]
        # 将处理后的键值对添加到新的有序字典中
        new_state_dict[name] = v

    return new_state_dict

# 定义函数，用于计算均方根误差（RMSE）
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# 定义测试函数，用于评估模型在测试集上的性能
def test(test_loader, network, result_dir):
    # 创建平均指标计算对象，用于记录PSNR指标
    PSNR = AverageMeter()
    # 创建平均指标计算对象，用于记录SSIM指标
    SSIM = AverageMeter()

    # 清空CUDA缓存，释放显存
    torch.cuda.empty_cache()

    # 将网络设置为评估模式，关闭一些在训练时使用的特殊层，如Dropout
    network.eval()

    # 创建保存图像的目录，如果目录已存在则不会报错
    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    # 打开结果保存文件，用于记录每个样本的评估结果
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    # 遍历测试数据加载器中的每个批次
    for idx, batch in enumerate(test_loader):
        # 将输入数据移动到CUDA设备上
        input = batch['source'].cuda()
        # 将目标数据移动到CUDA设备上
        target = batch['target'].cuda()

        # 获取当前批次中第一个样本的文件名
        filename = batch['filename'][0]

        # 关闭梯度计算，提高推理速度并节省显存
        with torch.no_grad():
            # 前向传播，得到模型的输出，并将输出限制在[-1, 1]范围内
            output = network(input).clamp_(-1, 1)

            # 将输出和目标数据从[-1, 1]范围转换到[0, 1]范围
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            # 计算峰值信噪比（PSNR）
            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            # 获取输出的高度和宽度
            _, _, H, W = output.size()
            # 计算下采样比例，参考Zhou Wang的方法
            down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
            # 计算结构相似性指数（SSIM）
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()
        # 更新PSNR的平均指标
        PSNR.update(psnr_val)
        # 更新SSIM的平均指标
        SSIM.update(ssim_val)

        # 将当前样本的文件名、PSNR和SSIM结果写入结果文件
        f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))

        # 将输出数据从张量转换为图像格式，并调整通道顺序
        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        # 将目标数据从张量转换为图像格式，并调整通道顺序
        tar_img = chw_to_hwc(target.detach().cpu().squeeze(0).numpy())
        # 将输出图像保存到指定目录
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)
        # 打印当前批次的评估结果和平均评估结果
        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})\t'
              .format(idx, psnr=PSNR, ssim=SSIM))

    # 关闭结果保存文件
    f_result.close()

    # 将结果文件重命名，文件名包含平均PSNR和平均SSIM
    os.rename(os.path.join(result_dir, 'results.csv'),
              os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))

# 程序入口
if __name__ == '__main__':
    # 根据命令行参数动态创建模型对象
    network = eval(args.model.replace('-', '_'))()
    # 将模型移动到CUDA设备上
    network.cuda()
    # 拼接保存的模型文件路径
    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

    # 检查保存的模型文件是否存在
    if os.path.exists(saved_model_dir):
        # 打印开始测试的信息，包含当前使用的模型名称
        print('==> Start testing, current model name: ' + args.model)
        # 加载保存的模型状态字典
        network.load_state_dict(single(saved_model_dir))
    else:
        # 打印没有找到训练好的模型的信息
        print('==> No existing trained model!')
        # 退出程序
        exit(0)

    # 拼接测试数据集的路径
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    print("dataset_dir:" + dataset_dir)
    # 创建测试数据集对象
    test_dataset = PairLoader(dataset_dir, '', 'test','test.txt')
    
    # 创建测试数据加载器对象
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    # 拼接结果保存目录的路径
    result_dir = os.path.join(args.result_dir, args.dataset, args.model)
    # 调用测试函数进行模型评估
    test(test_loader, network, result_dir)