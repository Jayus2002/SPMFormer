# 导入必要的模块
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity
from myutils import AverageMeter,chw_to_hwc
from datasets.loader import PairLoader, TPairLoader
from  datasets.loader import MixUp_AUG
from models import *
import cv2
import random
import numpy as np
from torchprofile import profile_macs
from skimage import color
from collections import OrderedDict
from ptcolor import rgb2lab
import swanlab



# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加模型名称参数
parser.add_argument('--model', default='spmformer-b', type=str, help='模型名称')
# 添加数据加载线程数参数
parser.add_argument('--num_workers', default=8, type=int, help='数据加载的线程数')
# 添加是否禁用自动混合精度训练的参数
parser.add_argument('--no_autocast', action='store_false', default=True, help='禁用自动混合精度训练')
# 添加模型保存路径参数
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='模型保存的路径')
# 添加数据集路径参数
parser.add_argument('--data_dir', default='./dataset/', type=str, help='数据集的路径')
# 添加日志保存路径参数
parser.add_argument('--log_dir', default='./logs/', type=str, help='日志保存的路径')
# 添加数据集名称参数
parser.add_argument('--dataset', default='UIEB', type=str, help='数据集名称')
# 添加实验设置参数
parser.add_argument('--exp', default='uieb', type=str, help='实验设置')
# 添加使用的GPU编号参数
parser.add_argument('--gpu', default='0', type=str, help='用于训练的GPU编号')
# 解析命令行参数
args = parser.parse_args()

# 构建配置文件路径
setting_filename = os.path.join('configs', args.exp, args.model+'.json')
# 如果配置文件不存在，使用默认配置文件
if not os.path.exists(setting_filename):
    setting_filename = os.path.join('configs', args.exp, 'default.json')
    # 打开配置文件并加载配置
with open(setting_filename, 'r') as f:
    setting = json.load(f)
# 初始化Swanlab

swanlab.init(
    # 设置项目名
    project="SPMFormer",
    # 设置实验名
    experiment=args.exp,
    config=setting,
)  


# 设置可见的GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# 定义傅里叶损失函数
def fourier_loss(input_image, target_image):
    # input_freq = torch.zeros_like(input_image, dtype=torch.complex64)
    # target_freq = torch.zeros_like(input_image, dtype=torch.complex64)
    # 对输入图像和目标图像进行二维傅里叶变换
    input_freq = torch.fft.fft2(input_image)
    # 计算输入图像傅里叶变换结果的幅度谱
    input_abs = torch.abs(input_freq)  # 幅度谱，求模得到
    # 计算输入图像傅里叶变换结果的相位谱
    input_ang = torch.angle(input_freq)  # 相位谱，求相角得到
    # 对目标图像进行二维傅里叶变换
    target_freq = torch.fft.fft2(target_image)
    # 计算目标图像傅里叶变换结果的幅度谱
    target_abs = torch.abs(target_freq)  # 幅度谱，求模得到
    # 计算目标图像傅里叶变换结果的相位谱
    target_ang = torch.angle(target_freq)  # 相位谱，求相角得到

    # 计算频域中幅度的平方差
    #freq_diff = torch.abs(input_freq) - torch.abs(target_freq)
    # 计算幅度谱的损失
    abs_loss = criterion(input_abs, target_abs)
    # 计算相位谱的损失
    ang_loss = criterion(input_ang, target_ang)
    # print("傅里叶变换损失:",abs_loss,ang_loss)
    return abs_loss + ang_loss

# 定义训练函数
def train(train_loader, network, criterion, optimizer, scaler, epoch):
    # 初始化损失平均计算器
    losses = AverageMeter()

    # 清空CUDA缓存
    torch.cuda.empty_cache()

    # 将网络设置为训练模式
    network.train()

    # 遍历训练数据加载器中的每个批次
    for batch in train_loader:
        # 获取源图像并移动到GPU上
        source_img = batch['source'].cuda()
        # 获取目标图像并移动到GPU上
        target_img = batch['target'].cuda()
        #mask_img = batch['mask'].cuda()

        # if epoch > 70:
        #    target_img, source_img, mask_img = MixUp_AUG().aug(target_img, source_img, mask_img)

        # 使用自动混合精度训练
        with autocast(args.no_autocast):
            # 前向传播，得到网络输出
            output = network(source_img)  # [batch, c, h, w] -> c-[b,g,r]
            # 初始化与输出相同形状的全1张量
            target_gr = torch.ones_like(output)  
            # 初始化与输出相同形状的全1张量
            tones = torch.ones_like(output)
            #ones[:,2,:,:] *= 3
            # 找到目标图像中不为零的位置
            target_non_zero_indices = target_img != 0  # 找到vector2中不为零的位置
            # 计算目标图像不为零位置的比例
            target_gr[target_non_zero_indices] = (output[target_non_zero_indices] / target_img[target_non_zero_indices])

            # 将输出的RGB通道顺序调整并进行归一化
            rgb_output = output[:, [2, 1, 0], :, :] * 0.5 + 0.5
            # 将目标图像的RGB通道顺序调整并进行归一化
            rgb_target = target_img[:, [2, 1, 0], :, :] * 0.5 + 0.5
            # lab_output = rgb2lab(rgb_output)
            # lab_target = rgb2lab(rgb_target)
            # 将RGB图像转换为Lab颜色空间并进行裁剪
            lab_output = torch.clamp(rgb2lab(rgb_output), -80.0, 80.0)
            # 将RGB目标图像转换为Lab颜色空间并进行裁剪
            lab_target = torch.clamp(rgb2lab(rgb_target), -80.0, 80.0)
            # 初始化与Lab输出相同形状的全1张量
            labones = torch.ones_like(lab_output)
            # 初始化与Lab输出相同形状的张量用于存储比例
            lab_ratio = torch.ones_like(lab_output)
            # 找到Lab目标图像中不为零的位置
            target_non_zero_indices1 = lab_target != 0  # 找到vector2中不为零的位置
            # 计算Lab目标图像不为零位置的比例
            lab_ratio[target_non_zero_indices1] = (lab_output[target_non_zero_indices1] / lab_target[target_non_zero_indices1])

            # 计算总损失
            loss = criterion(output, target_img) + 0.01 * criterion(tones, target_gr) + 0.01 * criterion(lab_output, lab_target) + 0.01 * fourier_loss(output, target_img)
        # 更新损失平均计算器
        losses.update(loss.item())

        # 清空优化器的梯度
        optimizer.zero_grad()
        # 缩放损失并进行反向传播
        scaler.scale(loss).backward()
        # 优化器更新参数
        scaler.step(optimizer)
        # 更新缩放器
        scaler.update()

    return losses.avg

# 定义验证函数
def valid(val_loader, network):
    # 初始化PSNR平均计算器
    PSNR = AverageMeter()

    # 清空CUDA缓存
    torch.cuda.empty_cache()

    # 将网络设置为评估模式
    network.eval()

    # 遍历验证数据加载器中的每个批次
    for batch in val_loader:
        # 获取源图像并移动到GPU上
        source_img = batch['source'].cuda()
        # 获取目标图像并移动到GPU上
        target_img = batch['target'].cuda()
        #mask_img = batch['mask'].cuda()

        # 不计算梯度
        with torch.no_grad():							# torch.no_grad() may cause warning
            # 前向传播，得到网络输出并进行裁剪
            output = network(source_img).clamp_(-1, 1)

        # 计算均方误差损失
        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        # 计算PSNR
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        # 更新PSNR平均计算器
        PSNR.update(psnr.item(), source_img.size(0))
    return PSNR.avg

# 定义计算模型大小的函数
def get_model_size(model: torch.nn.Module) -> float:
    # 计算模型的参数总数
    num_params = sum(p.numel() for p in model.parameters())
    #num_bytes = num_params * 4  # assuming 32-bit float
    # 将参数总数转换为MB
    num_megabytes = num_params / (1000 ** 2)
    return num_megabytes

if __name__ == '__main__':


    # 初始化计算标志
    calculate = True


    # 动态创建模型实例
    network = eval(args.model.replace('-', '_'))()
    # 使用数据并行并将模型移动到GPU上
    network = nn.DataParallel(network).cuda()
    #network = nn.DataParallel(network, device_ids=[0]).cuda()

    #criterionL2 = nn.SmoothL1Loss()
    # 定义损失函数
    criterion = nn.L1Loss()

    # 根据配置选择优化器
    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'], weight_decay=0.02)
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'], weight_decay=0.02)
    elif setting['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=setting['lr'], weight_decay=0.02)
    else:
        raise Exception("ERROR: 不支持的优化器")

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min= 1e-6)
    # 定义梯度缩放器
    scaler = GradScaler()

    # 构建数据集路径
    dataset_dir = os.path.join(args.data_dir, args.dataset)

    #UIEB dataset->800 / 90
    # dataset = PairLoader(dataset_dir, 'train', 'train',
    #                            setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [800, 90])
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=setting['batch_size'],
    #                           shuffle=True,
    #                           num_workers=args.num_workers,
    #                           pin_memory=True,
    #                           drop_last=True)
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=setting['batch_size'],
    #                         num_workers=args.num_workers,
    #                         pin_memory=True)
    # 创建训练数据集实例
    train_dataset = PairLoader(dataset_dir, '', 'train','train.txt',
                                setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
    # 创建训练数据加载器
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    # 创建验证数据集实例
    val_dataset = TPairLoader(dataset_dir, '', setting['valid_mode'],'val.txt',
                              setting['patch_size'])
    # 创建验证数据加载器
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    # 构建模型保存目录
    save_dir = os.path.join(args.save_dir, args.exp)
    # 创建模型保存目录，如果已存在则不报错
    os.makedirs(save_dir, exist_ok=True)

    # 如果模型文件不存在，则开始训练
    if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
        print('==> 开始训练，当前模型名称: ' + args.model)
        # print(network)

        # 初始化wandb配置
        #  config = wandb.config
        #  config.update(setting)
        
        calculate = True
        if calculate is True:
        # 计算模型的参数数量
            model_size = get_model_size(network)
            print(f"模型大小: {model_size:.2f} MB")
            calculate = False
        # 初始化最佳PSNR值
        best_psnr = 0
        

        # 开始训练循环
        for epoch in tqdm(range(setting['epochs'] + 1)):
            # 进行一轮训练并返回损失值
            loss = train(train_loader, network, criterion, optimizer, scaler,epoch)
            swanlab.log({"loss": loss})
            print(loss)

            # 学习率调度器更新学习率
            scheduler.step()

#            每隔一定轮数进行验证
            if epoch % setting['eval_freq'] == 0:
                # 进行验证并返回平均PSNR值
                avg_psnr = valid(val_loader, network)

                # #记录验证PSNR到wandb
                swanlab.log({'valid_psnr': avg_psnr}, step=epoch)
                print(f'当前PSNR: {avg_psnr:.2f}\t, 最佳PSNR: {best_psnr:.2f}')
                # 如果当前PSNR大于最佳PSNR，则更新最佳PSNR并保存模型
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict()},
                               os.path.join(save_dir, args.model + '.pth'))

                # 记录最佳PSNR到wandb
                swanlab.log({'best_psnr': best_psnr}, step=epoch)
            #torch.save({'state_dict': network.state_dict()},
           # os.path.join(save_dir, args.model + 'final.pth'))

    else:
        print('==> 已有训练好的模型')
        exit(1)
