# 导入所需的库

import os
import random
import numpy as np
import cv2
import torch
import sys
sys.path.append('/home/benchunlei/dl/SPMFormer/')
from torch.utils.data import Dataset

from myutils import hwc_to_chw, read_img, read_img_test

def load_image_paths(txt_file):
        # 以只读模式打开文本文件
        with open(txt_file, 'r') as f:
            # 读取文件的所有行，并去除每行末尾的换行符，返回一个列表
            return f.read().splitlines()


def load_images(image_dir, image_names, resize=None):
    # 初始化一个空列表，用于存储加载好的图像
    images = []
    # 遍历图像文件名列表
    for name in image_names:
        # 拼接完整的图像文件路径
        img_path = os.path.join(image_dir, name)
        # 打开图像文件，并将其转换为RGB格式
        img = Image.open(img_path).convert('RGB')
        # 如果指定了resize参数
        if resize:
            # 将图像调整为指定的尺寸
            img = img.resize(resize)
        # 将图像转换为NumPy数组，并添加到列表中
        images.append(np.array(img))
    # 返回包含所有图像的列表
    return images


# 定义一个类，用于对RGB图像进行旋转和翻转操作
### rotate and flip
class Augment_RGB_numpy:
    def __init__(self):
        # 初始化方法，此处无操作
        pass

    def transform0(self, numpy_array):
        # 不进行任何变换，直接返回原始数组
        return numpy_array

    def transform1(self, numpy_array):
        # 对数组进行逆时针90度旋转
        return np.rot90(numpy_array, k=1, axes=(-2,-1))

    def transform2(self, numpy_array):
        # 对数组进行逆时针180度旋转
        return np.rot90(numpy_array, k=2, axes=(-2,-1))

    def transform3(self, numpy_array):
        # 对数组进行逆时针270度旋转
        return np.rot90(numpy_array, k=3, axes=(-2,-1))

    def transform4(self, numpy_array):
        # 对数组在倒数第二个轴上进行翻转
        return np.flip(numpy_array, axis=-2)

    def transform5(self, numpy_array):
        # 先对数组进行逆时针90度旋转，再在倒数第二个轴上进行翻转
        return np.flip(np.rot90(numpy_array, k=1, axes=(-2,-1)), axis=-2)

    def transform6(self, numpy_array):
        # 先对数组进行逆时针180度旋转，再在倒数第二个轴上进行翻转
        return np.flip(np.rot90(numpy_array, k=2, axes=(-2,-1)), axis=-2)

    def transform7(self, numpy_array):
        # 先对数组进行逆时针270度旋转，再在倒数第二个轴上进行翻转
        return np.flip(np.rot90(numpy_array, k=3, axes=(-2,-1)), axis=-2)

# 定义一个类，用于混合两张图像
### mix two images
class MixUp_AUG:
    def __init__(self):
        # 初始化Beta分布，用于生成混合系数
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy, gray_mask):
        # 获取输入图像的批次大小
        bs = rgb_gt.size(0)
        # 生成随机排列的索引
        indices = torch.randperm(bs)
        # 根据索引获取第二组图像
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]
        gray_mask2 = gray_mask[indices]
        # gray_contour2 = gray_mask[indices]
        # 从Beta分布中采样混合系数，并调整形状后移到GPU上
        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        # 对图像和掩码进行混合操作
        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2
        gray_mask = lam * gray_mask + (1-lam) * gray_mask2
        # gray_mask = torch.where(gray_mask>0.01, torch.ones_like(gray_mask), torch.zeros_like(gray_mask))
        # gray_contour = lam * gray_contour + (1-lam) * gray_contour2
        return rgb_gt, rgb_noisy, gray_mask

# 定义一个函数，用于对齐图像
def align(imgs=[], size=320):
    # 获取第一张图像的高度和宽度
    H, W, _ = imgs[0].shape
    # 定义裁剪后的高度和宽度
    Hc, Wc = [size, size]

    # 计算裁剪的起始位置
    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    # 对所有图像进行裁剪操作
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]
    return imgs

# 定义一个函数，用于随机裁剪图像
def crop_images(imgs, ps=256):
    # 获取图像的高度和宽度
    _, H, W = imgs.shape
    # 随机生成裁剪的起始行索引
    r = 0 if H - ps == 0 else np.random.randint(0, H - ps)
    # 随机生成裁剪的起始列索引
    c = 0 if W - ps == 0 else np.random.randint(0, W - ps)
    # 对图像进行裁剪
    imgs = imgs[:, r:r+ps, c:c+ps]
    return imgs

# 创建Augment_RGB_numpy类的实例
augment  = Augment_RGB_numpy()
# 获取Augment_RGB_numpy类实例的所有非私有可调用方法
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]

# 定义一个数据集类，用于加载成对的图像
class PairLoader(Dataset):
# 修复非默认参数跟随默认参数的语法错误，将 txt_file 移到默认参数之前
    def __init__(self, data_dir, sub_dir, mode, txt_file, size=256, edge_decay=0, only_h_flip=False):
        # 确保模式参数在指定范围内
        assert mode in ['train', 'valid', 'test']

        # 保存模式、裁剪大小、边缘衰减系数和是否仅水平翻转的标志
        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        # 拼接数据集根目录
        self.root_dir = os.path.join(data_dir, sub_dir)
        # 获取条件图像的文件名列表
        # self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'cond')))
        self.train_txt = os.path.join(self.root_dir, txt_file)
        print("train_txt:" + self.train_txt)
        self.img_names =  sorted(load_image_paths(self.train_txt))
        # 保存图像数量
        self.img_num = len(self.img_names)

    def __len__(self):
        # 返回数据集的长度，即图像数量
        return self.img_num

    def __getitem__(self, idx):
        # 关闭OpenCV的多线程和OpenCL加速
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        # 获取当前索引对应的图像文件名
        img_name = self.img_names[idx]
        # 读取条件图像，并将像素值从[0, 1]缩放到[-1, 1]
        source_img = read_img(os.path.join(self.root_dir, 'input', img_name)) *2 -1
        #source_img[:,:,2] *= 3.5
        #source_img = source_img * 2 - 1
        # 读取目标图像，并将像素值从[0, 1]缩放到[-1, 1]
        target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
        #mask_img = read_img(os.path.join(self.root_dir, 'mask256', img_name)) * 2 - 1
        # 创建与条件图像形状相同的全1掩码图像
        mask_img = np.ones_like(source_img)

        # 将图像从HWC格式转换为CHW格式
        source_img = hwc_to_chw(source_img)
        target_img = hwc_to_chw(target_img)
        mask_img = hwc_to_chw(mask_img)
        #condmask_img = hwc_to_chw(condmask_img)

        #source_img = crop_images(source_img)
        #target_img = crop_images(target_img)
        #mask_img = crop_images(mask_img)

        #apply_trans = transforms_aug[random.getrandbits(3)]
        #source_img = getattr(augment, apply_trans)(source_img)
        #target_img = getattr(augment, apply_trans)(target_img)
        #mask_img = getattr(augment, apply_trans)(mask_img)
        [source_img, target_img, mask_img] = [source_img, target_img, mask_img]

        # if self.mode == 'train':
        #     [source_img, target_img, mask_img, condmask_img] = augment([source_img, target_img, mask_img, condmask_img], self.size, self.edge_decay, self.only_h_flip)
        #
        # if self.mode == 'valid':
        # 将图像转换为连续内存布局
        source_img = np.ascontiguousarray(source_img)
        target_img = np.ascontiguousarray(target_img)
        mask_img = np.ascontiguousarray(mask_img)
        #condmask_img = np.ascontiguousarray(condmask_img)
        # 返回包含条件图像、目标图像、掩码图像和文件名的字典
        return {'source': (source_img), 'target': (target_img),'mask':(mask_img), 'filename': img_name}

# 定义一个数据集类，用于测试时加载成对的图像
class TPairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, txt_file, size=256, edge_decay=0, only_h_flip=False):
        # 确保模式参数在指定范围内
        assert mode in ['train', 'valid', 'test']

        # 保存模式、裁剪大小、边缘衰减系数和是否仅水平翻转的标志
        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        # 拼接数据集根目录
        self.root_dir = os.path.join(data_dir, sub_dir)
        # 获取目标图像的文件名列表
        # self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'gt')))
        self.val_txt = os.path.join(self.root_dir, "val.txt")
        self.img_names =  sorted(load_image_paths(self.val_txt))
        # 保存图像数量
        self.img_num = len(self.img_names)

    def __len__(self):
        # 返回数据集的长度，即图像数量
        return self.img_num

    def __getitem__(self, idx):
        # 关闭OpenCV的多线程和OpenCL加速
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        # 获取当前索引对应的图像文件名
        img_name = self.img_names[idx]
        # 读取条件图像，并将像素值从[0, 1]缩放到[-1, 1]
        source_img = read_img_test(os.path.join(self.root_dir, 'input', img_name)) *2 -1
        #source_img[:,:,2] *= 3.5
        #source_img = source_img * 2 - 1
        # 读取目标图像，并将像素值从[0, 1]缩放到[-1, 1]
        target_img = read_img_test(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
        #mask_img = read_img(os.path.join(self.root_dir, 'mask256', img_name)) * 2 - 1
        # 创建与条件图像形状相同的全1掩码图像
        mask_img = np.ones_like(source_img)

        # 将图像从HWC格式转换为CHW格式
        source_img = hwc_to_chw(source_img)
        target_img = hwc_to_chw(target_img)
        mask_img = hwc_to_chw(mask_img)

        # 将图像转换为连续内存布局
        source_img = np.ascontiguousarray(source_img)
        target_img = np.ascontiguousarray(target_img)
        mask_img = np.ascontiguousarray(mask_img)
        #condmask_img = np.ascontiguousarray(condmask_img)
        # 返回包含条件图像、目标图像、掩码图像和文件名的字典
        return {'source': (source_img), 'target': (target_img),'mask':(mask_img), 'filename': img_name}

# 定义一个数据集类，用于加载单张图像
class SingleLoader(Dataset):
    def __init__(self, root_dir):
        # 保存数据集根目录
        self.root_dir = root_dir
        # 获取图像的文件名列表
        # self.img_names = sorted(os.listdir(self.root_dir))
        train_txt = os.path.join(root_dir, "train.txt")
        self.img_names =  sorted(load_image_paths(train_txt))
        # 保存图像数量
        self.img_num = len(self.img_names)

    def __len__(self):
        # 返回数据集的长度，即图像数量
        return self.img_num

    def __getitem__(self, idx):
        # 关闭OpenCV的多线程和OpenCL加速
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        # 获取当前索引对应的图像文件名
        img_name = self.img_names[idx]
        # 读取图像，并将像素值从[0, 1]缩放到[-1, 1]
        img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

        # 返回包含图像和文件名的字典
        return {'img': hwc_to_chw(img), 'filename': img_name}


def main():
    """
    测试数据加载器的主函数
    """
    # print("测试SingleLoader:")
    # single_loader = SingleLoader('/home/benchunlei/dl/SPMFormer/dataset/UIEB/challenging-60')
    # print(f"数据集长度: {len(single_loader)}")
    # sample = single_loader[0]
    # print(f"样本包含的键: {sample.keys()}")
    # print(f"图像形状: {sample['img'].shape}")
    
    # print("\n测试PairLoader:")
    # pair_loader = PairLoader(
    #     data_dir='/home/benchunlei/dl/SPMFormer/dataset',
    #     sub_dir='LSUI',
    #     mode='train',
    #     txt_file='train.txt'
    # )

    # pair_sample = pair_loader[0]
    # print(f"长度: {len(pair_loader)}")
    # print(f"样本包含的键: {pair_sample.keys()}")
    # print(f"源图像形状: {pair_sample['source'].shape}")
    # print(f"目标图像形状: {pair_sample['target'].shape}")




    print("\n测试TPairLoader:")
    pair_loader = TPairLoader(
        data_dir='/home/benchunlei/dl/SPMFormer/dataset',
        sub_dir='LSUI',
        mode='train',
        txt_file='val.txt'
    )
    pair_sample1 = pair_loader[0]
    print(f"长度: {len(pair_loader)}")
    print(f"样本包含的键: {pair_sample1.keys()}")
    print(f"源图像形状: {pair_sample1['source'].shape}")
    print(f"目标图像形状: {pair_sample1['target'].shape}")

if __name__ == '__main__':
    main()


