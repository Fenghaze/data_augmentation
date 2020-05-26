from PIL import Image
from torchvision import transforms, utils
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
from augmentation import *

class MyDataset(Dataset):
    def __init__(self, file, train=True, augmentation=None):
        self.file = file
        self.means = [46.3, 53.9, 56.9] # self.means = utils.compute_mean()
        #self.means = [104.00699, 116.66877, 122.67892]
        self.train = train
        self.augmentation = augmentation
        data = pd.read_csv(self.file, index_col=0)
        self.imgs_lst = data.iloc[:, 0]
        self.label_lst = data.iloc[:, 1]

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join('./datasets/images', self.imgs_lst[idx]))
        img = img.astype('float64')
        label = cv2.imread(os.path.join('./datasets/labels', self.label_lst[idx]))
        if self.train:
            img, label = self.augmentation(img, label)
            img = cv2.resize(img, (224, 224))
            label = cv2.resize(label, (224, 224))
            img -= self.means
        # cv2.imread(img, flag)，flag=0表示将图片转为灰度图（三通道改为单通道）
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # 将图片转换为张量形式 HWC → CHW
        to_tensor = transforms.ToTensor()
        # 使用cv2读取图片时，需要转换为RGB通道
        img = to_tensor(img[:, :, ::-1].copy())
        return img, label

    def __len__(self):
        return len(self.imgs_lst)

# 整合数据增强的类
class Compose:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, label):
        assert img.size == label.size
        for tf in self.transform:
            img, label = tf(img, label)
            assert img.size == label.size

        if not isinstance(img, np.ndarray):
            img, label = np.array(img), np.array(label, dtype=np.uint8)
        return img, label

def show_batch(data):
    img = data[0]
    batch_size = len(img)
    grid1 = utils.make_grid(img, batch_size)
    plt.imshow(grid1.numpy().transpose(1,2,0))
    plt.show()

if __name__ == '__main__':
    augmentation = Compose([
        RandomFlip(), # 随机翻转
        RandomRotate(), # 随机旋转90°
        RandomCenterRotate(), # 随机中心旋转
        RandomWrap(), # 随机仿射变换
        RandomCrop(), # 随机裁剪
        Scale(), # 按比例缩放
    ])
    batch_size = 2
    train_data = MyDataset('train.csv', train=True, augmentation=augmentation)
    dataloader = DataLoader(train_data,  batch_size=batch_size,shuffle=True, num_workers=0)
    for i, data in enumerate(dataloader):
        plt.figure()
        show_batch(data)
        break
