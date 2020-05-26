"""
使用 cv2 实现以下数据增强类：
- RandomFlip：随机翻转（水平 or 垂直 or 水平+垂直）
- RandomRotate：随机旋转90°
- RandomWrap：随机中心旋转
- Scale：按比例缩放
- RandomCrop：随机裁剪
- RandomWrap：随机仿射变换
"""
import torchvision.transforms.functional as tf
import random
import numpy as np
from PIL import Image, ImageOps
import cv2

# 随机翻转
class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, img, label):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            label = cv2.flip(label, d)
        return img, label

# 随机旋转90°
class RandomRotate:
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, img, label):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            label = np.rot90(label, factor)
        return img, label

# 随机中心旋转
class  RandomCenterRotate:
    def __init__(self, prob=0.5, limit=90):
        self.prob = prob
        self.limit = limit # 旋转角度限制

    def __call__(self, img, label):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)
            h, w, _ = img.shape
            # 定义旋转中心
            Mat = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1.0)
            img = cv2.warpAffine(img, Mat, (h, w))
            label = cv2.warpAffine(label, Mat, (h, w))
        return img, label

# 随机仿射变换
class RandomWrap:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, label):
        if random.random() < self.prob:
            h, w, _ = img.shape
            # 随机指定变换前的 n 个角点
            pts1 = np.float32([[0, 0], [h - np.random.randint(10), 0], [0, w - np.random.randint(10)]])
            # 定义变换后的 n 个角点
            pts2 = np.float32([[h * 0.3, w * 0.1], [h * 0.9, w * 0.2], [h * 0.1, w * 0.9]])
            # 定义变换中心
            Mat = cv2.getAffineTransform(pts1, pts2)
            img = cv2.warpAffine(img, Mat, (h, w))
            label = cv2.warpAffine(label, Mat, (h, w))
        return img, label

# 按比例缩放
class Scale:
    def __init__(self, prob=0.5, size=224):
        self.size = size

    def __call__(self, img, label):
        w, h, _ = img.shape
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, label
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return cv2.resize(img, (ow, oh)), cv2.resize(label, (ow, oh)),
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return cv2.resize(img, (ow, oh)), cv2.resize(label, (ow, oh))

# 随机裁剪
class RandomCrop:
    def __init__(self, crop_size=(224, 224), padding=None, pad_if_needed=False, fill=0):
        self.crop_size = crop_size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill

    def __call__(self, img, label):
        w, h, _ = img.shape
        ch, cw = self.crop_size
        if w == cw and h == ch:
            return img, label
        i = random.randint(0, h - ch)
        j = random.randint(0, w - cw)
        return img[i:h, j:w], label[i:h, j:w]
