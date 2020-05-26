"""
1、生成train.csv文件
2、计算训练集图片的像素均值
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# 生成数据集文件 data.csv
def data_csv():
    basedir = './datasets'
    img_lst = os.listdir(basedir + '/images')
    lable_lst = os.listdir(basedir + '/labels')
    data = {
        'img': img_lst,
        'lable': lable_lst
    }
    data_file = pd.DataFrame(data)
    data_file.to_csv('data.csv')

# 查看生成的 csv 文件
def show_csv():
    data = pd.read_csv('data.csv', index_col=0)
    print(data.head())

# 划分数据集并生成 train.csv 和 val.csv
def split_dataset():
    data = pd.read_csv('data.csv', index_col=0)
    X = data.iloc[:, 0]
    y = data.iloc[:, 1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)
    # 生成训练集文件 train.csv
    train_data = {
        'img': X_train,
        'label': y_train
    }
    train_file = pd.DataFrame(train_data).reset_index(drop=True)
    train_file.to_csv('train.csv')
    # 生成验证集文件 val.csv
    val_data = {
        'img': X_val,
        'label': y_val
    }
    val_file = pd.DataFrame(val_data).reset_index(drop=True)
    val_file.to_csv('val.csv')

# 计算图片均值
def compute_mean():
    train_datasets = pd.read_csv('train.csv', index_col=0)
    train_imgs = train_datasets.iloc[:, 0]
    means = [0, 0, 0]
    for img_name in train_imgs:
        # img = mpimg.imread(os.path.join('./datasets/images', img_name))
        img = cv2.imread(os.path.join('./datasets/images', img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(3):
            means[i] += img[:, :, i].mean()
    means = np.array(means) / len(train_imgs)


    # 显示减去均值后的图片
    img = mpimg.imread(os.path.join('./datasets/images', '0001TP_006690.png'))
    img1 = img - means
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(img1)
    plt.show()

    return means

if __name__ == '__main__':
    # data_csv()
    # show_csv()
    # split_dataset()
    means = compute_mean()
    print(means)