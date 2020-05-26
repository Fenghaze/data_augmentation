# 语义分割的数据增强
- utils.py:  分割数据集并生成`.csv`文件
- augmentation.py:  使用cv2实现以下变化
  - 几何变换：
    - RandomFlip：随机翻转（水平 or 垂直 or 水平+垂直）
    - RandomRotate：随机旋转90°
    - RandomWrap：随机中心旋转
    - Scale：按比例缩放
    - RandomCrop：随机裁剪
    - RandomWrap：随机仿射变换
- dataset.py:  自定义数据集类



未完待续