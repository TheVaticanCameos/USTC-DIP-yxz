# 03 Play With GANs

> 本仓库是中国科学技术大学2024年秋季学期深度学习（MATH6420P.02）的课程作业，拥有者为杨萱泽。

本实验实现了：
- 基于 Pix2Pix + GAN 的图像语义分割
- 基于 DragGAN 和 Facial Landmarks Detection 的人脸图像编辑

## 环境依赖

### 创建并激活虚拟环境

#### 创建虚拟环境

在当前目录终端中运行

```bash
python -m venv env-03
```

#### 激活虚拟环境

在`Linux`系统中，运行：
```bash
source env-03/bin/activate
```

在`Windows`系统中，运行：
```bash
.\env-03\Scripts\activate
```

#### 安装依赖

运行
```bash
pip install -r requirements.txt
```

### 下载数据集

在当前目录运行
```bash
bash download_dataset.sh
```

等待运行完成即可。

## 实验过程与运行结果

基于 Pix2Pix + GAN 的图像语义分割：[Pix2PixWithGAN](Pix2PixWithGAN/README.md)

基于 DragGAN 和 Facial Landmarks Detection 的人脸图像编辑：[AutoFaceGradGAN](AutoFaceDragGAN/README.md)
