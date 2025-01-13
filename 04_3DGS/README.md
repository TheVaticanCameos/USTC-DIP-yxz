# 04 3DGS

> 本仓库是中国科学技术大学2024年秋季学期深度学习（MATH6420P.02）的课程作业，拥有者为杨萱泽。

本实验实现了：
- 基于多视角图像的 3DGS 重建

## 实验步骤

### Step 1. 运行 SfM

本步骤中使用`colmap`进行相机位姿估计。在`source`目录下运行 

```bash
python mvs_with_colmap.py --data_dir ../data/chair
```

和

```bash
python mvs_with_colmap.py --data_dir ../data/lego
```

从而估计出多视角图片数据集的相机位姿。

此时可以在当前目录运行

```bash
python debug_mvs_by_projecting_pts.py --data_dir ../data/chair
```

以及 

```bash
python debug_mvs_by_projecting_pts.py --data_dir ../data/lego
```

来检查相机位姿估计结果。

### Step 2. 简化的 3DGS 重建

这一步骤中将稀疏的3D点转化为3DGS表达的场景，并将其投影到2维空间，进行体渲染。具体步骤如下：

#### 2.1 3DGS初始化

将稀疏的点转化为3DGS分布，并指定协方差矩阵。同时对每个Gauss椭球附加上不透明度以及颜色属性，以便于后续做体渲染使用。

#### 2.2 投影到2D空间

先从世界坐标转化到相机坐标，然后对其投影到2D空间。

#### 2.3 计算Gauss椭球

使用二维空间中的Gauss分布进行体渲染。具体公式如下：

$$
f(\mathbf{x}; \boldsymbol{\mu}\_{i}, \boldsymbol{\Sigma}\_{i}) = \frac{1}{2 \pi \sqrt{ | \boldsymbol{\Sigma}\_{i} |}} \exp \left ( {-\frac{1}{2}} (\mathbf{x} - \boldsymbol{\mu}\_{i})^T \boldsymbol{\Sigma}\_{i}^{-1} (\mathbf{x} - \boldsymbol{\mu}\_{i}) \right ) = \frac{1}{2 \pi \sqrt{ | \boldsymbol{\Sigma}\_{i} |}} \exp \left ( P_{(\mathbf{x}, i)} \right )
$$

#### 2.4 体渲染

使用 $ \alpha $ blending 进行体渲染，具体公式为

$$
T_{\left(x, i\right)}=\prod_{j=1}^{i-1}\left(1-\alpha_{\left(x, j\right)}\right)
$$

---

在source目录运行如下命令

```bash
python train.py --data_dir ../data/chair
```

和 

```bash
python train.py --data_dir ../data/lego
```

即可完成上述步骤 2.1~2.4。

### Step 3. 与原始 3DGS 重建结果对比

由于本实验中并没有针对 3DGS 进行优化，因此计算效率上远远不如 3DGS 的原始实现。

[返回根目录文档](../README.md)
