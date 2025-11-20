---
title: Python Yolo Demo
date: 2025-11-20 14:12:10
tags: [python,yolo]
categories: Python
---
<img src="https://cdn.prod.website-files.com/680a070c3b99253410dd3dcf/68430089e5357bbc12e74c31_hero-image-2.avif" style="max-height: 352px; width: auto;" />
<!-- more -->

#### 简介
[YOLO](https://www.ultralytics.com/) 是一种高效的实时目标检测算法，它通过单个网络直接预测目标的边界框和类别概率，从而实现了快速的目标检测。

#### 安装环境
##### CUDA
[CUDA](https://www.nvidia.cn/geforce/) 官网下载安装，YOLO基于可以使用英伟达CUDA技术加速
我这里使用 [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) 

##### Python
[Python](https://www.python.org/) 官网下载安装，或者使用 [Anaconda](https://www.anaconda.com/) 管理器安装，可以同时管理多个 Python 环境。

##### PyCharm（可选）
[PyCharm](https://www.jetbrains.com/pycharm/) 官网下载安装，[JetBrains](https://www.jetbrains.com/) 对 Python 提供的 IDE。

##### PyTorch
[PyTorch](https://pytorch.org/) 深度学习框架，大部分YOLO基于PyTorch实现，在 [Installing previous versions of PyTorch](https://pytorch.org/get-started/previous-versions/) 中搜索 v2.5.0 版本 对应 CUDA 11.8 的安装命令
``` cmd
# CUDA 11.8 
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

##### Ultralytics YOLO
[Ultralytics YOLO](https://docs.ultralytics.com/zh/) 官方安装测试

###### 安装
``` cmd
# Install or upgrade the ultralytics package from PyPI 
pip install -U ultralytics
```

###### 测试
``` python
import torch  
  
# 检查GPU是否可用  
print("PyTorch Version:", torch.__version__)  
print("CUDA Available:", torch.cuda.is_available())  
print("CUDA Version:", torch.version.cuda)  
print("CUDA Device:", torch.cuda.current_device())  
print("CUDA DeviceName:", torch.cuda.get_device_name(0))
```

###### 最小预测程序
``` python
from ultralytics import YOLO  
  
# Load a pre-trained YOLO model  
model = YOLO("yolo11n.pt")  
  
# Start tracking objects in a video  
# You can also use live video streams or webcam input  
model.track(source=r"D:\1.mp4", show=True, save=False)
```

###### 最小训练程序
``` python
from ultralytics import YOLO  
  
if __name__ == "__main__":  
    # Load a model  
    model = YOLO("yolo11n.pt")  
  
    # Train the model  
    results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```