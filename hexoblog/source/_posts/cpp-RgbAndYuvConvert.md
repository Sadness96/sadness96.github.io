---
title: RGB 与 YUV 互相转换计算
date: 2022-09-12 20:18:00
tags: [c++,ffmpeg,cuda]
categories: C++
---
### 单一像素 RGB 与 YUV 转换计算公式
<!-- more -->
#### 简介
从 FFmpeg 读取视频帧，无论是 RGB 格式或 YUV(YUV420、NV12) 转换到另一种格式都可以通过 [SwsContext](https://www.ffmpeg.org/doxygen/2.2/structSwsContext.html) 系列方法转换，但是使用 CUDA 处理图像时需要单独处理每一像素，在 YUV 与 RBG 间互相转换。

#### 转换代码
##### YUV 转 RBG
``` CPP
public void YuvToRgb(int Y, int U, int V)
{
    double B, G, R;
    R = Y + 1.402 * (V - 128);
    G = Y - 0.34414 * (U - 128) - 0.71414 * (V - 128);
    B = Y + 1.772 * (U - 128);
    cout << "R:" << R << " - G:" << G << " - B:" << B << endl;
}
```

##### RBG 转 YUV
``` CPP
public void RgbToYuv(int R, int G, int B)
{
    double Y, U, V;
    Y = 0.299 * R + 0.587 * G + 0.114 * B;
    U = -0.1687 * R - 0.3313 * G + 0.5 * B + 128;
    V = 0.5 * R - 0.4187 * G - 0.0813 * B + 128;
    cout << "Y:" << Y << " - U:" << U << " - V:" << V << endl;
}
```

#### 相关问题
##### YUV 默认都是 0 时显示绿色
调用方法 YuvToRgb(0, 0, 0) 获取到值：
R:-179.456 - G:135.45984 - B:-226.816
RGB 取值范围为正整数 0 ~ 255，所以显示为:
R:0 - G:135 - B:0

<div style="background: #008700;width: 120px;height: 30px;text-align: center;color: white;">#008700</div>

##### 设置 YUV 为黑色
设置 Y = 0; U = 128; V = 128;