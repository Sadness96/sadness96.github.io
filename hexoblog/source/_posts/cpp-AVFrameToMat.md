---
title: AVFrame To Mat
date: 2022-04-07 21:12:18
tags: [c++,ffmpeg,opencv]
categories: C++
---
### FFmpeg(yuv420p、nv12) 转换为 OpenCV Mat 类型
<!-- more -->
#### 简介
[OpenCV](https://opencv.org/) 可以使用 [VideoCapture](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html) 类读取视频，虽然同样是封装了 [FFmpeg](https://ffmpeg.org/)，但是也屏蔽了很多接口，想做一些复杂操作就很不方便。
所以改用 FFmpeg 读取视频传递给 OpenCV 使用，将视频帧 FFmpeg [AVFrame](https://www.ffmpeg.org/doxygen/4.1/structAVFrame.html) 转换为 OpenCV [Mat](https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html)。

#### 详细转换
##### 软解码
软解码解析出的 AVFrame 格式为：(AVPixelFormat)AV_PIX_FMT_YUV420P。
需要使用 [SwsContext](https://ffmpeg.org/doxygen/2.2/structSwsContext.html) 类转换为 Mat BGR24。

##### 硬解码
硬解码解析出的 AVFrame 格式为：显存 NV12。
硬解码类型：AV_HWDEVICE_TYPE_DXVA2 解析结果为 (AVPixelFormat)AV_PIX_FMT_FXVA2_VLD。
硬解码类型：AV_HWDEVICE_TYPE_CUDA 解析结果为 (AVPixelFormat)AV_PIX_FMT_CUDA。

使用 [av_hwframe_transfer_data](https://ffmpeg.org/doxygen/3.2/hwcontext_8h.html) 函数把显存数据统一转换为内存数据 NV12。

``` C++
if (pCodecCtx->hw_device_ctx)
{
    AVFrame* hw_frame;
    hw_frame = av_frame_alloc();
    av_hwframe_transfer_data(hw_frame, frame, 0);
}
```

内存数据 NV12 格式为：(AVPixelFormat)AV_PIX_FMT_NV12。
接下来使用 [memcpy](https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/memcpy-wmemcpy?view=msvc-170) 内存拷贝函数与 [cvtColor](https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html) 函数将图像转换为 Mat NV12。

#### 转换代码
``` C++
/// <summary>
/// 从 FFmpeg 图片类型转换为 OpenCV 类型
/// </summary>
Mat VideoDecoder::AVFrameToMat(const AVFrame* frame)
{
	int64 width = frame->width, height = frame->height;
	Mat image(height, width, CV_8UC3);

	switch (frame->format)
	{
	case AV_PIX_FMT_YUV420P:
	{
		int cvLinesizes[1];
		cvLinesizes[0] = image.step1();
		SwsContext* conversion = sws_getContext(width, height, (AVPixelFormat)frame->format, width, height, AVPixelFormat::AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
		sws_scale(conversion, frame->data, frame->linesize, 0, height, &image.data, cvLinesizes);
		sws_freeContext(conversion);
		break;
	}
	case AV_PIX_FMT_NV12:
	{
		Mat tmp_img = Mat::zeros(height * 3 / 2, width, CV_8UC1);
		memcpy(tmp_img.data, frame->data[0], width * height);
		memcpy(tmp_img.data + width * height, frame->data[1], width * height / 2);
		cvtColor(tmp_img, image, CV_YUV2BGR_NV12);
		break;
	}
	}
	return image;
}
```