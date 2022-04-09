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

部分视频解析出的格式为：(AVPixelFormat)AV_PIX_FMT_YUVJ420P，直接转换会提示警告：Convert Deprecated Format，警告不重要，但最好还是转换不推荐的格式。

``` C++
/// <summary>
/// 转换不推荐的格式
/// </summary>
AVPixelFormat ConvertDeprecatedFormat(enum AVPixelFormat format)
{
	switch (format)
	{
	case AV_PIX_FMT_YUVJ420P:
		return AV_PIX_FMT_YUV420P;
		break;
	case AV_PIX_FMT_YUVJ422P:
		return AV_PIX_FMT_YUV422P;
		break;
	case AV_PIX_FMT_YUVJ444P:
		return AV_PIX_FMT_YUV444P;
		break;
	case AV_PIX_FMT_YUVJ440P:
		return AV_PIX_FMT_YUV440P;
		break;
	default:
		return format;
		break;
	}
}
```

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
同样需要使用 [SwsContext](https://ffmpeg.org/doxygen/2.2/structSwsContext.html) 类转换为 Mat BGR24。

#### 转换代码
``` C++
/// <summary>
/// 从 FFmpeg 图片类型转换为 OpenCV 类型
/// </summary>
Mat VideoDecoder::AVFrameToMat(const AVFrame* frame)
{
	int64 width = frame->width, height = frame->height;
	Mat image(height, width, CV_8UC3);
	int cvLinesizes[1]{ image.step1() };
	auto srcFormat = ConvertDeprecatedFormat((AVPixelFormat)frame->format);
	SwsContext* conversion = sws_getContext(width, height, srcFormat, width, height, AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
	sws_scale(conversion, frame->data, frame->linesize, 0, height, &image.data, cvLinesizes);
	sws_freeContext(conversion);
	return image;
}
```