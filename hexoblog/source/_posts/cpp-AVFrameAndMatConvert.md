---
title: AVFrame And Mat Convert
date: 2022-04-07 21:12:18
tags: [c++,ffmpeg,opencv]
categories: C++
---
### FFmpeg(yuv420p、nv12) 与 OpenCV Mat 互相转换
<!-- more -->
#### 简介
[OpenCV](https://opencv.org/) 可以使用 [VideoCapture](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html) 类读取视频，虽然同样是封装了 [FFmpeg](https://ffmpeg.org/)，但是也屏蔽了很多接口，想做一些复杂操作就很不方便。
所以改用 FFmpeg 读取视频传递给 OpenCV 使用，将视频帧 FFmpeg [AVFrame](https://www.ffmpeg.org/doxygen/4.1/structAVFrame.html) 转换为 OpenCV [Mat](https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html)。

#### 解码帧
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
硬解码类型：AV_HWDEVICE_TYPE_CUDA 解析结果为 (AVPixelFormat)AV_PIX_FMT_CUDA。
硬解码类型：AV_HWDEVICE_TYPE_DXVA2 解析结果为 (AVPixelFormat)AV_PIX_FMT_FXVA2_VLD。
硬解码类型：AV_HWDEVICE_TYPE_D3D11VA 解析结果为 (AVPixelFormat)AV_PIX_FMT_D3D11。



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

#### 转换 AVFrame To Mat
``` C++
/// <summary>
/// 从 FFmpeg 图片类型转换为 OpenCV 类型
/// 修改尺寸
/// </summary>
/// <param name="frame">FFmpeg 图像</param>
/// <param name="dstWidth">输出图像宽度</param>
/// <param name="dstHeight">输出图像高度</param>
/// <param name="isfree">是否释放内存</param>
/// <returns>Mat</returns>
Mat AVFrameToMat(AVFrame* frame, int dstWidth, int dstHeight, bool isfree)
{
	Mat image(dstHeight, dstWidth, CV_8UC3);

	int srcWidth = frame->width;
	int srcHeight = frame->height;

	int cvLinesizes[1]{ image.step1() };
	auto srcFormat = ConvertDeprecatedFormat((AVPixelFormat)frame->format);
	SwsContext* conversion = sws_getContext(srcWidth, srcHeight, srcFormat, dstWidth, dstHeight, AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
	sws_scale(conversion, frame->data, frame->linesize, 0, srcHeight, &image.data, cvLinesizes);
	sws_freeContext(conversion);
	if (isfree)
	{
		av_frame_free(&frame);
	}
	return image;
}

/// <summary>
/// 从 FFmpeg 图片类型转换为 OpenCV 类型
/// </summary>
/// <param name="frame">FFmpeg 图像</param>
/// <param name="isfree">是否释放内存</param>
/// <returns>Mat</returns>
Mat AVFrameToMat(AVFrame* frame, bool isfree)
{
	return AVFrameToMat(frame, frame->width, frame->height, isfree);
}
```

#### 转换 Mat To AVFrame
``` cpp
/// <summary>
/// 从 OpenCV 图片类型转换为 FFmpeg 类型
/// 修改尺寸
/// </summary>
/// <param name="image">OpenCV 图像</param>
/// <param name="frame">FFmpeg 图像</param>
/// <param name="dstWidth">输出图像宽度</param>
/// <param name="dstHeight">输出图像高度</param>
/// <returns>AVFrame</returns>
AVFrame* MatToAVFrame(Mat* image, AVFrame* frame, int dstWidth, int dstHeight)
{
	if (frame == NULL)
	{
		frame = av_frame_alloc();
		frame->format = AV_PIX_FMT_YUV420P;
		frame->width = dstWidth;
		frame->height = dstHeight;
		av_frame_get_buffer(frame, 0);
		av_frame_make_writable(frame);
	}

	int srcWidth = image->cols;
	int srcHeight = image->rows;

	int cvLinesizes[1]{ image->step1() };
	SwsContext* conversion = sws_getContext(srcWidth, srcHeight, AV_PIX_FMT_BGR24, dstWidth, dstHeight, (AVPixelFormat)frame->format, SWS_FAST_BILINEAR, NULL, NULL, NULL);
	sws_scale(conversion, &image->data, cvLinesizes, 0, srcHeight, frame->data, frame->linesize);
	sws_freeContext(conversion);
	return frame;
}

/// <summary>
/// 从 OpenCV 图片类型转换为 FFmpeg 类型
/// </summary>
/// <param name="image">OpenCV 图像</param>
/// <param name="frame">FFmpeg 图像</param>
/// <returns>AVFrame</returns>
AVFrame* MatToAVFrame(Mat* image, AVFrame* frame)
{
	return MatToAVFrame(image, frame, image->cols, image->rows);
}
```