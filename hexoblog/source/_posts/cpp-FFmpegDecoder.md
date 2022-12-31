---
title: FFmpeg 解码视频
date: 2021-11-08 09:54:40
tags: [c++,ffmpeg]
categories: C++
---
### 使用 FFmpeg 解码视频并显示
<!-- more -->
#### 简介
[FFmpeg](https://ffmpeg.org/) 是一个完整的跨平台解决方案，用于录制、转换和流式传输音频和视频。

#### 安装与配置开发环境
这里不记录如何编译 FFmpeg，不需要精简打包大小或开关一些功能的话直接 [下载](https://ffmpeg.org/download.html) 编译好的版本即可，该篇代码基于第三方的发布版本 n4.4-178-g4b583e5425-20211018 开发。

##### 配置系统环境
1. 解压下载后的压缩包到文件目录。
1. 配置文件夹下 .\bin 目录到环境变量。
1. 在命令行中输入 ffmpeg 与 ffplay 测试配置成功。

##### 配置开发环境
1. 拷贝 ffmpeg 目录中 .\include 与 .\lib 到 C++ 工程目录。
1. 项目属性中：VC++ 目录 → 包含目录，选择 .\include 文件夹。
1. 项目属性中：VC++ 目录 → 库目录，选择 .\lib 文件夹。

#### 核心代码
##### 参数变量
``` cpp
private:
	/// <summary>
	/// 是否启用 TCP 优化解码
	/// </summary>
	bool is_tcp_decode_ = true;

	/// <summary>
	/// 是否多线程软解码
	/// </summary>
	bool is_thread_soft_decoding_ = false;

	/// <summary>
	/// 是否硬解码
	/// </summary>
	bool is_hard_decoding_ = true;

	/// <summary>
	/// 硬解码类型
	/// </summary>
	int hw_type_ = AV_HWDEVICE_TYPE_CUDA;

	/// <summary>
	/// 控制 FPS
	/// 读取文件视频时使用
	/// 读取 RTSP 视频流会导致花屏
	/// </summary>
	bool is_control_fps_ = true;
```

##### 解码视频
``` cpp
// 引用 FFmpeg C 头文件
extern "C"
{
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libavutil/time.h>
#include <libavutil/fifo.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
}

#pragma comment(lib,"winmm.lib")
#pragma comment(lib,"avcodec.lib")
#pragma comment(lib,"avformat.lib")
#pragma comment(lib,"avutil.lib")
#pragma comment(lib,"swscale.lib")

using namespace std;

void main()
{
	/// <summary>
	/// 视频路径
	/// </summary>
	string videoUrl_ = "rtsp://localhost:8554/live";

	// 初始化 FFmpeg
	av_register_all();
	avformat_network_init();

	AVFormatContext* inputContext = NULL;
	inputContext = avformat_alloc_context();

	// 设置连接超时
	AVDictionary* dict = nullptr;
	if (is_tcp_decode_)
	{
		// 读取最大字节数 100KB
		inputContext->probesize = 100 * 1024;
		// 读取最大时长 200ms
		inputContext->max_analyze_duration = 200 * 1000;
		// 优先连接方式改为 TCP
		av_dict_set(&dict, "rtsp_transport", "tcp", 0);
		// 扩大缓冲区，减少卡顿或花屏
		av_dict_set(&dict, "buffer_size", "1024000", 0);
	}
	// 设置超时断开
	av_dict_set(&dict, "stimeout", "2000000", 0);

	int ret = avformat_open_input(&inputContext, videoUrl_.c_str(), NULL, &dict);
	if (ret < 0)
	{
		av_dict_free(&dict);
		avformat_free_context(inputContext);
		PrintError(ret);
		return;
	}
	ret = avformat_find_stream_info(inputContext, NULL);
	if (ret < 0)
	{
		PrintError(ret);
	}

	// 打印视频信息
	av_dump_format(inputContext, NULL, videoUrl_.c_str(), 0);

	// 打印音视频信息
	AVStream* audioStream = nullptr;
	AVStream* videoStream = nullptr;
	for (int i = 0; i < inputContext->nb_streams; i++)
	{
		if (inputContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
		{
			audioStream = inputContext->streams[i];
			cout << "===== 音频 =====" << endl;
			cout << "sample_rate:" << audioStream->codecpar->sample_rate << endl;
		}
		else if (inputContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			videoStream = inputContext->streams[i];
			cout << "===== 视频 =====" << endl;
			cout << "width:" << videoStream->codecpar->width << " height:" << videoStream->codecpar->height << endl;
		}
	}

	// 解码视频数据
	int videoIndex = -1;
	for (int i = 0; i < inputContext->nb_streams; i++)
	{
		if (inputContext->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			videoIndex = i;
			break;
		}
	}
	if (videoIndex == -1)
	{
		cout << "Didn't find a video stream\n" << endl;
		return;
	}
	AVCodecContext* pCodecCtx;
	AVCodec* pCodec;
	pCodecCtx = inputContext->streams[videoIndex]->codec;
	pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
	if (pCodec == NULL)
	{
		printf("Codec not found.\n");
		return;
	}
	if (is_thread_soft_decoding_)
	{
		// 启用多线程软解码
		pCodecCtx->thread_count = 0;
	}
	if (is_hard_decoding_)
	{
		// 启用硬解码
		AVBufferRef* hw_ctx = nullptr;
		av_hwdevice_ctx_create(&hw_ctx, (AVHWDeviceType)hw_type_, NULL, NULL, 0);
		pCodecCtx->hw_device_ctx = av_buffer_ref(hw_ctx);
	}
	if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0)
	{
		printf("Could not open codec.\n");
		return;
	}

	// 解码帧
	AVPacket* packet;
	packet = (AVPacket*)av_malloc(sizeof(AVPacket));
	AVFrame* frame, * hw_frame;
	frame = av_frame_alloc();
	hw_frame = av_frame_alloc();

	while (true)
	{
		clock_t startSendTimeOut, stopSendTimeOut;
		startSendTimeOut = clock();

		ret = av_read_frame(inputContext, packet);
		if (ret < 0) {
			cout << "Read Frame Error. Error Code:" << ret << endl;
			PrintError(ret);
			// 释放 AVPacket
			av_free_packet(packet);
			return;
		}

		if (!packet || packet->stream_index != videoStream->index)
		{
			// 判断是否是视频流
			continue;
		}

		ret = avcodec_send_packet(pCodecCtx, packet);
		if (ret < 0)
		{
			cout << "Send Packet Error. Error Code:" << ret << endl;
			PrintError(ret);
			// 释放 AVPacket
			av_free_packet(packet);
			return;
		}
		stopSendTimeOut = clock();

		while (ret >= 0)
		{
			clock_t startReceiveTimeOut;
			startReceiveTimeOut = clock();

			if (frame == nullptr)
			{
				frame = av_frame_alloc();
			}
			auto ret = avcodec_receive_frame(pCodecCtx, frame);
			if (ret < 0)
			{
				av_frame_free(&frame);
				break;
			}
			if (ret == 0)
			{
				auto pframe = frame;
				if (pCodecCtx->hw_device_ctx)
				{
					// 硬解码转换 显存 => 内存
					av_hwframe_transfer_data(hw_frame, frame, 0);
					pframe = hw_frame;
				}

				// TODO: 可使用 SDL 或 OpenCV 显示视频
				// 打印每一帧数据 AVFrame 编码类型
				cout << pframe->format << endl;

				// 控制 FPS
				if (is_control_fps_)
				{
					auto dur = RescaleToMs(inputContext, packet->duration, packet->stream_index);
					if (dur < 40)
					{
						dur = 40;
					}

					auto vSendTimeOut = stopSendTimeOut - startSendTimeOut;
					auto vReceiveTimeOut = clock() - startReceiveTimeOut;
					auto vTimeOut = vSendTimeOut = vReceiveTimeOut;
					if (vTimeOut >= 0 && vTimeOut < dur)
					{
						dur -= vTimeOut;

						timeBeginPeriod(1);
						this_thread::sleep_for(milliseconds(dur));
						timeEndPeriod(1);
					}
				}
			}
			// 释放 AVFrame
			av_frame_free(&frame);
		}
		// 释放 AVPacket
		av_free_packet(packet);
	}
	// 释放 AVFormatContext
	avformat_close_input(&inputContext);
}
```

#### 注意事项
##### TCP 解码
FFmpeg 默认使用 UDP 解码视频，可设置强制使用 TCP 解码，传输更稳定。
``` cpp
string videoUrl_ = "";
AVFormatContext* inputContext = NULL;
AVDictionary* dict = nullptr;
...
// 读取最大字节数 100KB
inputContext->probesize = 100 * 1024;
// 读取最大时长 200ms
inputContext->max_analyze_duration = 200 * 1000;
// 优先连接方式改为 TCP
av_dict_set(&dict, "rtsp_transport", "tcp", 0);
// 扩大缓冲区，减少卡顿或花屏
av_dict_set(&dict, "buffer_size", "1024000", 0);
...
int ret = avformat_open_input(&inputContext, videoUrl_.c_str(), NULL, &dict);
```

##### 多线程软解码（CPU YUV420）
FFmpeg 软解码默认使用单线程解码，可设置为多线程解码，速度更快。
``` cpp
AVCodecContext* pCodecCtx;
...
// 启用多线程软解码
// 设置 CPU 线程数（0 - 16），默认值为单线程 1，值为 0 时自动检测
pCodecCtx->thread_count = 0;
...
```

##### 硬解码（GPU NV12）
FFmpeg 默认使用软解码（CPU），可设置以硬解码（GPU）的方式解码。
###### 获取支持的硬解码类型
``` cpp
AVCodecID codec_id = AV_CODEC_ID_H264;
auto codec = avcodec_find_decoder(codec_id);

for (int i = 0;; i++) {
    auto config = avcodec_get_hw_config(codec, i);
    if (!config)break;
    if (config->device_type) {
        cout << av_hwdevice_get_type_name(config->device_type) << endl;
    }
}
```

当前机器测试的支持环境有
``` cpp
enum AVHWDeviceType {
    AV_HWDEVICE_TYPE_CUDA,
    AV_HWDEVICE_TYPE_DXVA2,
    AV_HWDEVICE_TYPE_D3D11VA
};
```

###### 启用硬解码
``` cpp
AVCodecContext* pCodecCtx;
...
// 启用硬解码 例如：CUDA
AVBufferRef* hw_ctx = nullptr;
av_hwdevice_ctx_create(&hw_ctx, (AVHWDeviceType)AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0);
pCodecCtx->hw_device_ctx = av_buffer_ref(hw_ctx);
...
```

###### 硬解码转换
* 使用硬解码返回的数据类型为显存类型
``` cpp
enum AVPixelFormat {
    AV_PIX_FMT_CUDA, // 119
    AV_PIX_FMT_DXVA2_VLD, // 53
    AV_PIX_FMT_D3D11 // 174
};
```

* 需要转换为内存类型
``` cpp
AVFrame* frame, * hw_frame;
frame = av_frame_alloc();
hw_frame = av_frame_alloc();
...
auto pframe = frame;
if (pCodecCtx->hw_device_ctx)
{
	// 硬解码转换 显存 => 内存
	av_hwframe_transfer_data(hw_frame, frame, 0);
	pframe = hw_frame;
}
// 打印每一帧数据 AVFrame 编码类型
cout << pframe->format << endl;
```

* 转换后数据类型为内存类型 NV12
``` cpp
enum AVPixelFormat {
    AV_PIX_FMT_NV12 // 23
};
```

##### 控制 FPS
在读取文件视频时，为了转码则忽略，如果是为了显示视频，需要控制帧率，按照视频播放帧率解码渲染，否则会以解码速度加速显示，在读取 RTSP/RTMP 等流视频时则忽略，否则会造成花屏。
C++11 中的 [this_thread::sleep_for](https://docs.microsoft.com/zh-cn/cpp/standard-library/thread-functions?view=msvc-170&redirectedfrom=MSDN&f1url=%3FappId%3DDev16IDEF1%26l%3DZH-CN%26k%3Dk) 不够精准，使用 [timeBeginPeriod](https://docs.microsoft.com/zh-cn/windows/win32/api/timeapi/nf-timeapi-timebeginperiod?f1url=%3FappId%3DDev16IDEF1%26l%3DZH-CN%26k%3Dk) 与 [timeEndPeriod](https://docs.microsoft.com/en-us/previous-versions/ms713415(v=vs.85)) 改变系统计时器的分辨率的方式可以让 sleep 更加精准。此功能会影响全局的 Windows 设置，所以必须将每次对 timeBeginPeriod 的调用与对 timeEndPeriod 的调用相匹配，并在两个调用中指定相同的最小分辨率。
``` cpp
while (true)
{
    int dur = 帧间隔（毫秒） - 解码耗时（毫秒）;

    timeBeginPeriod(1);
    this_thread::sleep_for(milliseconds(dur));
    timeEndPeriod(1);
}
```