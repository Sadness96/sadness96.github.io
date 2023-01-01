---
title: FFmpeg 硬编码视频
date: 2022-12-31 15:40:55
tags: [c++,ffmpeg]
categories: C++
---
### 使用 FFmpeg QSV NVENC 硬编码视频并推流
<!-- more -->
### 简介
在处理 FFmpeg 硬编码前需了解 [FFmpeg 编码视频](https://sadness96.github.io/blog/2022/08/20/cpp-FFmpegEncoder/)。
在 FFmpeg 中软编码（AV_CODEC_ID_H264、AV_CODEC_ID_HEVC）对硬件要求较低，可以说能运行系统的 CPU 都可以正常编码，而硬编码（Intel qsv、NVIDIA nvenc）却直接要求硬件型号以及相对应驱动版本，但是同样的可以给编码效率带来大幅度提升。

### 硬编码
首先检查 FFmpeg 支持哪些硬编码格式，可以通过命令行或代码查询，FFmpeg 支持仅代表编译时添加了该模块，还需要检查系统硬件是否支持，或者直接调用之后等报错。
``` cmd
:: 查看编解码器支持
ffmpeg -codecs

:: 其中这两项表示 h264 与 h265 的编码器
DEV.LS h264    H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (decoders: h264 h264_qsv h264_cuvid ) (encoders: libx264 libx264rgb h264_amf h264_mf h264_nvenc h264_qsv nvenc nvenc_h264 )
DEV.L. hevc    H.265 / HEVC (High Efficiency Video Coding) (decoders: hevc hevc_qsv hevc_cuvid ) (encoders: libx265 nvenc_hevc hevc_amf hevc_mf hevc_nvenc hevc_qsv )
```

在代码调用上，使用 [avcodec_find_encoder_by_name](https://ffmpeg.org/doxygen/3.4/group__lavc__encoding.html#gaa614ffc38511c104bdff4a3afa086d37) 方法即可获取硬编码 AVCodec。
``` cpp
// 设置编码器 h264_qsv/hevc_qsv/h264_nvenc/hevc_nvenc
AVCodec* codec = avcodec_find_encoder_by_name("h264_qsv");
```

另外在设置 pts 与 dts 时有一些差异，软解码时设置 pts 即可，dts 会在编码是自动计算，所以在 AVFrame 中设置或是在 AVPacket 中设置都可以，但是硬编码却需要同时设置 pts 与 dts，所以硬解码必须在 AVPacket 中设置。

#### Intel qsv
##### 检查是否支持
如果在环境异常的系统中直接调用，会报以下错误：
``` cmd
[h264_qsv @ 000002b3d5e76540] Error initializing an internal MFX session: unsupported (-3)
FFmpeg Error Code:-40 Info:Function not implemented
```

首先检查：计算机管理 -> 设备管理器 -> 显示适配器 中是否包含英特尔核显
<img src="https://sadness96.github.io/images/blog/cpp-FFmpegHWEncoder/设备管理器.jpg"/>

如果未显示核显设备，首先去搜索以下自己的 Intel CPU 是否支持核显，以及是否支持英特尔 Quick Sync Video 技术，大部分 CPU 都是支持的，但是问题出现在了主板上，现在的主板大多在插入独显后会屏蔽核显，可以在 BIOS 中设置 Graphics Device 开启核显，但是我找了好几台电脑，很多主板甚至不提供这个选项。
如果成功开起了核显后就可以安装驱动了，我测试在安装了 Intel 驱动的电脑中可以直接开启 QSV 编码，但是似乎有人还是会报错，可以尝试安装 [Intel® Media SDK](https://www.intel.com/content/www/us/en/developer/tools/media-sdk/overview.html)，Intel 的注册流程过于麻烦，可以直接点击[下载](https://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/17861/MSDK2021R1.exe)，安装后尝试。

##### 测试耗时
找了好几台电脑，仅有的一台可以正确显示核显的电脑，CPU 的性能并不是很高，所以测试结果仅供参考，测试 7680 分辨率不支持编码，不清楚支持的最大分辨率是多少，没有在官方文档中找到相关资料。
* CPU：Intel i7 8750H
* 核显：Intel UHD Graphics 630
* 测试视频分辨率：3840 x 2160
* 测试编码耗时方法：[avcodec_send_frame](https://ffmpeg.org/doxygen/3.4/group__lavc__decoding.html#ga9395cb802a5febf1f00df31497779169)、[avcodec_receive_packet](https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga5b8eff59cf259747cf0b31563e38ded6)

| 编码方式 | 编码类型 | 耗时 |
| --- | --- | --- |
| cpu h264 | yuv420p | 13ms - 154ms |
| cpu h264 | nv12 | 11ms - 330ms |
| cpu h265 | yuv420p | 74ms - 1037ms |
| cpu h265 | nv12 | 不支持 |
| qsv h264 | yuv420p | 不支持 |
| qsv h264 | nv12 | 3ms - 15ms |
| qsv h265 | yuv420p | 不支持 |
| qsv h265 | nv12 | 22ms - 51ms |

#### NVIDIA nvenc
##### 检查是否支持
如果在环境异常的系统中直接调用，会报以下错误：
``` cmd
[h264_nvenc @ 000001c6a13a7f80] Driver does not support the required nvenc API version. Required: 11.1 Found: 11.0
[h264_nvenc @ 000001c6a13a7f80] The minimum required Nvidia driver for nvenc is (unknown) or newer
avcodec_open2 failed!Function not implemented
```

可能是所使用的英伟达显卡不是被支持的型号，也有可能是驱动程序需要更新，从 [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk) 中查找自己的显卡是否支持，并更新显卡驱动。

##### 测试耗时
nvenc 最大解码分辨率为 4096 x 4096。
* CPU：Intel i9 9900K
* 独显：NVIDIA GeForce RTX 2060
* 测试视频分辨率：3840 x 2160
* 测试编码耗时方法：[avcodec_send_frame](https://ffmpeg.org/doxygen/3.4/group__lavc__decoding.html#ga9395cb802a5febf1f00df31497779169)、[avcodec_receive_packet](https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga5b8eff59cf259747cf0b31563e38ded6)

| 编码方式 | 编码类型 | 耗时 |
| --- | --- | --- |
| cpu h264 | yuv420p | 4ms - 8ms |
| cpu h264 | nv12 | 4ms - 8ms |
| cpu h265 | yuv420p | 36ms - 551ms |
| cpu h265 | nv12 | 不支持 |
| nvenc h264 | yuv420p | 2ms - 4ms |
| nvenc h264 | nv12 | 1ms - 4ms |
| nvenc h265 | yuv420p | 2ms - 4ms |
| nvenc h265 | nv12 | 1ms - 4ms |

#### 完整代码
##### 参数变量
``` cpp
private:
	/// <summary>
	/// RTSP 标识
	/// </summary>
	string rtspJudgment_ = "rtsp";

	/// <summary>
	/// RTMP 标识
	/// </summary>
	string rtmpJudgment_ = "rtmp";

	/// <summary>
	/// 是否启用 TCP 优化解码
	/// </summary>
	bool is_tcp_decode_ = true;

	/// <summary>
	/// 是否多线程软解码
	/// </summary>
	bool is_thread_soft_decoding_ = true;

	/// <summary>
	/// 是否硬解码
	/// </summary>
	bool is_hard_decoding_ = true;

	/// <summary>
	/// 硬解码类型
	/// AV_HWDEVICE_TYPE_CUDA、AV_HWDEVICE_TYPE_DXVA2、AV_HWDEVICE_TYPE_D3D11VA
	/// </summary>
	int hw_type_ = AV_HWDEVICE_TYPE_CUDA;

	/// <summary>
	/// 编码帧率
	/// </summary>
	int fps_ = 25;

	/// <summary>
	/// 编码数据类型
	/// 硬解码为 AV_PIX_FMT_NV12、软解码为 AV_PIX_FMT_YUV420P
	/// </summary>
	AVPixelFormat pix_fmt_ = is_hard_decoding_ ? AV_PIX_FMT_NV12 : AV_PIX_FMT_YUV420P;
```

##### 解码 - 编码 - 推流
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

/// <summary>
/// 打印 FFmpeg 错误信息
/// </summary>
/// <param name="error">异常代码</param>
void PrintError(int error)
{
	char buf[1024] = { 0 };
	av_strerror(error, buf, sizeof(buf) - 1);
	printf("FFmpeg Error Code:%d Info:%s\n", error, buf);
}

void main()
{
	/// <summary>
	/// 视频路径
	/// </summary>
	string videoUrl_ = "rtsp://localhost:8554/live";

	/// <summary>
	/// 推流路径
	/// </summary>
	string output = "rtsp://localhost:8554/live2";
	//string output = "rtmp://localhost/live2";

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

	// TODO: 编码需要设置宽高，此处应根据实际需求设置
	int width_ = pCodecCtx->width;
	int height_ = pCodecCtx->height;

	// 创建输出流连接
	AVFormatContext* outputContext;
	string rtspJudgment = "rtsp";
	string rtmpJudgment = "rtmp";

	// 初始化 rtsp 连接
	if (output.rfind(rtspJudgment_, 0) == 0)
	{
		ret = avformat_alloc_output_context2(&outputContext, NULL, "rtsp", output.c_str());
		if (ret < 0)
		{
			av_log(NULL, AV_LOG_ERROR, "open output context failed\n");
		}
	}

	// 初始化 rtmp 连接
	if (output.rfind(rtmpJudgment_, 0) == 0)
	{
		int ret = avformat_alloc_output_context2(&outputContext, nullptr, "flv", output.c_str());
		if (ret < 0)
		{
			av_log(NULL, AV_LOG_ERROR, "open output context failed\n");
		}

		ret = avio_open2(&outputContext->pb, output.c_str(), AVIO_FLAG_READ_WRITE, nullptr, nullptr);
		if (ret < 0)
		{
			char buf[1024] = { 0 };
			av_strerror(ret, buf, sizeof(buf) - 1);
			cerr << buf << endl;

			av_log(NULL, AV_LOG_ERROR, "open avio failed");
		}
	}

	// 设置编码器 h264_qsv/hevc_qsv/h264_nvenc/hevc_nvenc
	AVCodec* codec = avcodec_find_encoder_by_name("h264_qsv");
	if (!codec)
	{
		cerr << "codec not find!" << endl;
		return;
	}

	AVCodecContext* codecContext = avcodec_alloc_context3(codec);
	if (!codecContext)
	{
		cerr << "avcodec_alloc_context3 failed!" << endl;
		return;
	}

	codecContext->codec_id = codec->id;
	codecContext->width = width_;
	codecContext->height = height_;
	codecContext->time_base = { 1, fps_ };
	codecContext->pix_fmt = pix_fmt_;

	ret = avcodec_open2(codecContext, codec, NULL);
	if (ret != 0)
	{
		char buf[1024] = { 0 };
		av_strerror(ret, buf, sizeof(buf) - 1);
		cerr << "avcodec_open2 failed!" << buf << endl;
		return;
	}
	cout << "avcodec_open2 success!" << endl;

	AVCodecParameters* pa = avcodec_parameters_alloc();
	pa->codec_type = AVMEDIA_TYPE_VIDEO;
	pa->codec_id = codec->id;
	pa->width = width_;
	pa->height = height_;

	// RTMP 需要写入设置 sps pps
	if (output.find(rtmpJudgment) != string::npos)
	{
		unsigned char sps_pps[23] = { 0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0a, 0xf8, 0x0f, 0x00, 0x44, 0xbe, 0x8, 0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80 };
		pa->extradata_size = 23;
		pa->extradata = (uint8_t*)av_malloc(23 + AV_INPUT_BUFFER_PADDING_SIZE);
		if (pa->extradata == NULL) {
			printf("could not av_malloc the video params extradata!\n");
			return;
		}
		memcpy(pa->extradata, sps_pps, 23);
	}

	AVStream* stream = avformat_new_stream(outputContext, codec);
	if (stream == NULL) {
		av_log(NULL, AV_LOG_ERROR, "avformat_new_stream is null");
		return;
	}

	ret = avcodec_parameters_copy(stream->codecpar, pa);
	if (ret < 0)
	{
		av_log(NULL, AV_LOG_ERROR, "copy coddec context failed");
	}

	ret = avformat_write_header(outputContext, NULL);
	if (ret < 0)
	{
		av_log(NULL, AV_LOG_ERROR, "format write header failed");
	}
	else
	{
		av_log(NULL, AV_LOG_INFO, " Open output file success %s\n", output.c_str());
	}

	// 解码帧
	AVPacket* packet;
	packet = (AVPacket*)av_malloc(sizeof(AVPacket));
	AVFrame* frame, * hw_frame;
	frame = av_frame_alloc();
	hw_frame = av_frame_alloc();

	AVPacket* packetEX = (AVPacket*)av_malloc(sizeof(AVPacket));
	int pts = 0;

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

				ret = avcodec_send_frame(codecContext, pframe);
				if (ret < 0)
				{
					char buf[1024] = { 0 };
					av_strerror(ret, buf, sizeof(buf) - 1);
					cerr << "avcodec_send_frame failed!" << buf << endl;
					av_frame_free(&pframe);
					return;
				}
				while (ret >= 0)
				{
					ret = avcodec_receive_packet(codecContext, packetEX);
					if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
					{
						break;
					}
					if (ret < 0)
					{
						char buf[1024] = { 0 };
						av_strerror(ret, buf, sizeof(buf) - 1);
						cerr << "avcodec_send_frame failed!" << buf << endl;
						break;
					}

					// 设置 pts 与 dts 推流视频
					packetEX->pts = packetEX->dts = pts += av_rescale_q(1000.0f / 25, { 1, 1000 }, outputContext->streams[0]->time_base);

					if (av_interleaved_write_frame(outputContext, packetEX) >= 0)
					{
						//cout << "WritePacket Success!" << endl;
					}
					else if (ret < 0)
					{
						cout << "WritePacket failed! ret = " << ret << endl;
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