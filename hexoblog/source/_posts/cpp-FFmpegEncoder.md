---
title: FFmpeg 编码视频
date: 2022-08-20 19:38:58
tags: [c++,ffmpeg]
categories: C++
---
### 使用 FFmpeg 编码视频并推流或保存文件
<!-- more -->
#### 简介
[FFmpeg](https://ffmpeg.org/) 是一个完整的跨平台解决方案，用于录制、转换和流式传输音频和视频。
在处理 FFmpeg 编码视频前需了解 [FFmpeg 解码视频](https://sadness96.github.io/blog/2021/11/08/cpp-FFmpegDecoder/) 与 [FFmpeg 转发推流到 rtsp/rtmp](https://sadness96.github.io/blog/2022/07/11/cpp-FFmpegPushCurrent/)，此文以编码推流为目的，当然也可以编码后储存为文件或作为其他作用。
设置解码方式：TCP 优化、软解码（多线程）、硬解码（CUDA、DXVA2、D3D11VA）
设置编码器：[H.264](https://zh.wikipedia.org/wiki/H.264/MPEG-4_AVC) 、 [H.265](https://zh.wikipedia.org/wiki/%E9%AB%98%E6%95%88%E7%8E%87%E8%A7%86%E9%A2%91%E7%BC%96%E7%A0%81)
设置推流方式：[RTSP](https://zh.wikipedia.org/wiki/%E5%8D%B3%E6%99%82%E4%B8%B2%E6%B5%81%E5%8D%94%E5%AE%9A) 、 [RTMP](https://zh.wikipedia.org/wiki/%E5%AE%9E%E6%97%B6%E6%B6%88%E6%81%AF%E5%8D%8F%E8%AE%AE)

#### 核心代码
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
	/// 编码器 ID
	/// AV_CODEC_ID_H264、AV_CODEC_ID_HEVC
	/// 测试 RTMP 推流不支持 H265 编码
	/// 测试 H265 解码不支持硬编码类型 NV12
	/// </summary>
	AVCodecID codec_id_ = AV_CODEC_ID_H264;

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

	if (output.rfind(rtspJudgment, 0) == 0)
	{
		// 初始化 rtsp 连接
		ret = avformat_alloc_output_context2(&outputContext, NULL, "rtsp", output.c_str());
		if (ret < 0)
		{
			av_log(NULL, AV_LOG_ERROR, "open output context failed\n");
		}
	}
	else if (output.rfind(rtmpJudgment, 0) == 0)
	{
		// 初始化 rtmp 连接
		int ret = avformat_alloc_output_context2(&outputContext, nullptr, "flv", output.c_str());
		if (ret < 0)
		{
			av_log(NULL, AV_LOG_ERROR, "open output context failed\n");
		}

		ret = avio_open2(&outputContext->pb, output.c_str(), AVIO_FLAG_READ_WRITE, nullptr, nullptr);
		if (ret < 0)
		{
			PrintError(ret);
			av_log(NULL, AV_LOG_ERROR, "open avio failed");
		}
	}
	else
	{
		// 判断文件夹是否合法
		string outDir = output.substr(0, output.find_last_of("\\") + 1);
		if (strlen(outDir.c_str()) > MAX_PATH)
		{
			cerr << "Maximum path length exceeded!" << endl;
			return;
		}

		// 文件夹不存在则创建
		int ipathLength = strlen(outDir.c_str());
		int ileaveLength = 0;
		int iCreatedLength = 0;
		char szPathTemp[MAX_PATH] = { 0 };
		for (int i = 0; (NULL != strchr(outDir.c_str() + iCreatedLength, '\\')); i++)
		{
			ileaveLength = strlen(strchr(outDir.c_str() + iCreatedLength, '\\')) - 1;
			iCreatedLength = ipathLength - ileaveLength;
			strncpy(szPathTemp, outDir.c_str(), iCreatedLength);
			if (access(szPathTemp, 0))
			{
				if (mkdir(szPathTemp))
				{
					cerr << "mkdir " << szPathTemp << " false, errno:" << errno << " errmsg:" << strerror(errno) << endl;
					return;
				}
			}
		}
		if (iCreatedLength < ipathLength)
		{
			if (access(outDir.c_str(), 0))
			{
				if (mkdir(outDir.c_str()))
				{
					cerr << "mkdir " << outDir << " false, errno:" << errno << " errmsg:" << strerror(errno) << endl;
					return;
				}
			}
		}

		// 初始化文件连接
		ret = avformat_alloc_output_context2(&outputContext, NULL, NULL, output.c_str());
		if (ret < 0)
		{
			av_log(NULL, AV_LOG_ERROR, "open output context failed\n");
		}

		ret = avio_open2(&outputContext->pb, output.c_str(), AVIO_FLAG_READ_WRITE, nullptr, nullptr);
		if (ret < 0)
		{
			PrintError(ret);
			av_log(NULL, AV_LOG_ERROR, "open avio failed");
		}
	}

	// 设置编码器 AV_CODEC_ID_H264 AV_CODEC_ID_HEVC
	AVCodec* codec = avcodec_find_encoder(codec_id_);
	if (!codec)
	{
		cerr << "codec not find!" << endl;
		return;
	}
	AVStream* stream = avformat_new_stream(outputContext, codec);

	AVCodecContext* codecContext = avcodec_alloc_context3(codec);
	if (!codecContext)
	{
		cerr << "avcodec_alloc_context3 failed!" << endl;
		return;
	}

	codecContext->codec_id = codec_id_;
	codecContext->width = width_;
	codecContext->height = height_;
	codecContext->time_base = { 1, fps_ };
	codecContext->pix_fmt = pix_fmt_;

	ret = avcodec_open2(codecContext, codec, NULL);
	if (ret != 0)
	{
		PrintError(ret);
		return;
	}
	cout << "avcodec_open2 success!" << endl;

	AVCodecParameters* pa = avcodec_parameters_alloc();
	pa->codec_type = AVMEDIA_TYPE_VIDEO;
	pa->codec_id = codec_id_;
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
	clock_t startPts = clock();
	clock_t stopPts;
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

				// 设置 pts 推流视频
				stopPts = clock();
				auto setPtsMs = stopPts - startPts;
				pframe->pts = pts += av_rescale_q(setPtsMs, { 1, 1000 }, outputContext->streams[0]->time_base);
				cout << "setPtsMs:" << setPtsMs << " - pts:" << pframe->pts << endl;
				startPts = stopPts;

				ret = avcodec_send_frame(codecContext, pframe);
				if (ret < 0)
				{
					PrintError(ret);
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
						PrintError(ret);
						break;
					}

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
	
	// 释放输入 AVFormatContext
	avformat_close_input(&inputContext);

	// 写入文件尾
	if (av_write_trailer(outputContext) < 0)
	{
		av_log(NULL, AV_LOG_ERROR, "format write trailer failed");
	}

	// 释放 AVCodecContext
	avcodec_free_context(&pCodecCtx);
	avcodec_free_context(&codecContext);

	// 释放输出 AVFormatContext
	avformat_close_input(&outputContext);
}
```

#### 注意事项
##### 不支持的内容
* 测试 RTMP 推流不支持 H265 编码，似乎可以重新编译 FFmpeg 来支持。
* 测试 H265 解码不支持硬编码类型 NV12。

##### 编码帧率
编码推流视频会根据帧率显示，常见的帧率：
* 电影 24fps
* 监控 25fps
* 普通视频 30fps/60fps

##### 推流到 RTSP / RTMP
1. 创建 RTSP / RTMP 流需要通过 [avformat_alloc_output_context2](https://ffmpeg.org/doxygen/3.0/avformat_8h.html#a6ddf3d982feb45fa5081420ee911f5d5) 创建 "rtsp" / "flv" 上下文。
1. 创建 RTMP 流需要创建并初始化一个 [AVIOContext](https://ffmpeg.org/doxygen/trunk/structAVIOContext.html) 以访问 url 指示的资源。
1. 创建 RTMP 流需要写入流标头前写入 sps pps，此处没有验证具体含义，可以使用其他方式写入，但是测试时对各类视频没有影响。
	``` cpp
	unsigned char sps_pps[23] = { 0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0a, 0xf8, 0x0f, 0x00, 0x44, 0xbe, 0x8, 0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80 };
	pa->extradata_size = 23;
	pa->extradata = (uint8_t*)av_malloc(23 + AV_INPUT_BUFFER_PADDING_SIZE);
	if (pa->extradata == NULL) {
		printf("could not av_malloc the video params extradata!\n");
		return;
	}
	memcpy(pa->extradata, sps_pps, 23);
	```

1. 使用 [avformat_write_header](https://ffmpeg.org/doxygen/3.3/group__lavf__encoding.html#ga18b7b10bb5b94c4842de18166bc677cb) 写入流标头。

##### 保存到本地文件
测试保存本地文件支持的文件格式有：mp4、flv、mov、ts、avi。
1. 保存到本地文件首先需要判断文件路径的可用，如果文件夹为空时自动创建。
	``` cpp
	// 判断文件夹是否合法
	string outDir = output.substr(0, output.find_last_of("\\") + 1);
	if (strlen(outDir.c_str()) > MAX_PATH)
	{
		cerr << "Maximum path length exceeded!" << endl;
		return;
	}

	// 文件夹不存在则创建
	int ipathLength = strlen(outDir.c_str());
	int ileaveLength = 0;
	int iCreatedLength = 0;
	char szPathTemp[MAX_PATH] = { 0 };
	for (int i = 0; (NULL != strchr(outDir.c_str() + iCreatedLength, '\\')); i++)
	{
		ileaveLength = strlen(strchr(outDir.c_str() + iCreatedLength, '\\')) - 1;
		iCreatedLength = ipathLength - ileaveLength;
		strncpy(szPathTemp, outDir.c_str(), iCreatedLength);
		if (access(szPathTemp, 0))
		{
			if (mkdir(szPathTemp))
			{
				cerr << "mkdir " << szPathTemp << " false, errno:" << errno << " errmsg:" << strerror(errno) << endl;
				return;
			}
		}
	}
	if (iCreatedLength < ipathLength)
	{
		if (access(outDir.c_str(), 0))
		{
			if (mkdir(outDir.c_str()))
			{
				cerr << "mkdir " << outDir << " false, errno:" << errno << " errmsg:" << strerror(errno) << endl;
				return;
			}
		}
	}
	```
1. 保存到本地文件需要通过 [avformat_alloc_output_context2](https://ffmpeg.org/doxygen/3.0/avformat_8h.html#a6ddf3d982feb45fa5081420ee911f5d5) 创建 NULL 上下文即可，FFmpeg 可以通过文件路径中的后缀名自动创建类型。
1. 创建并初始化一个 AVIOContext 以访问 url 指示的资源。
1. 使用 [avformat_write_header](https://ffmpeg.org/doxygen/3.3/group__lavf__encoding.html#ga18b7b10bb5b94c4842de18166bc677cb) 写入流标头。
1. 与推流不同，在写入本地视频的结尾，需要使用 [av_write_trailer](https://ffmpeg.org/doxygen/3.4/group__lavf__encoding.html#ga7f14007e7dc8f481f054b21614dfec13) 写入流尾并释放数据，否则会对一些格式造成一些影响，例如： mp4 格式无法播放，flv 格式无法正确显示时间轴。

##### 设置 pts
推流到流媒体根据目标类型需要有不同的设置。
* 推流到 RTSP：time_base 默认为 90000，pts 平均以 3600 递增。
* 推流到 RTMP：time_base 默认为 1000，pts 平均以 40 递增。

由于编码是多线程，测试时以当前编码间隔计算 pts，使用 [av_rescale_q](https://www.ffmpeg.org/doxygen/0.6/mathematics_8c.html#32ddc164f0fd4972907f68c12fcdbd89) 从间隔时间戳转换到 pts，有效防止计算溢出的情况。
``` cpp
clock_t startPts = clock();
clock_t stopPts;
int pts = 0;
...
// 循环编码
while (true)
{
    // 设置 pts 推流视频
    stopPts = clock();
    auto setPtsMs = stopPts - startPts;
    pframe->pts = pts += av_rescale_q(setPtsMs, { 1, 1000 }, outputContext->streams[0]->time_base);
    cout << "setPtsMs:" << setPtsMs << " - pts:" << pframe->pts << endl;
    startPts = stopPts;
}
```