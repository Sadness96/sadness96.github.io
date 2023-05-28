---
title: FFmpeg 转发推流到 rtsp/rtmp
date: 2022-07-11 22:14:25
tags: [c++,ffmpeg]
categories: C++
---
### 使用 FFmpeg 以代码的方式分别转发推流到 rtsp/rtmp
<!-- more -->
#### 简介
转发任意支持的格式，推流到 rtsp/rtmp 的最简代码，数据源为文件的话需要额外添加 FPS 控制。
使用流媒体服务为： [rtsp-simple-server](https://github.com/aler9/rtsp-simple-server)
``` cmd
:: rtsp-simple-server 提供的推流命令

:: 推流到 RTSP
ffmpeg -re -stream_loop -1 -i C:\Video.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live

:: 推流到 RTMP
ffmpeg -re -stream_loop -1 -i C:\Video.mp4 -c copy -f flv rtmp://localhost/live
```

#### 代码
``` cpp
void av_packet_rescale_ts(AVPacket* pkt, AVRational src_tb, AVRational dst_tb)
{
	if (pkt->pts != AV_NOPTS_VALUE)
		pkt->pts = av_rescale_q(pkt->pts, src_tb, dst_tb);
	if (pkt->dts != AV_NOPTS_VALUE)
		pkt->dts = av_rescale_q(pkt->dts, src_tb, dst_tb);
	if (pkt->duration > 0)
		pkt->duration = av_rescale_q(pkt->duration, src_tb, dst_tb);
}

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

int main()
{
	av_register_all();
	avformat_network_init();

	string input = "rtsp://localhost:8554/live";
	//string output = "rtsp://localhost:8554/live2";
	string output = "rtmp://localhost/live2";

	// 创建输入流连接
	AVFormatContext* inputContext = avformat_alloc_context();
	int ret = avformat_open_input(&inputContext, input.c_str(), NULL, NULL);
	if (ret < 0)
	{
		av_log(NULL, AV_LOG_ERROR, "Input file open input failed\n");
		return  ret;
	}
	ret = avformat_find_stream_info(inputContext, NULL);
	if (ret < 0)
	{
		av_log(NULL, AV_LOG_ERROR, "Find input file stream inform failed\n");
	}
	else
	{
		av_log(NULL, AV_LOG_INFO, "Open input file  %s success\n", input.c_str());
	}

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
		ret = avformat_alloc_output_context2(&outputContext, NULL, "flv", output.c_str());
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

	for (int i = 0; i < inputContext->nb_streams; i++)
	{
		AVStream* stream = avformat_new_stream(outputContext, inputContext->streams[i]->codec->codec);
		ret = avcodec_copy_context(stream->codec, inputContext->streams[i]->codec);
		if (ret < 0)
		{
			av_log(NULL, AV_LOG_ERROR, "copy coddec context failed");
		}
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

	// 转发数据流
	AVPacket* packet = (AVPacket*)av_malloc(sizeof(AVPacket));
	while (true)
	{
		ret = av_read_frame(inputContext, packet);
		if (ret < 0) {
			av_free_packet(packet);
			return -1;
		}

		auto inputStream = inputContext->streams[packet->stream_index];
		auto outputStream = outputContext->streams[packet->stream_index];
		av_packet_rescale_ts(packet, inputStream->time_base, outputStream->time_base);
		if (av_interleaved_write_frame(outputContext, packet) >= 0)
		{
			cout << "WritePacket Success!" << endl;
		}
		else if (ret < 0)
		{
			cout << "WritePacket failed! ret = " << ret << endl;
		}

		av_free_packet(packet);
	}
	return 0;
}
```