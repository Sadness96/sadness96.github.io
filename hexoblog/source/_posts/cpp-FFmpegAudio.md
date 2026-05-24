---
title: FFmpeg 音频播放
date: 2026-04-16 21:16:40
tags: [c++,ffmpeg]
categories: C++
---
### 使用 FFmpeg 解码音频，使用 WASAPI 播放
<!-- more -->
### 简介
[FFmpeg 解码视频](/blog/2021/11/08/cpp-FFmpegDecoder/) 用于解码音视频包，增加音频播放线程，使用 WASAPI 播放音频。
[WASAPI (Windows Audio Session API)](https://learn.microsoft.com/zh-cn/windows/win32/coreaudio/wasapi) 让客户端应用程序能够管理应用程序与音频终结点设备之间的音频数据流。

### 核心代码
#### PacketQueue.h
需要封装一个可以阻塞等待的队列
``` cpp
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>

class PacketQueue {
public:
	explicit PacketQueue(int max_size = 30)
		: max_size_(max_size), aborted_(false)
	{
	}

	// 生产者：满了就阻塞，直到有空间
	void push(AVPacket* pkt)
	{
		std::unique_lock<std::mutex> lk(mtx_);
		cond_push_.wait(lk, [this] {
			return (int)q_.size() < max_size_ || aborted_;
			});
		if (aborted_) { av_packet_free(&pkt); return; }
		q_.push(pkt);
		cond_pop_.notify_one();
	}

	// 消费者：空了就阻塞
	bool pop(AVPacket*& pkt, int timeout_ms = 100)
	{
		std::unique_lock<std::mutex> lk(mtx_);
		if (!cond_pop_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
			[this] { return !q_.empty() || aborted_; }))
			return false;
		if (aborted_ && q_.empty()) return false;
		pkt = q_.front(); q_.pop();
		cond_push_.notify_one();
		return true;
	}

	void abort()
	{
		std::unique_lock<std::mutex> lk(mtx_);
		aborted_ = true;
		cond_push_.notify_all();
		cond_pop_.notify_all();
	}

	void reset()
	{
		std::unique_lock<std::mutex> lk(mtx_);
		aborted_ = false;
		// 清理残留数据
		while (!q_.empty()) {
			AVPacket* p = q_.front(); q_.pop();
			av_packet_free(&p);
		}
	}

	void clear()
	{
		std::unique_lock<std::mutex> lk(mtx_);
		while (!q_.empty()) {
			AVPacket* p = q_.front(); q_.pop();
			av_packet_free(&p);
		}
		cond_push_.notify_all();
	}

	int size()
	{
		std::unique_lock<std::mutex> lk(mtx_);
		return (int)q_.size();
	}

	bool empty()
	{
		std::unique_lock<std::mutex> lk(mtx_);
		return q_.empty();
	}

private:
	std::queue<AVPacket*>   q_;
	std::mutex              mtx_;
	std::condition_variable cond_push_;
	std::condition_variable cond_pop_;
	int  max_size_;
	bool aborted_;
};
```

#### WasapiPlayer.h
WASAPI 音频核心代码
``` cpp
#include <mmdeviceapi.h>   
#include <audioclient.h> 
#include <avrt.h>          
#include <endpointvolume.h>  

extern "C"
{
#pragma comment(lib, "avrt.lib")
#pragma comment(lib, "ole32.lib")
}

/// <summary>
/// 音频时钟（主同步时钟）
/// </summary>
struct AudioClock
{
	// 当前音频实际播出位置（秒）
	atomic<double> pts{ 0.0 };
	// 时钟是否已开始计时
	atomic<bool> valid{ false };

	void set(double p) { pts.store(p); valid.store(true); }
	double get() const { return pts.load(); }
};

/// <summary>
/// WASAPI 音频
/// </summary>
class WasapiPlayer
{
public:
	// 设备枚举器
	IMMDeviceEnumerator* pEnum = nullptr;
	// 默认音频输出设备
	IMMDevice* pDevice = nullptr;
	// 音频客户端（流管理）
	IAudioClient* pClient = nullptr;
	// 渲染客户端（写 PCM 数据）
	IAudioRenderClient* pRender = nullptr;
	// 音量控制接口
	IAudioEndpointVolume* pVolume = nullptr;

	// 音频格式描述
	WAVEFORMATEX wfx{};
	// 硬件缓冲区总帧数
	UINT32 bufferFrames = 0;
	bool initialized = false;

    // reqBufferMs: requested buffer size in milliseconds (default 200ms)
	bool init(int sampleRate, int channels, int reqBufferMs = 200)
	{
		CoInitializeEx(nullptr, COINIT_MULTITHREADED);

		if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
			CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), (void**)&pEnum)))
			return false;
		if (FAILED(pEnum->GetDefaultAudioEndpoint(eRender, eConsole, &pDevice)))
			return false;
		if (FAILED(pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&pClient)))
			return false;

		wfx.wFormatTag = WAVE_FORMAT_PCM;
		wfx.nChannels = (WORD)channels;
		wfx.nSamplesPerSec = (DWORD)sampleRate;
		wfx.wBitsPerSample = 16;
		wfx.nBlockAlign = wfx.nChannels * wfx.wBitsPerSample / 8;
		wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;

        // requested buffer in 100-ns units: ms * 10000
		REFERENCE_TIME hnsReq = (REFERENCE_TIME)reqBufferMs * 10000; // default 200ms
		if (FAILED(pClient->Initialize(AUDCLNT_SHAREMODE_SHARED,
			AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
			hnsReq, 0, &wfx, nullptr)))
			return false;

		pClient->GetBufferSize(&bufferFrames);
		if (FAILED(pClient->GetService(__uuidof(IAudioRenderClient), (void**)&pRender)))
			return false;

		// 获取音量控制接口（可选，失败不影响播放）
		pDevice->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_ALL, nullptr, (void**)&pVolume);

		pClient->Start();
		initialized = true;
		return true;
	}

	// 写入 PCM s16 数据，返回实际写入帧数
	int write(const uint8_t* data, int frames)
	{
		if (!initialized) return 0;
		UINT32 padding = 0;
		pClient->GetCurrentPadding(&padding);
		UINT32 avail = bufferFrames - padding;
		UINT32 toWrite = min((UINT32)frames, avail);
		if (toWrite == 0) return 0;

		BYTE* buf = nullptr;
		if (FAILED(pRender->GetBuffer(toWrite, &buf))) return 0;
		memcpy(buf, data, toWrite * wfx.nBlockAlign);
		pRender->ReleaseBuffer(toWrite, 0);
		return (int)toWrite;
	}

	// 获取缓冲区中尚未播出的帧数（用于时钟补偿）
	double getBufferedSeconds() const
	{
		if (!initialized) return 0.0;
		UINT32 padding = 0;
		pClient->GetCurrentPadding(&padding);
		return (double)padding / (double)wfx.nSamplesPerSec;
	}

	// 设置音量，vol 范围 0.0（静音）~ 1.0（最大）
	void setVolume(float vol)
	{
		if (pVolume) {
			vol = max(0.0f, min(1.0f, vol));
			pVolume->SetMasterVolumeLevelScalar(vol, nullptr);
		}
	}

	// 获取当前音量 0.0 ~ 1.0
	float getVolume() const
	{
		float vol = 1.0f;
		if (pVolume) pVolume->GetMasterVolumeLevelScalar(&vol);
		return vol;
	}

	~WasapiPlayer()
	{
		if (pClient) { pClient->Stop(); pClient->Release(); }
		if (pRender) pRender->Release();
		if (pVolume) pVolume->Release();
		if (pDevice) pDevice->Release();
		if (pEnum)   pEnum->Release();
		CoUninitialize();
	}
};
``` 

#### 调用音频
##### 头文件
``` cpp
#include "PacketQueue.h"
#include "WasapiPlayer.h"

extern "C"
{
#include <libswresample/swresample.h>
}

/// <summary>
/// 音频解码上下文
/// </summary>
AVCodecContext* pAudioCodecCtx_ = NULL;

/// <summary>
/// WASAPI 音频
/// </summary>
WasapiPlayer* pWasapiPlayer_ = NULL;

/// <summary>
/// 音频时钟（主同步时钟）
/// </summary>
AudioClock audioClock;

/// <summary>
/// 音频流下标
/// </summary>
int audio_stream_index_ = -1;

/// <summary>
/// 音频 TimeBase
/// </summary>
AVRational audio_time_base_;

/// <summary>
/// 音频 AVPacket 队列
/// </summary>
PacketQueue audio_packets_;

/// <summary>
/// 多线程解码音频
/// </summary>
thread audio_thread_;

/// <summary>
/// 关闭音频解码
/// </summary>
/// <returns></returns>
bool close_audio_decoder();

/// <summary>
/// 多线程解码视频
/// </summary>
void video_thread_func();
``` 

##### 获取音频媒体流
``` cpp
// 获取音频媒体流
AVStream* audioStream = NULL;
...
if (inputContext_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
{
    audioStream = inputContext_->streams[i];
    audio_stream_index_ = audioStream->index;
    audio_time_base_ = inputContext_->streams[audio_stream_index_]->time_base;
}
...
if (audioStream != NULL && audio_stream_index_ >= 0)
{
    AVCodec* aCodec = avcodec_find_decoder(audioStream->codecpar->codec_id);
    pAudioCodecCtx_ = avcodec_alloc_context3(aCodec);
    ret = avcodec_parameters_to_context(pAudioCodecCtx_, audioStream->codecpar);
    if (ret < 0)
    {
        avcodec_free_context(&pAudioCodecCtx_);
    }

    if (avcodec_open2(pAudioCodecCtx_, aCodec, nullptr))
    {
        avcodec_free_context(&pAudioCodecCtx_);
    }

    pWasapiPlayer_ = new WasapiPlayer();
	// use smaller WASAPI buffer (200ms) to reduce latency for volume changes
	if (!pWasapiPlayer_->init(44100, 2, 200))
    {
        printf("WASAPI init failed, playing without audio!\n");
        delete pWasapiPlayer_;
        pWasapiPlayer_ = nullptr;
    }

    // 启动解码音频线程
    if (!audio_thread_running_)
    {
        audio_thread_running_ = true;
        audio_thread_ = thread(&FFmpegDecoder::audio_thread_func, this);
    }
}
```

##### 传入音频包
``` cpp
...
if (packet && packet->stream_index == audio_stream_index_)
{
    // 音频流
    if (audio_thread_running_)
    {
        AVPacket* copy_packet = av_packet_alloc();
        av_packet_ref(copy_packet, packet);
        audio_packets_.push(copy_packet);
    }
}
...
```

##### 多线程音频播放
``` cpp
/// <summary>
/// 多线程音频播放
/// </summary>
void FFmpegDecoder::audio_thread_func()
{
	thread::id tid = this_thread::get_id();
	unsigned int threadId = *(unsigned int*)&tid;
	printf("Start audio thread! Index:%d ThreadID:%d\n", videoIndex_, threadId);

	SwrContext* swr = swr_alloc_set_opts(nullptr,
		av_get_default_channel_layout(2), AV_SAMPLE_FMT_S16, 44100,
		pAudioCodecCtx_->channel_layout ? pAudioCodecCtx_->channel_layout
		: av_get_default_channel_layout(pAudioCodecCtx_->channels),
		pAudioCodecCtx_->sample_fmt, pAudioCodecCtx_->sample_rate,
		0, nullptr);
	swr_init(swr);

	AVFrame* frame = av_frame_alloc();
	double playedSamples = 0.0;
	double sampleRate = 44100.0;

	while (audio_thread_running_)
	{
		AVPacket* packet = nullptr;
		if (!audio_packets_.pop(packet, 100))
			continue;

		// 开启录像后缓存数据到队列
		if (videotape_thread_running_ && videotape_packets_.size() < cacheNumber_)
		{
			if (videotape_thread_running_)
			{
				AVPacket* copy_packet = av_packet_alloc();
				av_packet_ref(copy_packet, packet);
				videotape_packets_.push(copy_packet);
			}
		}

		avcodec_send_packet(pAudioCodecCtx_, packet);
		av_packet_free(&packet);

		if (!frame)
		{
			// 重连后重新分配
			frame = av_frame_alloc();
		}
		while (avcodec_receive_frame(pAudioCodecCtx_, frame) == 0)
		{
			// 计算 pts（秒）
			double pts = (frame->pts != AV_NOPTS_VALUE)
				? frame->pts * av_q2d(audio_time_base_)
				: playedSamples / sampleRate;

			// 重采样到 s16 stereo 44100
			int outSamples = av_rescale_rnd(
				swr_get_delay(swr, pAudioCodecCtx_->sample_rate) + frame->nb_samples,
				44100, pAudioCodecCtx_->sample_rate, AV_ROUND_UP);

			std::vector<uint8_t> outBuf(outSamples * 2 * 2); // stereo s16
			uint8_t* outPtr = outBuf.data();
			int converted = swr_convert(swr, &outPtr, outSamples,
				(const uint8_t**)frame->data, frame->nb_samples);
			if (converted <= 0) continue;

			// 写入 WASAPI（可能需要多次写）
			int written = 0;
			while (written < converted && audio_thread_running_)
			{
				int w = pWasapiPlayer_->write(outPtr + written * 4, converted - written);
				if (w == 0) { Sleep(1); continue; }
				written += w;
			}

			playedSamples += converted;
			// 更新音频时钟：减去 WASAPI 缓冲区中尚未播出的时长，得到实际播出位置
			double buffered = pWasapiPlayer_->getBufferedSeconds();
			audioClock.set(pts + (double)converted / sampleRate - buffered);
		}
	}

	av_frame_free(&frame);
}
```

##### 释放资源
``` cpp
/// <summary>
/// 关闭音频解码
/// </summary>
/// <returns></returns>
bool FFmpegDecoder::close_audio_decoder()
{
	if (pAudioCodecCtx_)
	{
		int ret = avcodec_close(pAudioCodecCtx_);
		if (ret < 0)
		{
			printf("Failed to audio close codec\n");
			PrintError(ret);
			return false;
		}

		avcodec_free_context(&pAudioCodecCtx_);
		pAudioCodecCtx_ = NULL;
	}
}

/// <summary>
/// 视频停止
/// </summary>
/// <returns>是否停止成功</returns>
bool FFmpegDecoder::VideoStop()
{
    ...
    // 停止解码音频线程
	audio_packets_.abort();
	audio_thread_running_ = false;
	if (audio_thread_.joinable())
	{
		audio_thread_.join();
	}
	audio_packets_.clear();
    ...
}
```