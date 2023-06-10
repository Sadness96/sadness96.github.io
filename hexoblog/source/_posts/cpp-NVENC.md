---
title: NVENC 硬编码视频
date: 2023-05-05 00:50:40
tags: [c++,ffmpeg,cuda,nvidia]
categories: C++
---
### 使用 NVENC 硬编码视频
<!-- more -->
### 简介
NVENC 是 [NVIDIA 视频编解码器](https://developer.nvidia.com/video-codec-sdk)中的编码部分，使用硬件加速视频编码，官方[参考文档](https://docs.nvidia.com/video-technologies/index.html)。

### 测试编码能力
* H264 编码最大分辨率为 4096 x 4096。
* H265 编码最大分辨率为 8192 x 8192。
* NVENC NV12 编码视频 H264，不支持推流到 RTMP。
* NVENC NV12 编码视频 H265，不支持推流到 RTSP、RTMP、UDP，所以选用推流到 TCP，使用 TCP 转发到 RTSP 的方式。

### 构建代码
#### 构建环境
-   Windows 10
-   [Visual Studio 2019](https://visualstudio.microsoft.com/zh-hans/)
-   [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
-   [CMake 3.17.2](https://github.com/Kitware/CMake/releases/tag/v3.17.2)

#### 构建步骤
1. 下载 [NVIDIA VIDEO CODEC SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download)
2. 使用 CMake 构建项目 ./Video_Codec_SDK_12.0.16/Samples
3. 启动构建好的 NvCodec.sln
4. 本文基于工程中 AppEncCuda 项目开发

### 配置参数
#### 编码数据类型
设置编码输入格式，指编码数据的类型。
``` cpp
typedef enum _NV_ENC_BUFFER_FORMAT
{
	// NVDEC 解码默认格式，介绍说存储格式与 YUV420P 相同，但是测试 UV 分量偏移。
	NV_ENC_BUFFER_FORMAT_IYUV,
	// YUV NV12 格式，可以使用 FFmpeg CUDA 硬解码直接使用。
	NV_ENC_BUFFER_FORMAT_NV12
} NV_ENC_BUFFER_FORMAT;
```

#### 编码器类型预设参数
设置编码器类型与预设参数，修改 NvEncoderCLIOptions.h 文件，在 NvEncoderInitParam 构造函数中增加配置。
1. 编码器类型支持 H264 或 H265。
2. 编码器预设参数，从 P1 到 P7，性能下降，质量提高。默认情况下，H264 的预设 P3 至 P7 和 HEVC 的预设 P2 至 P7。
``` cpp
	NvEncoderInitParam(const char* szParam = "",
		std::function<void(NV_ENC_INITIALIZE_PARAMS* pParams)>* pfuncInit = NULL, bool _bLowLatency = false)
		: strParam(szParam), bLowLatency(_bLowLatency)
		, guidCodec(NV_ENC_CODEC_HEVC_GUID)	// 编码器类型 NV_ENC_CODEC_H264_GUID、NV_ENC_CODEC_HEVC_GUID
		, guidPreset(NV_ENC_PRESET_P3_GUID)	// 编码器预设参数 NV_ENC_PRESET_P1_GUID - NV_ENC_PRESET_P7_GUID
```

### 代码
#### 初始化 NVENC
``` cpp
/// <summary>
/// Cuda 上下文连接
/// </summary>
CUcontext cuContext_ = NULL;

/// <summary>
/// NVENC 视频编码器
/// </summary>
unique_ptr<NvEncoderCuda> pEnc_;
```

``` cpp
NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_NV12;
int iGpu = 0;

NvEncoderInitParam encodeCLIOptions = NvEncoderInitParam("", NULL, true);

ck(cuInit(0));
int nGpu = 0;
ck(cuDeviceGetCount(&nGpu));
if (iGpu < 0 || iGpu >= nGpu)
{
	cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << endl;
	return;
}
CUdevice cuDevice = 0;
ck(cuDeviceGet(&cuDevice, iGpu));
char szDeviceName[80];
ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
cout << "GPU in use: " << szDeviceName << endl;
ck(cuCtxCreate(&cuContext_, 0, cuDevice));

pEnc_ = unique_ptr<NvEncoderCuda>(new NvEncoderCuda(cuContext_, width, height, eFormat));

// InitializeEncoder
NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };

initializeParams.encodeConfig = &encodeConfig;
pEnc_->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());

initializeParams.enableEncodeAsync = true;
encodeConfig.gopLength = control_fps_ * 2;
encodeConfig.frameIntervalP = 1;
if (encodeCLIOptions.IsCodecH264())
{
	// NVENC_INFINITE_GOPLENGTH;
	encodeConfig.encodeCodecConfig.h264Config.idrPeriod = encodeConfig.gopLength;
}
else
{
	// NVENC_INFINITE_GOPLENGTH;
	encodeConfig.encodeCodecConfig.hevcConfig.idrPeriod = encodeConfig.gopLength;
}

//	NV_ENC_PARAMS_RC_CONSTQP	固定质量
//	NV_ENC_PARAMS_RC_VBR	可变比特率
//	NV_ENC_PARAMS_RC_CBR	恒定比特率
encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;

// 平均比特率
int64_t bit = width * height * control_fps_ / 100;
if (bit > 2000000)
{
	// 控制比特率可以尽量保持帧数达到预期，但是过低的比特率会导致花屏
	encodeConfig.rcParams.averageBitRate = width * height * control_fps_ / 100;
}
else
{
	encodeConfig.rcParams.averageBitRate = 2000000;
}

initializeParams.frameRateDen = 1;
initializeParams.frameRateNum = control_fps_;

encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

pEnc_->CreateEncoder(&initializeParams);
```

#### 编码视频
``` cpp
// 原图像数据指针 YUV NV12
void* pSrcFrame;
// 表示源图像帧中每行像素（以字节为单位）的跨度
uint32_t nSrcPitch;

// 编码视频，格式为 vector<vector<uint8_t>>
vector<vector<uint8_t>> vPacket;

const NvEncInputFrame* encoderInputFrame = pEnc_->GetNextInputFrame();
NvEncoderCuda::CopyToDeviceFrame(cuContext_, pSrcFrame, nSrcPitch, (CUdeviceptr)encoderInputFrame->inputPtr,
	(int)encoderInputFrame->pitch,
	pEnc_->GetEncodeWidth(),
	pEnc_->GetEncodeHeight(),
	CU_MEMORYTYPE_HOST,
	encoderInputFrame->bufferFormat,
	encoderInputFrame->chromaOffsets,
	encoderInputFrame->numChromaPlanes);

pEnc_->EncodeFrame(vPacket);
```

#### 处理编码视频
##### 保存为本地文件
``` cpp
string szOutFilePath = "FilePath";
std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
if (!fpOut)
{
	std::ostringstream err;
	err << "Unable to open output file: " << szOutFilePath << std::endl;
	throw std::invalid_argument(err.str());
}
for (std::vector<uint8_t>& packet : packets)
{
	fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
}
fpOut.close();
```

##### 使用 FFmpeg 推流视频
``` cpp
	/// <summary>
	/// 封装 IO 上下文
	/// </summary>
	AVFormatContext* outputContext_;

	/// <summary>
	/// 控制最大 FPS
	/// </summary>
	int control_fps_ = 25;
	
	/// <summary>
	/// 编码 TimeBase
	/// </summary>
	AVRational time_base_;
	
	/// <summary>
	/// 递增 pts
	/// </summary>
	int64_t pts_ = 0;
```

``` cpp
// 在外部初始化 FFmpeg 推流配置
for (auto& packet : vPacket) {
	AVPacket avPacket;
	av_init_packet(&avPacket);

	avPacket.size = packet.size();
	avPacket.data = packet.data();

	// 设置 pts 推流视频
	avPacket.pts = av_rescale_q(pts_++, { 1, control_fps_ }, time_base_);

	ret = av_interleaved_write_frame(outputContext_, &avPacket);
	if (ret < 0)
	{
		av_packet_unref(&avPacket);
		printf("Push stream failed, unable to connect to streaming server!\n");
		return;
	}
	av_packet_unref(&avPacket);
}
```

#### 释放内存
``` cpp
// 释放 CUDA 上下文
cuCtxDestroy(cuContext_);
```

### 注意事项
#### 由于驱动版本导致的异常
NvEncodeAPIGetMaxSupportedVersion(&version) 报错 192>117：  
Current Driver Version does not support this NvEncodeAPI version, please upgrade driver
显卡驱动程序包含的 CUDA 版本号，可以从 nvidia-smi 中查看，NvEncodeAPI 12.0 要求 cuda 版本 12.0，22年12月

#### 编码会话数限制
[消除 Nvidia 对消费级 GPU 施加的最大同时 NVENC 视频编码会话数限制](https://github.com/keylase/nvidia-patch)

#### FFmpeg 推流 TCP 转发到 RTSP
推流到 TCP 需要先使用命令启动服务
``` cmd
:: 使用默认编码器
ffmpeg -listen 1 -i tcp://0.0.0.0:1234 -f rtsp rtsp://localhost:8554/main
:: 使用 libx264 编码器
ffmpeg -listen 1 -i tcp://0.0.0.0:1234 -c:v libx264 -f rtsp rtsp://localhost:8554/main
:: 使用 libx265 编码器
ffmpeg -listen 1 -i tcp://0.0.0.0:1234 -c:v libx265 -f rtsp rtsp://localhost:8554/main
```
