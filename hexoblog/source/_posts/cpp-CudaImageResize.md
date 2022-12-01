---
title: CUDA NV12 缩放图像
date: 2022-09-26 22:32:32
tags: [c++,ffmpeg,cuda]
categories: C++
---
### 最近邻插值法与双线性差值法缩放图像
<!-- more -->
### 简介
通常处理图像时经常会需要缩放图像，[FFmpeg](https://ffmpeg.org/) 可以使用 [SwsContext](https://www.ffmpeg.org/doxygen/2.2/structSwsContext.html) 方法缩放图像，[OpenCV](https://opencv.org/) 可以使用 [cv::resize](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d) 方法缩放图像，但是用这些方法缩放的同时会消耗更多时间，在选择使用 [CUDA](https://developer.nvidia.com/cuda-toolkit) 处理图像的情况下，就直接一起缩放图像。
图像缩放有多重算法，最常见的是：
* 最近邻插值法：速度最快，但是有时可以看到明显锯齿。
* 双线性差值法：使用最多的方式，使用邻近四个点来计算像素值。
* 双三次插值法（不实现该方法）：速度较慢，使用邻近十六个点来计算像素值。

### 实现
代码以 YUV NV12 图片格式编写，其他图片类型带入公式即可。

#### 最近邻插值法
##### 计算方法
计算图片缩放百分比，得到一个在原图像的百分比坐标吗，用当前坐标乘以百分比获取原图像像素值，通常为小数，删除小数部分取整使用左上角像素坐标，也可以选择四舍五入的方式，或者 +0.5 或 -0.5 后取整，作为缩放图像的像素值。
``` CPP
int fx = outX * (inWidth / outWidth);
int fy = outY * (inHeight / outHeight);
```

##### 代码
``` CPP
/// <summary>
/// 缩放图像核函数 YUV420P
/// 最近邻插值
/// </summary>
/// <param name="pInYData">输入图片 YUV YUV420P Y</param>
/// <param name="pInUData">输入图片 YUV YUV420P U</param>
/// <param name="pInVData">输入图片 YUV YUV420P V</param>
/// <param name="pInWidth">输入图片宽度</param>
/// <param name="pInHeight">输入图像高度</param>
/// <param name="pOutYData">输出图片 YUV YUV420P Y</param>
/// <param name="pOutUData">输出图片 YUV YUV420P U</param>
/// <param name="pOutVData">输出图片 YUV YUV420P V</param>
/// <param name="pOutWidth">输出图片宽度</param>
/// <param name="pOutHeight">输出图像高度</param>
/// <returns>缩放后图像</returns>
__global__ void ReSizeKernel_Nearest_YUV420P(unsigned char* pInYData, unsigned char* pInUData, unsigned char* pInVData, int pInWidth, int pInHeight,
	unsigned char* pOutYData, unsigned char* pOutUData, unsigned char* pOutVData, int pOutWidth, int pOutHeight)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < pOutWidth && tidy < pOutHeight)
	{
		int srcX = tidx * ((float)(pInWidth - 1) / (pOutWidth - 1));
		int srcY = tidy * ((float)(pInHeight - 1) / (pOutHeight - 1));

		int idx_in_y = srcY * pInWidth + srcX;
		int idx_in_uv = srcY / 2 * pInWidth / 2 + srcX / 2;

		int idx_out_y = tidy * pOutWidth + tidx;
		int idx_out_uv = tidy / 2 * pOutWidth / 2 + tidx / 2;

		// Y
		pOutYData[idx_out_y] = pInYData[idx_in_y];
		// U
		pOutUData[idx_out_uv] = pInUData[idx_in_uv];
		// V
		pOutVData[idx_out_uv] = pInVData[idx_in_uv];
	}
}

/// <summary>
/// 修改大小 最近邻插值 YUV420P
/// </summary>
/// <param name="frame">输入图像</param>
/// <param name="width">修改宽度</param>
/// <param name="height">修改高度</param>
/// <returns>修改后图像</returns>
AVFrame* ReSize_Nearest_YUV420P(AVFrame* frame, int width, int height)
{
	auto img_size_y = width * height * sizeof(unsigned char);
	auto img_size_uv = (width / 2) * (height / 2) * sizeof(unsigned char);

	AVFrame* dstImg;
	unsigned char* outputY = nullptr;
	unsigned char* outputU = nullptr;
	unsigned char* outputV = nullptr;

	dstImg = av_frame_alloc();
	av_image_alloc(dstImg->data, dstImg->linesize, width, height, (AVPixelFormat)frame->format, 1);
	dstImg->width = width;
	dstImg->height = height;
	dstImg->format = (AVPixelFormat)frame->format;

	cudaMalloc(&outputY, img_size_y);
	cudaMalloc(&outputU, img_size_uv);
	cudaMalloc(&outputV, img_size_uv);

	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	ReSizeKernel_Nearest_YUV420P << <grid, block >> > (frame->data[0], frame->data[1], frame->data[2], frame->width, frame->height, outputY, outputU, outputV, width, height);
	cudaThreadSynchronize();

	// 图像从 Gpu 拷贝到 Cpu
	cudaMemcpy(dstImg->data[0], outputY, img_size_y, cudaMemcpyDeviceToHost);
	cudaMemcpy(dstImg->data[1], outputU, img_size_uv, cudaMemcpyDeviceToHost);
	cudaMemcpy(dstImg->data[2], outputV, img_size_uv, cudaMemcpyDeviceToHost);
	return dstImg;
}

/// <summary>
/// 缩放图像核函数 NV12
/// 最近邻插值
/// </summary>
/// <param name="pInYData">输入图片 YUV NV12 Y</param>
/// <param name="pInUVData">输入图片 YUV NV12 UV</param>
/// <param name="pInWidth">输入图片宽度</param>
/// <param name="pInHeight">输入图像高度</param>
/// <param name="pOutYData">输出图片 YUV NV12 Y</param>
/// <param name="pOutUVData">输出图片 YUV NV12 UV</param>
/// <param name="pOutWidth">输出图片宽度</param>
/// <param name="pOutHeight">输出图像高度</param>
/// <returns>缩放后图像</returns>
__global__ void ReSizeKernel_Nearest_NV12(unsigned char* pInYData, unsigned char* pInUVData, int pInWidth, int pInHeight,
	unsigned char* pOutYData, unsigned char* pOutUVData, int pOutWidth, int pOutHeight)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < pOutWidth && tidy < pOutHeight)
	{
		int srcX = tidx * ((float)(pInWidth - 1) / (pOutWidth - 1));
		int srcY = tidy * ((float)(pInHeight - 1) / (pOutHeight - 1));

		int idx_in_y = srcY * pInWidth + srcX;
		int idx_in_uv = srcY / 2 * pInWidth + srcX;

		int idx_out_y = tidy * pOutWidth + tidx;
		int idx_out_uv = tidy / 2 * pOutWidth + tidx;

		// Y
		pOutYData[idx_out_y] = pInYData[idx_in_y];
		// U
		pOutUVData[tidx % 2 == 0 ? idx_out_uv : idx_out_uv - 1] = pInUVData[srcX % 2 == 0 ? idx_in_uv : idx_in_uv - 1];
		// V
		pOutUVData[tidx % 2 == 0 ? idx_out_uv + 1 : idx_out_uv] = pInUVData[srcX % 2 == 0 ? idx_in_uv + 1 : idx_in_uv];
	}
}

/// <summary>
/// 修改大小 最近邻插值 NV12
/// </summary>
/// <param name="frame">输入图像</param>
/// <param name="width">修改宽度</param>
/// <param name="height">修改高度</param>
/// <returns>修改后图像</returns>
AVFrame* ReSize_Nearest_NV12(AVFrame* frame, int width, int height)
{
	auto img_size_y = width * height * sizeof(unsigned char);
	auto img_size_uv = width * (height / 2) * sizeof(unsigned char);

	AVFrame* dstImg;
	unsigned char* outputY = nullptr;
	unsigned char* outputUV = nullptr;

	dstImg = av_frame_alloc();
	av_image_alloc(dstImg->data, dstImg->linesize, width, height, (AVPixelFormat)frame->format, 1);
	dstImg->width = width;
	dstImg->height = height;
	dstImg->format = (AVPixelFormat)frame->format;

	cudaMalloc(&outputY, img_size_y);
	cudaMalloc(&outputUV, img_size_uv);

	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	ReSizeKernel_Nearest_NV12 << <grid, block >> > (frame->data[0], frame->data[1], frame->width, frame->height, outputY, outputUV, width, height);
	cudaThreadSynchronize();

	// 图像从 Gpu 拷贝到 Cpu
	cudaMemcpy(dstImg->data[0], outputY, img_size_y, cudaMemcpyDeviceToHost);
	cudaMemcpy(dstImg->data[1], outputUV, img_size_uv, cudaMemcpyDeviceToHost);
	return dstImg;
}
```

#### 双线性差值法
##### 计算方法
与最近邻插值法一样，先计算图片缩放百分比，得到一个在原图像的百分比坐标吗，用当前坐标乘以百分比获取原图像像素值，通常为小数，取小数相邻的两个像素，比如计算像素坐标为 7.5，则取删除小数的 7 和填充小数的 8 作为相邻的两个值，放在图像坐标中上下左右相邻四个像素作为计算数据。

1. 获取在原图像的百分比像素，由于数据数组通常以 0 开始，宽度高度减 1 后计算更精准。
	``` CPP
	float fx = outX * ((float)(inWidth - 1) / (outWidth - 1));
	float fy = outY * ((float)(inHeight - 1) / (outHeight - 1));
	```

1. 取相邻四个像素坐标，直接取整为左上角坐标，如果还有余数的情况下 +1 为右下角坐标。
	``` CPP
	int fx0 = fx;
	int fy0 = fy;
	int fx1 = fx > fx0 ? fx0 + 1 : fx0;
	int fy1 = fy > fy0 ? fy0 + 1 : fy0;
	```

1. 取小数部分作为四个像素计算的分量
	``` CPP
	float xProportion = fx - fx0;
	float yProportion = fy - fy0;
	```

1. 带入公式计算新像素值
	``` Text
	f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy;
	```

##### 代码
``` CPP
/// <summary>
/// 缩放图像核函数 YUV420P
/// 双线性差值
/// f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy;
/// </summary>
/// <param name="pInYData">输入图片 YUV YUV420P Y</param>
/// <param name="pInUData">输入图片 YUV YUV420P U</param>
/// <param name="pInVData">输入图片 YUV YUV420P V</param>
/// <param name="pInWidth">输入图片宽度</param>
/// <param name="pInHeight">输入图像高度</param>
/// <param name="pOutYData">输出图片 YUV YUV420P Y</param>
/// <param name="pOutUData">输出图片 YUV YUV420P U</param>
/// <param name="pOutVData">输出图片 YUV YUV420P V</param>
/// <param name="pOutWidth">输出图片宽度</param>
/// <param name="pOutHeight">输出图像高度</param>
/// <returns>缩放后图像</returns>
__global__ void ReSizeKernel_Bilinear_YUV420P(unsigned char* pInYData, unsigned char* pInUData, unsigned char* pInVData, int pInWidth, int pInHeight,
	unsigned char* pOutYData, unsigned char* pOutUData, unsigned char* pOutVData, int pOutWidth, int pOutHeight)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < pOutWidth&& tidy < pOutHeight)
	{
		float srcX = tidx * ((float)(pInWidth - 1) / (pOutWidth - 1));
		float srcY = tidy * ((float)(pInHeight - 1) / (pOutHeight - 1));

		// 计算取图像坐标
		int fx0 = srcX;
		int fy0 = srcY;
		int fx1 = srcX > fx0 ? fx0 + 1 : fx0;
		int fy1 = srcY > fy0 ? fy0 + 1 : fy0;

		// 计算取像素比例
		float xProportion = srcX - fx0;
		float yProportion = srcY - fy0;

		// 四个输入坐标
		int idx_in_y_00 = fy0 * pInWidth + fx0;
		int idx_in_uv_00 = fy0 / 2 * pInWidth / 2 + fx0 / 2;

		int idx_in_y_10 = fy1 * pInWidth + fx0;
		int idx_in_uv_10 = fy1 / 2 * pInWidth / 2 + fx0 / 2;

		int idx_in_y_01 = fy0 * pInWidth + fx1;
		int idx_in_uv_01 = fy0 / 2 * pInWidth / 2 + fx1 / 2;

		int idx_in_y_11 = fy1 * pInWidth + fx1;
		int idx_in_uv_11 = fy1 / 2 * pInWidth / 2 + fx1 / 2;

		// 输出坐标
		int idx_out_y = tidy * pOutWidth + tidx;
		int idx_out_uv = tidy / 2 * pOutWidth / 2 + tidx / 2;

		// Y
		pOutYData[idx_out_y] =
			pInYData[idx_in_y_00] * (1 - xProportion) * (1 - yProportion) +
			pInYData[idx_in_y_10] * xProportion * (1 - yProportion) +
			pInYData[idx_in_y_01] * (1 - xProportion) * yProportion +
			pInYData[idx_in_y_11] * xProportion * yProportion;

		// U
		pOutUData[idx_out_uv] =
			pInUData[idx_in_uv_00] * (1 - xProportion) * (1 - yProportion) +
			pInUData[idx_in_uv_10] * xProportion * (1 - yProportion) +
			pInUData[idx_in_uv_01] * (1 - xProportion) * yProportion +
			pInUData[idx_in_uv_11] * xProportion * yProportion;

		// V
		pOutVData[idx_out_uv] =
			pInVData[idx_in_uv_00] * (1 - xProportion) * (1 - yProportion) +
			pInVData[idx_in_uv_10] * xProportion * (1 - yProportion) +
			pInVData[idx_in_uv_01] * (1 - xProportion) * yProportion +
			pInVData[idx_in_uv_11] * xProportion * yProportion;
	}
}

/// <summary>
/// 修改大小 双线性差值 YUV420P
/// </summary>
/// <param name="frame">输入图像</param>
/// <param name="width">修改宽度</param>
/// <param name="height">修改高度</param>
/// <returns>修改后图像</returns>
AVFrame* ReSize_Bilinear_YUV420P(AVFrame* frame, int width, int height)
{
	auto img_size_y = width * height * sizeof(unsigned char);
	auto img_size_uv = (width / 2) * (height / 2) * sizeof(unsigned char);

	AVFrame* dstImg;
	unsigned char* outputY = nullptr;
	unsigned char* outputU = nullptr;
	unsigned char* outputV = nullptr;

	dstImg = av_frame_alloc();
	av_image_alloc(dstImg->data, dstImg->linesize, width, height, (AVPixelFormat)frame->format, 1);
	dstImg->width = width;
	dstImg->height = height;
	dstImg->format = (AVPixelFormat)frame->format;

	cudaMalloc(&outputY, img_size_y);
	cudaMalloc(&outputU, img_size_uv);
	cudaMalloc(&outputV, img_size_uv);

	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	ReSizeKernel_Bilinear_YUV420P << <grid, block >> > (frame->data[0], frame->data[1], frame->data[2], frame->width, frame->height, outputY, outputU, outputV, width, height);
	cudaThreadSynchronize();

	// 图像从 Gpu 拷贝到 Cpu
	cudaMemcpy(dstImg->data[0], outputY, img_size_y, cudaMemcpyDeviceToHost);
	cudaMemcpy(dstImg->data[1], outputU, img_size_uv, cudaMemcpyDeviceToHost);
	cudaMemcpy(dstImg->data[2], outputV, img_size_uv, cudaMemcpyDeviceToHost);
	return dstImg;
}

/// <summary>
/// 缩放图像核函数 NV12
/// 双线性差值
/// f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy;
/// </summary>
/// <param name="pInYData">输入图片 YUV NV12 Y</param>
/// <param name="pInUVData">输入图片 YUV NV12 UV</param>
/// <param name="pInWidth">输入图片宽度</param>
/// <param name="pInHeight">输入图像高度</param>
/// <param name="pOutYData">输出图片 YUV NV12 Y</param>
/// <param name="pOutUVData">输出图片 YUV NV12 UV</param>
/// <param name="pOutWidth">输出图片宽度</param>
/// <param name="pOutHeight">输出图像高度</param>
/// <returns>缩放后图像</returns>
__global__ void ReSizeKernel_Bilinear_NV12(unsigned char* pInYData, unsigned char* pInUVData, int pInWidth, int pInHeight,
	unsigned char* pOutYData, unsigned char* pOutUVData, int pOutWidth, int pOutHeight)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < pOutWidth&& tidy < pOutHeight)
	{
		float srcX = tidx * ((float)(pInWidth - 1) / (pOutWidth - 1));
		float srcY = tidy * ((float)(pInHeight - 1) / (pOutHeight - 1));

		// 计算取图像坐标
		int fx0 = srcX;
		int fy0 = srcY;
		int fx1 = srcX > fx0 ? fx0 + 1 : fx0;
		int fy1 = srcY > fy0 ? fy0 + 1 : fy0;

		// 计算取像素比例
		float xProportion = srcX - fx0;
		float yProportion = srcY - fy0;

		// 四个输入坐标
		int idx_in_y_00 = fy0 * pInWidth + fx0;
		int idx_in_uv_00 = fy0 / 2 * pInWidth + fx0;

		int idx_in_y_10 = fy1 * pInWidth + fx0;
		int idx_in_uv_10 = fy1 / 2 * pInWidth + fx0;

		int idx_in_y_01 = fy0 * pInWidth + fx1;
		int idx_in_uv_01 = fy0 / 2 * pInWidth + fx1;

		int idx_in_y_11 = fy1 * pInWidth + fx1;
		int idx_in_uv_11 = fy1 / 2 * pInWidth + fx1;

		// 输出坐标
		int idx_out_y = tidy * pOutWidth + tidx;
		int idx_out_uv = tidy / 2 * pOutWidth + tidx;

		// Y
		pOutYData[idx_out_y] =
			pInYData[idx_in_y_00] * (1 - xProportion) * (1 - yProportion) +
			pInYData[idx_in_y_10] * xProportion * (1 - yProportion) +
			pInYData[idx_in_y_01] * (1 - xProportion) * yProportion +
			pInYData[idx_in_y_11] * xProportion * yProportion;

		// U
		pOutUVData[tidx % 2 == 0 ? idx_out_uv : idx_out_uv - 1] =
			pInUVData[fx0 % 2 == 0 ? idx_in_uv_00 : idx_in_uv_00 - 1] * (1 - xProportion) * (1 - yProportion) +
			pInUVData[fx0 % 2 == 0 ? idx_in_uv_10 : idx_in_uv_10 - 1] * xProportion * (1 - yProportion) +
			pInUVData[fx1 % 2 == 0 ? idx_in_uv_01 : idx_in_uv_01 - 1] * (1 - xProportion) * yProportion +
			pInUVData[fx1 % 2 == 0 ? idx_in_uv_11 : idx_in_uv_11 - 1] * xProportion * yProportion;

		// V
		pOutUVData[tidx % 2 == 0 ? idx_out_uv + 1 : idx_out_uv] =
			pInUVData[fx0 % 2 == 0 ? idx_in_uv_00 + 1 : idx_in_uv_00] * (1 - xProportion) * (1 - yProportion) +
			pInUVData[fx0 % 2 == 0 ? idx_in_uv_10 + 1 : idx_in_uv_10] * xProportion * (1 - yProportion) +
			pInUVData[fx1 % 2 == 0 ? idx_in_uv_01 + 1 : idx_in_uv_01] * (1 - xProportion) * yProportion +
			pInUVData[fx1 % 2 == 0 ? idx_in_uv_11 + 1 : idx_in_uv_11] * xProportion * yProportion;
	}
}

/// <summary>
/// 修改大小 双线性差值 NV12
/// </summary>
/// <param name="frame">输入图像</param>
/// <param name="width">修改宽度</param>
/// <param name="height">修改高度</param>
/// <returns>修改后图像</returns>
AVFrame* ReSize_Bilinear_NV12(AVFrame* frame, int width, int height)
{
	auto img_size_y = width * height * sizeof(unsigned char);
	auto img_size_uv = width * (height / 2) * sizeof(unsigned char);

	AVFrame* dstImg;
	unsigned char* outputY = nullptr;
	unsigned char* outputUV = nullptr;

	dstImg = av_frame_alloc();
	av_image_alloc(dstImg->data, dstImg->linesize, width, height, (AVPixelFormat)frame->format, 1);
	dstImg->width = width;
	dstImg->height = height;
	dstImg->format = (AVPixelFormat)frame->format;

	cudaMalloc(&outputY, img_size_y);
	cudaMalloc(&outputUV, img_size_uv);

	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	ReSizeKernel_Bilinear_NV12 << <grid, block >> > (frame->data[0], frame->data[1], frame->width, frame->height, outputY, outputUV, width, height);
	cudaThreadSynchronize();

	// 图像从 Gpu 拷贝到 Cpu
	cudaMemcpy(dstImg->data[0], outputY, img_size_y, cudaMemcpyDeviceToHost);
	cudaMemcpy(dstImg->data[1], outputUV, img_size_uv, cudaMemcpyDeviceToHost);
	return dstImg;
}
```

#### 外部调用方法
``` CPP
/// <summary>
/// 修改图像大小
/// </summary>
/// <param name="frame">输入图像</param>
/// <param name="width">修改宽度</param>
/// <param name="height">修改高度</param>
/// <param name="type">0:最近邻插值 1:双线性差值</param>
/// <returns>修改后图像</returns>
extern "C" AVFrame * ReSize(AVFrame * frame, int width, int height, int type)
{
	AVFrame* outFrame;

	switch (frame->format)
	{
	case AV_PIX_FMT_YUV420P:
		if (type == 0)
		{
			outFrame = ReSize_Nearest_YUV420P(frame, width, height);
		}
		else if (type == 1)
		{
			outFrame = ReSize_Bilinear_YUV420P(frame, width, height);
		}
		break;
	case AV_PIX_FMT_NV12:
		if (type == 0)
		{
			outFrame = ReSize_Nearest_NV12(frame, width, height);
		}
		else if (type == 1)
		{
			outFrame = ReSize_Bilinear_NV12(frame, width, height);
		}
		break;
	default:
		break;
	}

	return outFrame;
}
```