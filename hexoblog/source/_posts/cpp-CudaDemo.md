---
title: C++ Cuda Demo
date: 2021-09-14 00:46:21
tags: [c++,cuda]
categories: C++
---
### 使用 Cuda 并行计算对图像处理加速
<!-- more -->
#### 简介
[CUDA](https://www.nvidia.cn/geforce/technologies/cuda/) 是 [NVIDIA](https://www.nvidia.cn/) 发明的一种并行计算平台和编程模型。它通过利用图形处理器 (GPU) 的处理能力，可大幅提升计算性能。
参考：[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

#### 开发环境
* Windows 10
* [Visual Studio 2019](https://visualstudio.microsoft.com/zh-hans/)
* [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)

#### 安装步骤
1. 安装 Cuda 程序后使用命令 "nvcc -V" 验证安装完成。
1. 拷贝目录 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\" 下 "include"、"lib" 目录到项目下。
1. 工程项目添加 VC++ 目录：包含目录和库目录。
1. 连接器 → 输入 → 附加依赖项 中加入："cudart.lib"。
1. 生成依赖项 → 生成自定义 中勾选：CUDA 10.2(.targets,.props)。

#### 示例代码
简单的 Cuda 示例，使用两种方式进行内存和显存的相互拷贝。
TestCuda1：使用 GpuMat 方式拷贝，Opencv 提供的方式，代码更简洁。
TestCuda2：使用 cudaMalloc 方式拷贝，Cuda 方法，效率更高，拷贝速度更快。

##### main.cpp
``` CPP
extern "C" Mat TestCuda1(Mat img);
extern "C" Mat TestCuda2(Mat img);

int main()
{
    Mat img = imread("1.jpg");

    auto img1 = TestCuda1(img);
    auto img2 = TestCuda2(img);

    imshow("1", img1);
    imshow("2", img2);
}
```

##### CudaDemo.cu
* Cuda 代码文件以 .cu 后缀结尾。
* 使用前缀 "_\_global_\_ " 开头修饰的函数是核函数。
* 使用新的 <<<...>>> 调用。

``` CPP
__global__ void CudaCore1(PtrStepSz<uchar3> inputMat, PtrStepSz<uchar3> outputMat)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < inputMat.cols && tidy < inputMat.rows)
	{
		outputMat(tidy, tidx) = inputMat(tidy, tidx);
	}
}

extern "C" Mat TestCuda1(Mat img)
{
	GpuMat inputMat(img);
	auto outputMat = GpuMat(img.rows, img.cols, CV_8UC3);

	int width = img.cols;
	int height = img.rows;

	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	CudaCore1 << <grid, block >> > (inputMat, outputMat);
	cudaThreadSynchronize();

	Mat dstImg;
	outputMat.download(dstImg);
	return dstImg;
}

__global__ void CudaCore2(const uchar3* inputImg, uchar3* outputImg, int width, int height)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < width && tidy < height)
	{
		int idx = tidy * width + tidx;
		outputImg[idx] = inputImg[idx];
	}
}

extern "C" Mat TestCuda2(Mat img)
{
	int height = img.rows;
	int width = img.cols;
	auto img_size = sizeof(uchar3) * height * width;

	uchar3* inputImg = NULL;
	uchar3* outputImg = NULL;

	cudaMalloc((void**)&inputImg, img_size);
	cudaMalloc((void**)&outputImg, img_size);
	cudaMemcpy(inputImg, (uchar3*)img.data, img_size, cudaMemcpyHostToDevice);

	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	CudaCore2 << <grid, block >> > (inputImg, outputImg, width, height);
	cudaFree(inputImg);
	cudaThreadSynchronize();

	Mat dstImg(height, width, CV_8UC3);
	uchar3* outputUChar = (uchar3*)dstImg.data;
	cudaMemcpy(outputUChar, outputImg, img_size, cudaMemcpyDeviceToHost);
	cudaFree(outputImg);
	return dstImg;
}
```