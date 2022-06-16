---
title: Cuda 优化图像金字塔
date: 2022-06-15 23:28:00
tags: [c++,cuda,pyramid]
categories: C++
---
### 基于 Cuda 并行加速处理图像金字塔
<!-- more -->
#### 简介
[拉普拉斯金字塔多波段融合](https://sadness96.github.io/blog/2022/05/22/cpp-Pyramid/) 用于融合图像，OpenCV 提供了一系列方法实现这一功能，其实也提供了 GPU 加速版本，但是其中的 cuda::pyrUp 无法固定图像大小，导致很多问题，所以使用 Cuda 重写这些方法，也方便后期根据实际需求修改。

#### 需要重写的方法介绍
[cv.GaussianBlur](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1) 使用高斯滤波模糊图像，在下采样时保留重要信息，并且在上采样时还原信息。
[cv::pyrDown](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff) 使用高斯滤波模糊图像后再删除偶数行偶数列，输出的图像大小为：Size((src.cols+1)/2, (src.rows+1)/2)。
[cv::pyrUp](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gada75b59bdaaca411ed6fee10085eb784) 对图像上采样，然后使用四倍高斯模糊图像，默认输出的图像大小为：Size(src.cols*2, src.rows*2)，通常使用下采样前的图像大小作为输出图像大小。
[cv.subtract](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gaa0f00d98b4b5edeaeb7b8333b2de353b) 计算两个数组或数组与标量之间的每元素差，图像大小相等时，方法等同于 img1 - img2。
[cv.add](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6) 计算两个数组或一个数组和一个标量的每元素和，图像大小相等时，方法等同于 img1 + img2。

#### 加载原始图像
左上角加的两条红线用于测试下采样删除偶数行前的高斯滤波验证。
<img src="https://sadness96.github.io/images/blog/cpp-PyramidCuda/0.apple.jpg"/>

#### 重写代码
##### GaussianBlur
对图像使用 OpenCV 官方推荐的卷积核进行卷积操作模糊图像。
<img src="https://sadness96.github.io/images/blog/cpp-PyramidCuda/Gaussian kernel.jpg"/>

GaussianBlurGpu 高斯滤波函数用于下采样时使用。
GaussianBlurFourfoldGpu 四倍高斯滤波函数用于上采样时使用。

###### 效果
<img src="https://sadness96.github.io/images/blog/cpp-PyramidCuda/1.apple_gb_gpu.jpg"/>

###### 代码
``` C++
__global__ void GaussianBlurCore(PtrStepSz<uchar3> inputMat, PtrStepSz<uchar3> outputMat)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	float blur[5][5] = {
		{0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
		{0.015625,   0.0625,   0.09375,   0.0625,   0.015625  },
		{0.0234375,  0.09375,  0.140625,  0.09375,  0.0234375 },
		{0.015625,   0.0625,   0.09375,   0.0625,   0.015625  },
		{0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625} };

	int img_cols_max_index = inputMat.cols - 1;
	int img_rows_max_index = inputMat.rows - 1;

	if (tidx < outputMat.cols && tidy < outputMat.rows)
	{
		float b, g, r;
		for (int x = -2; x <= 2; x++)
		{
			for (int y = -2; y <= 2; y++)
			{
				int image_x = tidx + x;
				if (image_x < 0)
				{
					image_x = abs(image_x);
				}
				if (image_x >= inputMat.cols)
				{
					image_x = img_cols_max_index - (x - (img_cols_max_index - tidx));
				}
				int image_y = tidy + y;
				if (image_y < 0)
				{
					image_y = abs(image_y);
				}
				if (image_y >= inputMat.rows)
				{
					image_y = img_rows_max_index - (y - (img_rows_max_index - tidy));
				}
				b += (float)inputMat(image_y, image_x).x * (float)blur[y + 2][x + 2];
				g += (float)inputMat(image_y, image_x).y * (float)blur[y + 2][x + 2];
				r += (float)inputMat(image_y, image_x).z * (float)blur[y + 2][x + 2];
			}
		}
		outputMat(tidy, tidx).x = (int)b;
		outputMat(tidy, tidx).y = (int)g;
		outputMat(tidy, tidx).z = (int)r;
	}
}

/// <summary>
/// 高斯滤波
/// </summary>
/// <param name="img"></param>
/// <returns></returns>
extern "C" Mat GaussianBlurGpu(Mat img)
{
	GpuMat inputMat(img);

	int width = img.cols;
	int height = img.rows;
	auto outputMat = GpuMat(height, width, CV_8UC3);

	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	GaussianBlurCore << <grid, block >> > (inputMat, outputMat);

	cudaThreadSynchronize();

	Mat dstImg;
	outputMat.download(dstImg);
	return dstImg;
}

__global__ void GaussianBlurFourfoldCore(PtrStepSz<uchar3> inputMat, PtrStepSz<uchar3> outputMat)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	float blur[5][5] = {
		{0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
		{0.015625,   0.0625,   0.09375,   0.0625,   0.015625  },
		{0.0234375,  0.09375,  0.140625,  0.09375,  0.0234375 },
		{0.015625,   0.0625,   0.09375,   0.0625,   0.015625  },
		{0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625} };

	int img_cols_max_index = inputMat.cols - 1;
	int img_rows_max_index = inputMat.rows - 1;

	if (tidx < outputMat.cols && tidy < outputMat.rows)
	{
		float b, g, r;
		for (int x = -2; x <= 2; x++)
		{
			for (int y = -2; y <= 2; y++)
			{
				int image_x = tidx + x;
				if (image_x < 0)
				{
					image_x = abs(image_x);
				}
				if (image_x >= inputMat.cols)
				{
					image_x = img_cols_max_index - (x - (img_cols_max_index - tidx));
				}
				int image_y = tidy + y;
				if (image_y < 0)
				{
					image_y = abs(image_y);
				}
				if (image_y >= inputMat.rows)
				{
					image_y = img_rows_max_index - (y - (img_rows_max_index - tidy));
				}
				b += (float)inputMat(image_y, image_x).x * ((float)blur[y + 2][x + 2] * 4);
				g += (float)inputMat(image_y, image_x).y * ((float)blur[y + 2][x + 2] * 4);
				r += (float)inputMat(image_y, image_x).z * ((float)blur[y + 2][x + 2] * 4);
			}
		}
		outputMat(tidy, tidx).x = (int)b;
		outputMat(tidy, tidx).y = (int)g;
		outputMat(tidy, tidx).z = (int)r;
	}
}

/// <summary>
/// 四倍高斯滤波
/// </summary>
/// <param name="img"></param>
/// <returns></returns>
extern "C" Mat GaussianBlurFourfoldGpu(Mat img)
{
	GpuMat inputMat(img);

	int width = img.cols;
	int height = img.rows;
	auto outputMat = GpuMat(height, width, CV_8UC3);

	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	GaussianBlurFourfoldCore << <grid, block >> > (inputMat, outputMat);

	cudaThreadSynchronize();

	Mat dstImg;
	outputMat.download(dstImg);
	return dstImg;
}
```

##### pyrDown
调用高斯滤波函数模糊图像，设定图像大小为 Size((src.cols+1)/2, (src.rows+1)/2)，删除偶数行和偶数列。

###### 效果
<img src="https://sadness96.github.io/images/blog/cpp-PyramidCuda/2.apple_gb_down_gpu.jpg"/>

###### 代码
``` C++
__global__ void PyrDownCore(PtrStepSz<uchar3> inputMat, PtrStepSz<uchar3> outputMat)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < outputMat.cols && tidy < outputMat.rows)
	{
		outputMat(tidy, tidx) = inputMat(tidy * 2, tidx * 2);
	}
}

/// <summary>
/// 金字塔下采样
/// </summary>
/// <param name="img"></param>
/// <returns></returns>
extern "C" Mat PyrDownGpu(Mat img)
{
	// 1. 高斯滤波
	GpuMat inputMat(img);

	int gb_width = img.cols;
	int gb_height = img.rows;
	auto output_gb_mat = GpuMat(gb_height, gb_width, CV_8UC3);

	dim3 block_gb(32, 32);
	dim3 grid_gb((gb_width + block_gb.x - 1) / block_gb.x, (gb_height + block_gb.y - 1) / block_gb.y);
	GaussianBlurCore << <grid_gb, block_gb >> > (inputMat, output_gb_mat);

	// 2. 对高斯滤波后的图像删除偶数行列
	int down_width = (img.cols + 1) / 2;
	int down_height = (img.rows + 1) / 2;
	auto output_down_mat = GpuMat(down_height, down_width, CV_8UC3);

	dim3 block_down(32, 32);
	dim3 grid_down((down_width + block_down.x - 1) / block_down.x, (down_height + block_down.y - 1) / block_down.y);
	PyrDownCore << <grid_down, block_down >> > (output_gb_mat, output_down_mat);

	Mat dstImg;
	output_down_mat.download(dstImg);
	return dstImg;
}
```

##### pyrUp
设定图像大小为下采样前大小，偶数行偶数列填充为0，在使用四倍高斯滤波模糊图像。

###### 效果
<img src="https://sadness96.github.io/images/blog/cpp-PyramidCuda/3.apple_gb_up_gpu.jpg"/>

<img src="https://sadness96.github.io/images/blog/cpp-PyramidCuda/4.apple_gb_four_gpu.jpg"/>

###### 代码
``` C++
__global__ void PyrUpCore(PtrStepSz<uchar3> inputMat, PtrStepSz<uchar3> outputMat)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < outputMat.cols && tidy < outputMat.rows)
	{
		if (tidy % 2 == 0 && tidx % 2 == 0)
		{
			outputMat(tidy, tidx) = inputMat(tidy / 2, tidx / 2);
		}
		else
		{
			outputMat(tidy, tidx) = uchar3();
		}
	}
}

/// <summary>
/// 金字塔上采样
/// </summary>
/// <param name="img"></param>
/// <param name="img_size"></param>
/// <returns></returns>
extern "C" Mat PyrUpGpu(Mat img, Size img_size)
{
	// 1. 扩大范围填充0
	GpuMat inputMat(img);

	int up_width = img_size.width;
	int up_height = img_size.height;
	auto output_up_mat = GpuMat(up_height, up_width, CV_8UC3);

	dim3 block_up(32, 32);
	dim3 grid_up((up_width + block_up.x - 1) / block_up.x, (up_height + block_up.y - 1) / block_up.y);
	PyrUpCore << <grid_up, block_up >> > (inputMat, output_up_mat);

	// 2. 四倍高斯滤波
	int gb4_width = img_size.width;
	int gb4_height = img_size.height;
	auto output_gb4_mat = GpuMat(up_height, up_width, CV_8UC3);

	dim3 block_gb4(32, 32);
	dim3 grid_gb4((gb4_width + block_gb4.x - 1) / block_gb4.x, (gb4_height + block_gb4.y - 1) / block_gb4.y);
	GaussianBlurFourfoldCore << <grid_gb4, block_gb4 >> > (output_up_mat, output_gb4_mat);

	Mat dstImg;
	output_gb4_mat.download(dstImg);
	return dstImg;
}
```

#### subtract
计算两个数组或数组与标量之间的每元素差，生成拉普拉斯金字塔所需图像。

###### 效果
<img src="https://sadness96.github.io/images/blog/cpp-PyramidCuda/5.apple_lb_gpu.jpg"/>

###### 代码
``` C++
__global__ void SubtractCore(PtrStepSz<uchar3> inputMat1, PtrStepSz<uchar3> inputMat2, PtrStepSz<uchar3> outputMat)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < outputMat.cols && tidy < outputMat.rows)
	{
		outputMat(tidy, tidx).x = inputMat1(tidy, tidx).x - inputMat2(tidy, tidx).x;
		outputMat(tidy, tidx).y = inputMat1(tidy, tidx).y - inputMat2(tidy, tidx).y;
		outputMat(tidy, tidx).z = inputMat1(tidy, tidx).z - inputMat2(tidy, tidx).z;
	}
}

/// <summary>
/// 图像减除
/// </summary>
/// <param name="img1"></param>
/// <param name="img2"></param>
/// <returns></returns>
extern "C" Mat SubtractGpu(Mat img1, Mat img2)
{
	GpuMat inputMat1(img1);
	GpuMat inputMat2(img2);

	int width = img1.cols;
	int height = img1.rows;
	auto outputMat = GpuMat(height, width, CV_8UC3);

	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	SubtractCore << <grid, block >> > (inputMat1, inputMat2, outputMat);

	cudaThreadSynchronize();

	Mat dstImg;
	outputMat.download(dstImg);
	return dstImg;
}
```

#### add
计算两个数组或一个数组和一个标量的每元素和，使用上采样图像与拉普拉斯金字塔图像相加可还原图像

###### 效果
<img src="https://sadness96.github.io/images/blog/cpp-PyramidCuda/6.apple_add_gpu.jpg"/>

###### 代码
``` C++
__global__ void AddCore(PtrStepSz<uchar3> inputMat1, PtrStepSz<uchar3> inputMat2, PtrStepSz<uchar3> outputMat)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < outputMat.cols && tidy < outputMat.rows)
	{
		outputMat(tidy, tidx).x = inputMat1(tidy, tidx).x + inputMat2(tidy, tidx).x;
		outputMat(tidy, tidx).y = inputMat1(tidy, tidx).y + inputMat2(tidy, tidx).y;
		outputMat(tidy, tidx).z = inputMat1(tidy, tidx).z + inputMat2(tidy, tidx).z;
	}
}

/// <summary>
/// 图像增加
/// </summary>
/// <param name="img1"></param>
/// <param name="img2"></param>
/// <returns></returns>
extern "C" Mat AddGpu(Mat img1, Mat img2)
{
	GpuMat inputMat1(img1);
	GpuMat inputMat2(img2);

	int width = img1.cols;
	int height = img1.rows;
	auto outputMat = GpuMat(height, width, CV_8UC3);

	dim3 block(32, 32);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	AddCore << <grid, block >> > (inputMat1, inputMat2, outputMat);

	cudaThreadSynchronize();

	Mat dstImg;
	outputMat.download(dstImg);
	return dstImg;
}
```