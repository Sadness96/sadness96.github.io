---
title: 拉普拉斯金字塔多波段融合
date: 2022-05-22 01:00:18
tags: [c++,pyramid]
categories: C++
---
<img src="https://sadness96.github.io/images/blog/cpp-Pyramid/pyramidTitle.jpg"/>

<!-- more -->
#### 使用拉普拉斯金字塔（LaplacianPyramid）融合图像
[图像金字塔](https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html) 是图像的集合，所有图像都来自单个原始图像，这些图像被连续下采样，直到达到某个所需的停止点。

有两种常见的图像金字塔：
* 高斯金字塔：用于对图像进行下采样
* 拉普拉斯金字塔：用于从金字塔较低的图像（分辨率较低）重建上采样图像

使用拉普拉斯金字塔融合复原图像，可以解决拼接缝隙问题，也叫做多波段融合(MultibandBlending)

#### 直接拼接
直接使用蒙版拼接两张图片，存在较为明显的拼接缝隙。

| 图片1 | 蒙版 | 图片2 | 结果 |
| ---- | ---- | ---- | ---- |
| <img src="https://sadness96.github.io/images/blog/cpp-Pyramid/apple.jpg" width='120px'/> | <img src="https://sadness96.github.io/images/blog/cpp-Pyramid/mask.jpg" width='120px'/> | <img src="https://sadness96.github.io/images/blog/cpp-Pyramid/orange.jpg" width='120px'/> | <img src="https://sadness96.github.io/images/blog/cpp-Pyramid/result0.jpg" width='120px'/> |

#### 多波段融合
使用 5 层拉普拉斯金字塔融合图像，融合效果较好

| 图片1 | 蒙版 | 图片2 | 结果 |
| ---- | ---- | ---- | ---- |
| <img src="https://sadness96.github.io/images/blog/cpp-Pyramid/apple.jpg" width='120px'/> | <img src="https://sadness96.github.io/images/blog/cpp-Pyramid/mask.jpg" width='120px'/> | <img src="https://sadness96.github.io/images/blog/cpp-Pyramid/orange.jpg" width='120px'/> | <img src="https://sadness96.github.io/images/blog/cpp-Pyramid/result5.jpg" width='120px'/> |

##### 代码
``` CPP
/// <summary>
/// 创建高斯金字塔
/// </summary>
/// <param name="img">原图像</param>
/// <param name="num_levels">金字塔层数</param>
/// <param name="pyr">金字塔集合</param>
void CreateGaussianPyramid(const Mat& img, int num_levels, vector<Mat>& pyr)
{
	pyr.clear();
	Mat gp_img = img;
	pyr.push_back(gp_img);
	for (size_t i = 0; i < num_levels; i++)
	{
		Mat down;
		pyrDown(gp_img, down);
		pyr.push_back(down);
		gp_img = down;
	}
}

/// <summary>
/// 创建拉普拉斯金字塔
/// </summary>
/// <param name="img">原图像</param>
/// <param name="num_levels">金字塔层数</param>
/// <param name="pyr">金字塔集合</param>
/// <param name="highest_level">最高级别图像</param>
void CreateLaplacianPyramid(const Mat& img, int num_levels, vector<Mat>& pyr, Mat& highest_level)
{
	pyr.clear();
	Mat gp_img = img;
	for (size_t i = 0; i < num_levels; i++)
	{
		Mat down, up;
		pyrDown(gp_img, down);
		pyrUp(down, up, gp_img.size());
		Mat lap;
		subtract(gp_img, up, lap);
		pyr.push_back(lap);
		gp_img = down;
	}
	gp_img.copyTo(highest_level);
}

/// <summary>
/// 根据蒙版融合金字塔图像
/// </summary>
/// <param name="pyr_img1">图像1 拉普拉斯金字塔</param>
/// <param name="pyr_img2">图像2 拉普拉斯金字塔</param>
/// <param name="pyr_mask">蒙版 高斯金字塔</param>
/// <param name="num_levels">金字塔层数</param>
/// <param name="pyr">金字塔集合</param>
void FusionPyramidImage(vector<Mat>& pyr_img1, Mat& img1, vector<Mat>& pyr_img2, Mat& img2, vector<Mat>& pyr_mask, int num_levels, vector<Mat>& pyr, Mat& result_highest_level)
{
	pyr.clear();
	result_highest_level = img1.mul(pyr_mask.back()) + img2.mul(Scalar(1.0, 1.0, 1.0) - pyr_mask.back());
	for (size_t i = 0; i < num_levels; i++)
	{
		pyr.push_back(pyr_img1[i].mul(pyr_mask[i]) + pyr_img2[i].mul(Scalar(1.0, 1.0, 1.0) - pyr_mask[i]));
	}
}

/// <summary>
/// 重建图像
/// </summary>
/// <param name="pyr"></param>
/// <param name="num_levels"></param>
/// <returns></returns>
Mat ReconstructImg(vector<Mat>& pyr, Mat& img, int num_levels)
{
	Mat current_img = img;
	for (int i = num_levels - 1; i >= 0; i--)
	{
		Mat up;
		pyrUp(current_img, up, pyr[i].size());
		add(up, pyr[i], current_img);
	}
	return current_img;
}

int main()
{
	Mat img_apple = imread("apple.jpg");
	Mat img_orange = imread("orange.jpg");
	Mat img_mask = imread("mask.jpg");

	// 创建拉普拉斯金字塔层数
	int num_bands_ = 5;

	// 拼接图像转换为 CV_32F 类型
	img_apple.convertTo(img_apple, CV_32F);
	img_orange.convertTo(img_orange, CV_32F);
	// 蒙版图像转换为 CV_32F, 1.0 / 255.0 类型
	img_mask.convertTo(img_mask, CV_32F, 1.0 / 255.0);

	// 可选：创建以中间分隔的蒙版
	//Mat img_mask = Mat::zeros(img_apple.rows, img_apple.cols, CV_32FC1);
	//img_mask(Range::all(), Range(0, img_mask.cols * 0.5)) = 1.0;
	//cvtColor(img_mask, img_mask, CV_GRAY2BGR);

	// 创建拼接图像拉普拉斯金字塔
	vector<Mat> pyr_apple;
	Mat highest_level_apple;
	CreateLaplacianPyramid(img_apple, num_bands_, pyr_apple, highest_level_apple);
	vector<Mat> pyr_orange;
	Mat highest_level_orange;
	CreateLaplacianPyramid(img_orange, num_bands_, pyr_orange, highest_level_orange);
	// 创建蒙版高斯金字塔
	vector<Mat> pyr_mask;
	CreateGaussianPyramid(img_mask, num_bands_, pyr_mask);

	// 融合图像
	vector<Mat> pyr_result;
	Mat result_highest_level;

	// 以 orange 作为底图
	//FusionPyramidImage(pyr_apple, highest_level_apple, pyr_orange, highest_level_orange, pyr_mask, num_bands_, pyr_result, result_highest_level);

	// 以 apple 作为底图
	FusionPyramidImage(pyr_orange, highest_level_orange, pyr_apple, highest_level_apple, pyr_mask, num_bands_, pyr_result, result_highest_level);

	// 重建图像
	Mat result = ReconstructImg(pyr_result, result_highest_level, num_bands_);
	result.convertTo(result, CV_8UC3);

	imshow("result", result);
}
```