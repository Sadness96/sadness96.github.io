---
title: OpenCV 计算最大内接矩形
date: 2021-10-22 14:08:18
tags: [c++,c#,depends]
categories: C++
---
<img src="https://sadness96.github.io/images/blog/cpp-InscribedRectangle/441036.jpg"/>

<!-- more -->
### 简介
使用 OpenCV 计算任意形状图像内最大矩形，最大内接矩形。

### 计算最大内接矩形
#### 代码
``` CPP
/// <summary>
/// 获取有效像素百分比
/// </summary>
/// <param name="panomask">蒙版图像</param>
/// <param name="IsRow">是否是行判断</param>
/// <param name="Number">起始坐标</param>
/// <param name="StartPixel">起始像素</param>
/// <param name="StopPixel">中止像素</param>
/// <returns>空像素百分比</returns>
double GetEffectivePixel(Mat panomask, bool IsRow, int Number, int StartPixel, int StopPixel)
{
	int invalidNumber = 0;
	if (IsRow)
	{
		// 行裁切判断
		for (int i = StartPixel; i < StopPixel; i++)
		{
			Vec3b data = panomask.at<Vec3b>(Number, i);
			int B = data[0];
			int G = data[1];
			int R = data[2];
			if (B <= 0 && G <= 0 && R <= 0)
			{
				invalidNumber++;
			}
		}
	}
	else
	{
		// 列裁切判断
		for (int i = StartPixel; i < StopPixel; i++)
		{
			Vec3b data = panomask.at<Vec3b>(i, Number);
			int B = data[0];
			int G = data[1];
			int R = data[2];
			if (B <= 0 && G <= 0 && R <= 0)
			{
				invalidNumber++;
			}
		}
	}
	return (double)invalidNumber / ((double)StopPixel - (double)StartPixel);
}

/// <summary>
/// 计算裁切范围
/// </summary>
/// <param name="panomask">蒙版图像</param>
/// <returns>裁切范围</returns>
Rect CalcCuttingRange(Mat panomask)
{
	Mat panomaskRGB;
	panomask.convertTo(panomaskRGB, CV_8U);
	// 裁切前图像宽高
	int height = panomaskRGB.rows;
	int width = panomaskRGB.cols;
	// 上下左右边距
	int top = 0;
	int buttom = 0;
	int left = 0;
	int right = 0;
	// 当前边距百分比
	double topPercent = 1;
	double buttomPercent = 1;
	double leftPercent = 1;
	double rightPercent = 1;
	while (topPercent > 0 || buttomPercent > 0 || leftPercent > 0 || rightPercent > 0)
	{
		if (topPercent > 0 && topPercent >= buttomPercent && topPercent >= leftPercent && topPercent >= rightPercent)
		{
			top++;
			topPercent = GetEffectivePixel(panomaskRGB, true, top, left, width - right);
			continue;
		}
		if (buttomPercent > 0 && buttomPercent >= topPercent && buttomPercent >= leftPercent && buttomPercent >= rightPercent)
		{
			buttom++;
			buttomPercent = GetEffectivePixel(panomaskRGB, true, height - buttom, left, width - right);
			continue;
		}
		if (leftPercent > 0 && leftPercent >= topPercent && leftPercent >= buttomPercent && leftPercent >= rightPercent)
		{
			left++;
			leftPercent = GetEffectivePixel(panomaskRGB, false, left, top, height - buttom);
			continue;
		}
		if (rightPercent > 0 && rightPercent >= topPercent && rightPercent >= buttomPercent && rightPercent >= leftPercent)
		{
			right++;
			rightPercent = GetEffectivePixel(panomaskRGB, false, width - right, top, height - buttom);
			continue;
		}
	}

	Rect rect;
	rect.x = left;
	rect.y = top;
	rect.height = height - (top + buttom);
	rect.width = width - (left + right);
	return rect;
}

int main()
{
	Mat img_mask = imread("mask.jpg");
	auto img_rect = CalcCuttingRange(img_mask);
	auto img_cutting = img_mask(img_rect);
	waitKey(0);
}
```

#### 计算结果
<img src="https://sadness96.github.io/images/blog/cpp-InscribedRectangle/rect.jpg"/>

#### 计算过程
<img src="https://sadness96.github.io/images/blog/cpp-InscribedRectangle/CalcCuttingRange.gif"/>