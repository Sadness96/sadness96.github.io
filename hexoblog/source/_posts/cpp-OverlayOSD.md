---
title: 视频中叠加 OSD
date: 2023-01-18 23:16:32
tags: [c++,ffmpeg,opencv,cuda]
categories: C++
---
### 视频中叠加时间戳或固定文本
<!-- more -->
### 简介
在监控视频中普遍需要在视频上叠加时间戳与相机所在位置，方便回放监控时快速确认有效信息，在处理其他视频时也可以叠加水印等标识。
如果仅使用 FFmpeg 编解码，可以使用 [AVFilter](https://ffmpeg.org/doxygen/trunk/structAVFilter.html) 一系列方法给视频添加水印或文字信息，但是使用上多有不便，同时想解决 OpenCV Mat 无法叠加汉字的问题，所以使用了另一种方式。
参考文章：[Opencv310图片Mat中叠加汉字](https://blog.csdn.net/zmdsjtu/article/details/53133223) 中使用的 Windows [LOGFONTA](https://learn.microsoft.com/zh-cn/windows/win32/api/dimm/ns-dimm-logfonta?redirectedfrom=MSDN) 创建的位图，以 Opencv 或 CUDA 的方式叠加到视频中。

### 核心代码
#### OSDAlignment.h
``` cpp
#pragma once

/// <summary>
/// OSD 垂直对齐方式
/// </summary>
enum OSDVerticalAlignment {
	/// <summary>
	/// 顶部对齐
	/// </summary>
	Top,
	/// <summary>
	/// 居中对齐
	/// </summary>
	VCenter,
	/// <summary>
	/// 底部对齐
	/// </summary>
	Bottom
};

/// <summary>
/// OSD 水平对齐方式
/// </summary>
enum OSDHorizontalAlignment {
	/// <summary>
	/// 左对齐
	/// </summary>
	Left,
	/// <summary>
	/// 居中对齐
	/// </summary>
	HCenter,
	/// <summary>
	/// 右对齐
	/// </summary>
	Right
};
```

#### OverlayOSD.cu
``` cpp
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
#include <core/types.hpp>

using namespace cv;

/// <summary>
/// 叠加 OSD 核函数
/// </summary>
/// <param name="yData">图像 Y 分量</param>
/// <param name="uvData">图像 UV 分量</param>
/// <param name="imageWidth">图像宽度</param>
/// <param name="osdData">叠加 OSD 数据 RGB</param>
/// <param name="osdRect">叠加 OSD 参数</param>
/// <param name="drawLineStep">叠加 OSD 行大小</param>
/// <returns></returns>
__global__ void OSDNearestKernel(unsigned char* yData, unsigned char* uvData, int imageWidth,
	unsigned char* osdData, Rect osdRect, int drawLineStep)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	if (tidx < osdRect.width && tidy < osdRect.height)
	{
		int srcX = tidx + osdRect.x;
		int srcY = tidy + osdRect.y;

		int idx_in_y = srcY * imageWidth + srcX;
		int idx_in_uv = srcY / 2 * imageWidth + srcX;

		int idx_osd = drawLineStep * (osdRect.height - tidy) + tidx * 3;

		if (osdData[idx_osd] > 0)
		{
			yData[idx_in_y] = 255;
			uvData[srcX % 2 == 0 ? idx_in_uv : idx_in_uv - 1] = 128;
			uvData[srcX % 2 == 0 ? idx_in_uv + 1 : idx_in_uv] = 128;
		}
		else
		{
			yData[idx_in_y] = yData[idx_in_y] >= 40 ? yData[idx_in_y] - 40 : 0;
			uvData[srcX % 2 == 0 ? idx_in_uv : idx_in_uv - 1] = uvData[srcX % 2 == 0 ? idx_in_uv : idx_in_uv - 1];
			uvData[srcX % 2 == 0 ? idx_in_uv + 1 : idx_in_uv] = uvData[srcX % 2 == 0 ? idx_in_uv + 1 : idx_in_uv];
		}
	}
}

/// <summary>
/// 叠加 OSD
/// </summary>
/// <param name="yData">图像 Y 分量</param>
/// <param name="uvData">图像 UV 分量</param>
/// <param name="imageWidth">图像宽度</param>
/// <param name="osdData">叠加 OSD 数据 RGB</param>
/// <param name="osdRect">叠加 OSD 参数</param>
/// <param name="drawLineStep">叠加 OSD 行大小</param>
/// <returns></returns>
extern "C" bool OSDNearest(unsigned char* yData, unsigned char* uvData, int imageWidth,
	unsigned char* osdData, Rect osdRect, int drawLineStep)
{
	dim3 block(32, 32);
	dim3 grid((osdRect.width + block.x - 1) / block.x, (osdRect.height + block.y - 1) / block.y);
	OSDNearestKernel << <grid, block >> > (yData, uvData, imageWidth, osdData, osdRect, drawLineStep);
	cudaError_t error = cudaThreadSynchronize();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		return false;
	}
	return true;
}
```

#### OverlayOSD.h
``` cpp
#pragma once

#include <windows.h>
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
#include <string>
#include <opencv2/opencv.hpp>
#include "OSDAlignment.h"

using namespace cv;

class OverlayOSD
{
private:
	const char* overlayText_;
	int imageWidth_ = 0;
	int imageHeight_ = 0;

	LOGFONTA logFonta_;
	HFONT hFont_, hOldFont_;
	HDC hDC_;
	HBITMAP hBmp_, hOldBmp_;

	int singleRow_ = 0;
	int drawLineStep_ = 0;
	void* pDibData_ = 0;

	/// <summary>
	/// 获取文本大小
	/// </summary>
	void GetStringSize(HDC hDC, const char* str, int* w, int* h);

public:
	/// <summary>
	/// 叠加坐标
	/// </summary>
	Rect OverlayCoordinate;

	/// <summary>
	/// 初始化 OSD
	/// </summary>
	/// <param name="str">叠加文本</param>
	/// <param name="fontSize">字体大小</param>
	/// <param name="faceName">字体类型</param>
	/// <param name="imageWidth">叠加图像宽度</param>
	/// <param name="imageHeight">叠加图像高度</param>
	/// <param name="verticalAlignment">垂直对齐方式</param>
	/// <param name="horizontalAlignment">水平对齐方式</param>
	/// <param name="startPoint">相对坐标</param>
	/// <returns></returns>
	void InitOSD(const char* str, int fontSize, const char* faceName, int imageWidth, int imageHeight,
		OSDVerticalAlignment verticalAlignment, OSDHorizontalAlignment horizontalAlignment, Point startPoint);

	/// <summary>
	/// 叠加日期时间 OSD
	/// </summary>
	/// <param name="dst">OpenCV Mat</param>
	void OverlayDateTime(Mat& dst);

	/// <summary>
	/// 叠加日期时间 OSD
	/// </summary>
	/// <param name="YData">Y 分量</param>
	/// <param name="UVData">UV 分量</param>
	void OverlayDateTime(unsigned char* yData, unsigned char* uvData);

	/// <summary>
	/// 叠加文本 OSD
	/// </summary>
	/// <param name="dst">OpenCV Mat</param>
	void OverlayText(Mat& dst);

	/// <summary>
	/// 叠加文本 OSD
	/// </summary>
	/// <param name="YData">Y 分量</param>
	/// <param name="UVData">UV 分量</param>
	void OverlayText(unsigned char* yData, unsigned char* uvData);

	/// <summary>
	/// 释放所有资源
	/// </summary>
	void Dispose();
};

/// <summary>
/// 叠加 OSD
/// </summary>
/// <param name="yData">图像 Y 分量</param>
/// <param name="uvData">图像 UV 分量</param>
/// <param name="imageWidth">图像宽度</param>
/// <param name="osdData">叠加 OSD 数据 RGB</param>
/// <param name="osdRect">叠加 OSD 参数</param>
/// <param name="drawLineStep">叠加 OSD 行大小</param>
/// <returns></returns>
extern "C" bool OSDNearest(unsigned char* yData, unsigned char* uvData, int imageWidth,
	unsigned char* osdData, Rect osdRect, int drawLineStep);
```

#### 
``` cpp
#include "OverlayOSD.h"

/// <summary>
/// 获取文本大小
/// </summary>
void OverlayOSD::GetStringSize(HDC hDC, const char* str, int* w, int* h)
{
	SIZE size;
	GetTextExtentPoint32A(hDC, str, strlen(str), &size);
	if (w != 0) *w = size.cx;
	if (h != 0) *h = size.cy;
}

/// <summary>
/// 初始化 OSD
/// </summary>
/// <param name="str">叠加文本</param>
/// <param name="fontSize">字体大小</param>
/// <param name="faceName">字体类型</param>
/// <param name="imageWidth">叠加图像宽度</param>
/// <param name="imageHeight">叠加图像高度</param>
/// <param name="verticalAlignment">垂直对齐方式</param>
/// <param name="horizontalAlignment">水平对齐方式</param>
/// <param name="startPoint">相对坐标</param>
/// <returns></returns>
void OverlayOSD::InitOSD(const char* str, int fontSize, const char* faceName, int imageWidth, int imageHeight,
	OSDVerticalAlignment verticalAlignment, OSDHorizontalAlignment horizontalAlignment, Point startPoint)
{
	overlayText_ = str;
	imageWidth_ = imageWidth;
	imageHeight_ = imageHeight;

	logFonta_.lfHeight = -fontSize;
	logFonta_.lfWidth = 0;
	logFonta_.lfEscapement = 0;
	logFonta_.lfOrientation = 0;
	logFonta_.lfWeight = 5;
	logFonta_.lfItalic = false;
	logFonta_.lfUnderline = false;
	logFonta_.lfStrikeOut = 0;
	logFonta_.lfCharSet = DEFAULT_CHARSET;
	logFonta_.lfOutPrecision = 0;
	logFonta_.lfClipPrecision = 0;
	logFonta_.lfQuality = PROOF_QUALITY;
	logFonta_.lfPitchAndFamily = 0;
	strcpy_s(logFonta_.lfFaceName, faceName);

	hFont_ = CreateFontIndirectA(&logFonta_);
	hDC_ = CreateCompatibleDC(0);
	hOldFont_ = (HFONT)SelectObject(hDC_, hFont_);

	int strBaseW = 0, strBaseH = 0;
	char buf[1 << 12];
	strcpy_s(buf, str);
	char* bufT[1 << 12];  // 这个用于分隔字符串后剩余的字符，可能会超出。
	//处理多行
	{
		int nnh = 0;
		int cw, ch;

		const char* ln = strtok_s(buf, "\n", bufT);
		while (ln != 0)
		{
			GetStringSize(hDC_, ln, &cw, &ch);
			strBaseW = max(strBaseW, cw);
			strBaseH = max(strBaseH, ch);

			ln = strtok_s(0, "\n", bufT);
			nnh++;
		}
		singleRow_ = strBaseH;
		strBaseH *= nnh;
	}

	BITMAPINFO bmp = { 0 };
	BITMAPINFOHEADER& bih = bmp.bmiHeader;
	drawLineStep_ = strBaseW * 3 % 4 == 0 ? strBaseW * 3 : (strBaseW * 3 + 4 - ((strBaseW * 3) % 4));

	bih.biSize = sizeof(BITMAPINFOHEADER);
	bih.biWidth = strBaseW;
	bih.biHeight = strBaseH;
	bih.biPlanes = 1;
	bih.biBitCount = 24;
	bih.biCompression = BI_RGB;
	bih.biSizeImage = strBaseH * drawLineStep_;
	bih.biClrUsed = 0;
	bih.biClrImportant = 0;

	hBmp_ = CreateDIBSection(hDC_, &bmp, DIB_RGB_COLORS, &pDibData_, 0, 0);

	CV_Assert(pDibData_ != 0);
	hOldBmp_ = (HBITMAP)SelectObject(hDC_, hBmp_);

	SetTextColor(hDC_, RGB(255, 255, 255));
	SetBkColor(hDC_, 0);

	// 计算叠加坐标
	int startX = 0;
	switch (horizontalAlignment)
	{
	case OSDHorizontalAlignment::Left: startX = startPoint.x; break;
	case OSDHorizontalAlignment::HCenter: startX = imageWidth / 2 - strBaseW / 2 + startPoint.x; break;
	case OSDHorizontalAlignment::Right: startX = imageWidth - strBaseW + (-startPoint.x); break;
	}

	int startY = 0;
	switch (verticalAlignment)
	{
	case OSDVerticalAlignment::Top: startY = startPoint.y; break;
	case OSDVerticalAlignment::VCenter: startY = imageHeight / 2 - strBaseH / 2 + startPoint.y; break;
	case OSDVerticalAlignment::Bottom: startY = imageHeight - strBaseH + (-startPoint.y); break;
	}

	OverlayCoordinate = Rect(startX, startY, strBaseW, strBaseH);
}

/// <summary>
/// 叠加日期时间 OSD
/// </summary>
/// <param name="dst">OpenCV Mat</param>
void OverlayOSD::OverlayDateTime(Mat& dst)
{
	SYSTEMTIME  systm;
	GetLocalTime(&systm);

	char text[1024];
	int n = 0;
	n += sprintf_s(text + n, sizeof(text) - n - 1, "%d-%02d-%02d",
		systm.wYear, systm.wMonth, systm.wDay);

	n += sprintf_s(text + n, sizeof(text) - n - 1, " %02d:%02d:%02d",
		systm.wHour, systm.wMinute, systm.wSecond);

	char buf[1 << 12];
	char* bufT[1 << 12];

	strcpy_s(buf, text);
	const char* ln = strtok_s(buf, "\n", bufT);
	int outTextY = 0;
	while (ln != 0)
	{
		TextOutA(hDC_, 0, outTextY, ln, strlen(ln));
		outTextY += singleRow_;
		ln = strtok_s(0, "\n", bufT);
	}

	int x, y, r, b;
	Point org = Point(OverlayCoordinate.x, OverlayCoordinate.y);
	int strBaseW = OverlayCoordinate.width;
	int strBaseH = OverlayCoordinate.height;
	if (org.x > dst.cols || org.y > dst.rows) return;
	x = org.x < 0 ? -org.x : 0;
	y = org.y < 0 ? -org.y : 0;

	r = org.x + strBaseW > dst.cols ? dst.cols - org.x - 1 : strBaseW - 1;
	b = org.y + strBaseH > dst.rows ? dst.rows - org.y - 1 : strBaseH - 1;
	org.x = org.x < 0 ? 0 : org.x;
	org.y = org.y < 0 ? 0 : org.y;

	Scalar color = Scalar(255, 255, 255);

	uchar* dstData = (uchar*)dst.data;
	int dstStep = dst.step / sizeof(dstData[0]);
	unsigned char* pImg = (unsigned char*)dst.data + org.x * dst.channels() + org.y * dstStep;
	unsigned char* pStr = (unsigned char*)pDibData_ + x * 3;
	for (int tty = y; tty <= b; ++tty)
	{
		unsigned char* subImg = pImg + (tty - y) * dstStep;
		unsigned char* subStr = pStr + (strBaseH - tty - 1) * drawLineStep_;
		for (int ttx = x; ttx <= r; ++ttx)
		{
			for (int n = 0; n < dst.channels(); ++n) {
				double vtxt = subStr[n] / 255.0;
				int cvv = vtxt * color.val[n] + (1 - vtxt) * subImg[n];
				subImg[n] = cvv > 255 ? 255 : (cvv < 0 ? 0 : cvv);
			}

			subStr += 3;
			subImg += dst.channels();
		}
	}
}

/// <summary>
/// 叠加日期时间 OSD
/// </summary>
/// <param name="YData">Y 分量</param>
/// <param name="UVData">UV 分量</param>
void OverlayOSD::OverlayDateTime(unsigned char* yData, unsigned char* uvData)
{
	SYSTEMTIME  systm;
	GetLocalTime(&systm);

	char text[1024];
	int n = 0;
	n += sprintf_s(text + n, sizeof(text) - n - 1, "%d-%02d-%02d",
		systm.wYear, systm.wMonth, systm.wDay);

	n += sprintf_s(text + n, sizeof(text) - n - 1, " %02d:%02d:%02d",
		systm.wHour, systm.wMinute, systm.wSecond);

	char buf[1 << 12];
	char* bufT[1 << 12];

	strcpy_s(buf, text);
	const char* ln = strtok_s(buf, "\n", bufT);
	int outTextY = 0;
	while (ln != 0)
	{
		TextOutA(hDC_, 0, outTextY, ln, strlen(ln));
		outTextY += singleRow_;
		ln = strtok_s(0, "\n", bufT);
	}

	auto osdText = (unsigned char*)pDibData_;

	auto osd_size = sizeof(unsigned char) * OverlayCoordinate.width * OverlayCoordinate.height * 3;
	unsigned char* osdData = nullptr;
	cudaMalloc(&osdData, osd_size);
	cudaMemcpy(osdData, osdText, osd_size, cudaMemcpyHostToDevice);

	OSDNearest(yData, uvData, imageWidth_, osdData, OverlayCoordinate, drawLineStep_);

	cudaFree(osdData);
}

/// <summary>
/// 叠加文本 OSD
/// </summary>
/// <param name="dst">OpenCV Mat</param>
void OverlayOSD::OverlayText(Mat& dst)
{
	char buf[1 << 12];
	char* bufT[1 << 12];

	strcpy_s(buf, overlayText_);
	const char* ln = strtok_s(buf, "\n", bufT);
	int outTextY = 0;
	while (ln != 0)
	{
		TextOutA(hDC_, 0, outTextY, ln, strlen(ln));
		outTextY += singleRow_;
		ln = strtok_s(0, "\n", bufT);
	}

	int x, y, r, b;
	Point org = Point(OverlayCoordinate.x, OverlayCoordinate.y);
	int strBaseW = OverlayCoordinate.width;
	int strBaseH = OverlayCoordinate.height;
	if (org.x > dst.cols || org.y > dst.rows) return;
	x = org.x < 0 ? -org.x : 0;
	y = org.y < 0 ? -org.y : 0;

	r = org.x + strBaseW > dst.cols ? dst.cols - org.x - 1 : strBaseW - 1;
	b = org.y + strBaseH > dst.rows ? dst.rows - org.y - 1 : strBaseH - 1;
	org.x = org.x < 0 ? 0 : org.x;
	org.y = org.y < 0 ? 0 : org.y;

	Scalar color = Scalar(255, 255, 255);

	uchar* dstData = (uchar*)dst.data;
	int dstStep = dst.step / sizeof(dstData[0]);
	unsigned char* pImg = (unsigned char*)dst.data + org.x * dst.channels() + org.y * dstStep;
	unsigned char* pStr = (unsigned char*)pDibData_ + x * 3;
	for (int tty = y; tty <= b; ++tty)
	{
		unsigned char* subImg = pImg + (tty - y) * dstStep;
		unsigned char* subStr = pStr + (strBaseH - tty - 1) * drawLineStep_;
		for (int ttx = x; ttx <= r; ++ttx)
		{
			for (int n = 0; n < dst.channels(); ++n) {
				double vtxt = subStr[n] / 255.0;
				int cvv = vtxt * color.val[n] + (1 - vtxt) * subImg[n];
				subImg[n] = cvv > 255 ? 255 : (cvv < 0 ? 0 : cvv);
			}

			subStr += 3;
			subImg += dst.channels();
		}
	}
}

/// <summary>
/// 叠加文本 OSD
/// </summary>
/// <param name="YData">Y 分量</param>
/// <param name="UVData">UV 分量</param>
void OverlayOSD::OverlayText(unsigned char* yData, unsigned char* uvData)
{
	char buf[1 << 12];
	char* bufT[1 << 12];

	strcpy_s(buf, overlayText_);
	const char* ln = strtok_s(buf, "\n", bufT);
	int outTextY = 0;
	while (ln != 0)
	{
		TextOutA(hDC_, 0, outTextY, ln, strlen(ln));
		outTextY += singleRow_;
		ln = strtok_s(0, "\n", bufT);
	}

	auto osdText = (unsigned char*)pDibData_;

	auto osd_size = sizeof(unsigned char) * OverlayCoordinate.width * OverlayCoordinate.height * 3;
	unsigned char* osdData = nullptr;
	cudaMalloc(&osdData, osd_size);
	cudaMemcpy(osdData, osdText, osd_size, cudaMemcpyHostToDevice);

	OSDNearest(yData, uvData, imageWidth_, osdData, OverlayCoordinate, drawLineStep_);

	cudaFree(osdData);
}

/// <summary>
/// 释放所有资源
/// </summary>
void OverlayOSD::Dispose()
{
	SelectObject(hDC_, hOldBmp_);
	SelectObject(hDC_, hOldFont_);
	DeleteObject(hBmp_);
	DeleteObject(hFont_);
	DeleteDC(hDC_);
}
```

### 调用方法
<img src="https://sadness96.github.io/images/blog/cpp-OverlayOSD/VideoOSD.jpg"/>

直接叠加单色文本，可能会与视频颜色完美融合，有两种解决方法：
1. 添加一个半透明矩形，既不影响视频内容又可以凸显出文字，可能不太美观。
1. 文字描边，使用白字黑边。
1. 使用反色叠加，如果亮度过高的像素使用黑色，亮度过低的使用白色，测试在单个像素计算显示可能会显得很凌乱，取一个区域的亮度整体调色或许会好很多。

#### 以 Mat 方式叠加
由于 OpenCV 无法直接叠加半透明矩形，所以使用 [addWeighted](https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html) 的方式叠加。
``` cpp
// 初始化 OSD 配置
OverlayOSD osd1 = OverlayOSD();
osd1.InitOSD("1970-01-01 00:00:00", 80, "黑体", width, height, Top, Left, Point(50, 50));
OverlayOSD osd2 = OverlayOSD();
osd2.InitOSD("测试文本OSD", 80, "黑体", width, height, Bottom, Right, Point(50, 50));

while (true)
{
    ...
    // 叠加 OSD Mat
    Scalar background = Scalar(0, 0, 0);
    double alpha = 0.4;

    Mat roi1 = mat(osd1.OverlayCoordinate);
    Mat color1(roi1.size(), CV_8UC3, background);
    addWeighted(color1, alpha, roi1, 1.0 - alpha, 0.0, roi1);
    osd1.OverlayDateTime(mat);

    Mat roi2 = mat(osd2.OverlayCoordinate);
    Mat color2(roi2.size(), CV_8UC3, background);
    addWeighted(color2, alpha, roi2, 1.0 - alpha, 0.0, roi2);
    osd2.OverlayText(mat);
    ...
}

// 释放 OSD 资源
osd1.Dispose();
osd2.Dispose();
```

#### 以 CUDA 方式叠加
使用 CUDA 方式的叠加半透明矩形仅需要在 CUDA 代码中的 Y 分量减去一定值即可。
``` cpp
// 初始化 OSD 配置
OverlayOSD osd1 = OverlayOSD();
osd1.InitOSD("1970-01-01 00:00:00", 80, "黑体", width, height, Top, Left, Point(50, 50));
OverlayOSD osd2 = OverlayOSD();
osd2.InitOSD("测试文本OSD", 80, "黑体", width, height, Bottom, Right, Point(50, 50));

while (true)
{
    ...
    // 叠加 OSD CUDA
    osd1.OverlayDateTime(outYData, outUVData);
    osd2.OverlayText(outYData, outUVData);
    ...
}

// 释放 OSD 资源
osd1.Dispose();
osd2.Dispose();
```