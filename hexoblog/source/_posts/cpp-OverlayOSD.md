---
title: 视频中叠加 OSD
date: 2023-01-18 23:16:32
tags: [c++,ffmpeg,opencv,cuda]
categories: C++
---
### 使用的 Windows LOGFONTA 叠加 OSD
<!-- more -->
### 简介
在监控视频中普遍需要在视频上叠加时间戳与相机所在位置，方便回放监控时快速确认有效信息，在处理其他视频时也可以叠加水印等标识。  
如果仅使用 FFmpeg 编解码，可以使用 [AVFilter](https://ffmpeg.org/doxygen/trunk/structAVFilter.html) 一系列方法给视频添加水印或文字信息，但是使用上多有不便，同时想解决 OpenCV Mat 无法叠加汉字的问题，所以使用了另一种方式。
参考文章：[Opencv310图片Mat中叠加汉字](https://blog.csdn.net/zmdsjtu/article/details/53133223) 中使用的 Windows [LOGFONTA](https://learn.microsoft.com/zh-cn/windows/win32/api/dimm/ns-dimm-logfonta?redirectedfrom=MSDN) 创建的位图，以 CUDA 的方式叠加到视频中。

### 实现效果
<img src="https://sadness96.github.io/images/blog/cpp-OverlayOSD/OverlayOSD.jpg"/>

直接叠加单色文本，可能会与视频颜色重叠，导致 OSD 内容不清晰，有以下几种解决方法，目前实现使用半透明矩形。
1. 添加一个半透明矩形，既不影响视频内容又可以凸显出文字。
1. 文字描边，普遍使用白字黑边。
1. 使用反色叠加，如果亮度过高的像素使用黑色，亮度过低的使用白色，测试在单个像素计算显示可能会显得很凌乱，取一个区域的亮度整体调色或许会好很多。

### 核心代码
#### OSDConfig.h
``` cpp
#pragma once

/// <summary>
/// OSD 类型
/// </summary>
enum OSDType {
	/// <summary>
	/// 空值(未选择)
	/// </summary>
	NullValue,
	/// <summary>
	/// 日期时间类型
	/// </summary>
	DateTime,
	/// <summary>
	/// 固定文本类型
	/// </summary>
	FixedText
};

/// <summary>
/// OSD 样式类型
/// </summary>
enum OSDStyleType {
	/// <summary>
	/// 半透明底白字
	/// </summary>
	TransparentBackgroundWhiteFont,
	/// <summary>
	/// 半透明底黑字
	/// </summary>
	TransparentBackgroundBlackFont,
	/// <summary>
	/// 白字黑边框
	/// </summary>
	WhiteFontWithBlackBorder,
	/// <summary>
	/// 黑字白边框
	/// </summary>
	BlackFontWithWhiteBorder,
	/// <summary>
	/// 黑白亮度反色
	/// </summary>
	BlackandWhiteBrightnessInversion
};

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

/// <summary>
/// 拼接 OSD 配置
/// </summary>
struct OSDConfig
{
	/// <summary>
	/// OSD 类型
	/// </summary>
	OSDType osdType;

	/// <summary>
	/// 日期时间类型
	/// 0：yyyy-MM-dd HH:mm:ss
	/// 1：yyyy-MM-dd hh:mm:ss AM/PM
	/// 2：yyyy年MM月dd日 HH:mm:ss
	/// 3：yyyy年MM月dd日 hh:mm:ss AM/PM
	/// </summary>
	int dateTimeType;

	/// <summary>
	/// 固定文本内容
	/// </summary>
	char* fixedText;

	/// <summary>
	/// 样式类型
	/// </summary>
	OSDStyleType osdStyleType;

	/// <summary>
	/// 文字大小
	/// </summary>
	int fontSize;

	/// <summary>
	/// 文字字体
	/// </summary>
	char* fontFamily;

	/// <summary>
	/// 垂直对齐方式
	/// </summary>
	OSDVerticalAlignment verticalAlignment;

	/// <summary>
	/// 垂直相对坐标
	/// </summary>
	int verticalCoordinate;

	/// <summary>
	/// 水平对齐方式
	/// </summary>
	OSDHorizontalAlignment horizontalAlignment;

	/// <summary>
	/// 水平相对坐标
	/// </summary>
	int horizontalCoordinate;
};
```

#### OverlayOSD.h
``` cpp
#pragma once

#include <windows.h>
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
#include <string>
#include <opencv2/opencv.hpp>
#include "OSDConfig.h"

using namespace cv;

class OverlayOSD
{
private:
	/// <summary>
	/// OSD 配置
	/// </summary>
	OSDConfig osdConfig_;

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

	/// <summary>
	/// 根据类型获取日期时间
	/// </summary>
	/// <param name="dateTimeType">
	/// 0：yyyy-MM-dd HH:mm:ss
	/// 1：yyyy-MM-dd hh:mm:ss AM/PM
	/// 2：yyyy年MM月dd日 HH:mm:ss
	/// 3：yyyy年MM月dd日 hh:mm:ss AM/PM
	/// </param>
	/// <param name="dateTimeText">日期时间</param>
	void GetDateTime(int dateTimeType, char* text);

public:
	/// <summary>
	/// 叠加坐标
	/// </summary>
	Rect OverlayCoordinate;

	/// <summary>
	/// 初始化 OSD
	/// </summary>
	/// <param name="osdConfig">OSD 配置</param>
	/// <param name="imageWidth">叠加图像宽度</param>
	/// <param name="imageHeight">叠加图像高度</param>
	void InitOSD(OSDConfig osdConfig, int imageWidth, int imageHeight);

	/// <summary>
	/// 叠加文本 Mat
	/// </summary>
	/// <param name="dst">OpenCV Mat</param>
	void OverlayText(Mat& dst);

	/// <summary>
	/// 叠加文本 YUV NV12
	/// </summary>
	/// <param name="yData">YUV NV12 Y 分量</param>
	/// <param name="uvData">YUV NV12 UV 分量</param>
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
/// <param name="imageHeight">图像高度</param>
/// <param name="osdData">叠加 OSD 数据 RGB</param>
/// <param name="osdRect">叠加 OSD 参数</param>
/// <param name="drawLineStep">叠加 OSD 行大小</param>
/// <param name="osdStyleType">OSD 样式类型</param>
/// <returns></returns>
extern "C" bool OSDNearest(unsigned char* yData, unsigned char* uvData, int imageWidth, int imageHeight,
	unsigned char* osdData, Rect osdRect, int drawLineStep, OSDStyleType osdStyleType);
```

#### OverlayOSD.cpp
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
/// 根据类型获取日期时间
/// </summary>
/// <param name="dateTimeType">
/// 0：yyyy-MM-dd HH:mm:ss
/// 1：yyyy-MM-dd hh:mm:ss AM/PM
/// 2：yyyy年MM月dd日 HH:mm:ss
/// 3：yyyy年MM月dd日 hh:mm:ss AM/PM
/// </param>
/// <param name="dateTimeText">日期时间</param>
void OverlayOSD::GetDateTime(int dateTimeType, char* dateTimeText)
{
	SYSTEMTIME  systm;
	GetLocalTime(&systm);

	char text[1024];
	int n = 0;

	if (dateTimeType == 0)
	{
		// yyyy-MM-dd HH:mm:ss
		n += sprintf_s(text + n, sizeof(text) - n - 1, "%d-%02d-%02d",
			systm.wYear, systm.wMonth, systm.wDay);

		n += sprintf_s(text + n, sizeof(text) - n - 1, " %02d:%02d:%02d",
			systm.wHour, systm.wMinute, systm.wSecond);
	}
	else if (dateTimeType == 1)
	{
		// yyyy-MM-dd hh:mm:ss AM/PM
		n += sprintf_s(text + n, sizeof(text) - n - 1, "%d-%02d-%02d",
			systm.wYear, systm.wMonth, systm.wDay);

		int hour = systm.wHour % 12;
		if (hour == 0)
			hour = 12;

		n += sprintf_s(text + n, sizeof(text) - n - 1, " %02d:%02d:%02d",
			hour, systm.wMinute, systm.wSecond);

		n += sprintf_s(text + n, sizeof(text) - n - 1, " %s",
			systm.wHour < 12 ? "AM" : "PM");

	}
	else if (dateTimeType == 2)
	{
		// yyyy年MM月dd日 HH:mm:ss
		n += sprintf_s(text + n, sizeof(text) - n - 1, "%d年%02d月%02d日",
			systm.wYear, systm.wMonth, systm.wDay);

		n += sprintf_s(text + n, sizeof(text) - n - 1, " %02d:%02d:%02d",
			systm.wHour, systm.wMinute, systm.wSecond);
	}
	else if (dateTimeType == 3)
	{
		// yyyy年MM月dd日 hh:mm:ss AM/PM
		n += sprintf_s(text + n, sizeof(text) - n - 1, "%d年%02d月%02d日",
			systm.wYear, systm.wMonth, systm.wDay);

		int hour = systm.wHour % 12;
		if (hour == 0)
			hour = 12;

		n += sprintf_s(text + n, sizeof(text) - n - 1, " %02d:%02d:%02d",
			hour, systm.wMinute, systm.wSecond);

		n += sprintf_s(text + n, sizeof(text) - n - 1, " %s",
			systm.wHour < 12 ? "AM" : "PM");
	}

	strcpy(dateTimeText, text);
}

/// <summary>
/// 初始化 OSD
/// </summary>
/// <param name="osdConfig">OSD 配置</param>
/// <param name="imageWidth">叠加图像宽度</param>
/// <param name="imageHeight">叠加图像高度</param>
void OverlayOSD::InitOSD(OSDConfig osdConfig, int imageWidth, int imageHeight)
{
	osdConfig_ = osdConfig;

	char text[1024];
	if (osdConfig_.osdType == DateTime)
	{
		GetDateTime(osdConfig_.dateTimeType, text);
	}
	else if (osdConfig_.osdType == FixedText)
	{
		strcpy(text, osdConfig_.fixedText);
	}

	overlayText_ = text;
	imageWidth_ = imageWidth;
	imageHeight_ = imageHeight;

	logFonta_.lfHeight = -osdConfig_.fontSize;
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
	strcpy_s(logFonta_.lfFaceName, osdConfig_.fontFamily);

	hFont_ = CreateFontIndirectA(&logFonta_);
	hDC_ = CreateCompatibleDC(0);
	hOldFont_ = (HFONT)SelectObject(hDC_, hFont_);

	int strBaseW = 0, strBaseH = 0;
	char buf[1 << 12];
	strcpy_s(buf, text);
	// 这个用于分隔字符串后剩余的字符，可能会超出。
	char* bufT[1 << 12];
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
	Point startPoint = Point(osdConfig_.horizontalCoordinate, osdConfig_.verticalCoordinate);

	int startX = 0;
	switch (osdConfig_.horizontalAlignment)
	{
	case OSDHorizontalAlignment::Left: startX = startPoint.x; break;
	case OSDHorizontalAlignment::HCenter: startX = imageWidth / 2 - strBaseW / 2 + startPoint.x; break;
	case OSDHorizontalAlignment::Right: startX = imageWidth - strBaseW + (-startPoint.x); break;
	}

	int startY = 0;
	switch (osdConfig_.verticalAlignment)
	{
	case OSDVerticalAlignment::Top: startY = startPoint.y; break;
	case OSDVerticalAlignment::VCenter: startY = imageHeight / 2 - strBaseH / 2 + startPoint.y; break;
	case OSDVerticalAlignment::Bottom: startY = imageHeight - strBaseH + (-startPoint.y); break;
	}

	OverlayCoordinate = Rect(startX, startY, strBaseW, strBaseH);
}

/// <summary>
/// 叠加文本 Mat
/// </summary>
/// <param name="dst">OpenCV Mat</param>
void OverlayOSD::OverlayText(Mat& dst)
{
	char text[1024];
	if (osdConfig_.osdType == DateTime)
	{
		GetDateTime(osdConfig_.dateTimeType, text);
	}
	else if (osdConfig_.osdType == FixedText)
	{
		strcpy(text, osdConfig_.fixedText);
	}

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
/// 叠加文本 YUV NV12
/// </summary>
/// <param name="yData">YUV NV12 Y 分量</param>
/// <param name="uvData">YUV NV12 UV 分量</param>
void OverlayOSD::OverlayText(unsigned char* yData, unsigned char* uvData)
{
	char text[1024];
	if (osdConfig_.osdType == DateTime)
	{
		GetDateTime(osdConfig_.dateTimeType, text);
	}
	else if (osdConfig_.osdType == FixedText)
	{
		strcpy(text, osdConfig_.fixedText);
	}

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

	OSDNearest(yData, uvData, imageWidth_, imageHeight_, osdData, OverlayCoordinate, drawLineStep_, osdConfig_.osdStyleType);

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

#### OverlayOSD.cu
``` cpp
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
#include <core/types.hpp>
#include "OSDConfig.h"

using namespace cv;

/// <summary>
/// 叠加 OSD 核函数
/// 半透明黑低白字
/// 半透明白低黑字
/// </summary>
/// <param name="yData">图像 Y 分量</param>
/// <param name="uvData">图像 UV 分量</param>
/// <param name="imageWidth">图像宽度</param>
/// <param name="isWhite">是否白字</param>
/// <param name="osdData">叠加 OSD 数据 RGB</param>
/// <param name="osdRect">叠加 OSD 参数</param>
/// <param name="drawLineStep">叠加 OSD 行大小</param>
/// <returns></returns>
__global__ void OSDNearestKernel_Background(unsigned char* yData, unsigned char* uvData, int imageWidth, bool isWhite,
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

		int idx_osd = drawLineStep * (osdRect.height - tidy - 1) + tidx * 3;

		if (osdData[idx_osd] > 0)
		{
			if (isWhite)
			{
				yData[idx_in_y] = 255;
				uvData[srcX % 2 == 0 ? idx_in_uv : idx_in_uv - 1] = 128;
				uvData[srcX % 2 == 0 ? idx_in_uv + 1 : idx_in_uv] = 128;
			}
			else
			{
				yData[idx_in_y] = 0;
			}
		}
		else
		{
			if (isWhite)
			{
				yData[idx_in_y] = yData[idx_in_y] >= 40 ? yData[idx_in_y] - 40 : 0;
			}
			else
			{
				yData[idx_in_y] = yData[idx_in_y] <= 255 - 40 ? yData[idx_in_y] + 40 : 255;
			}
		}
	}
}

/// <summary>
/// 叠加 OSD 核函数
/// 白字黑边框
/// 黑字白边框
/// </summary>
/// <param name="yData">图像 Y 分量</param>
/// <param name="uvData">图像 UV 分量</param>
/// <param name="imageWidth">图像宽度</param>
/// <param name="isWhite">是否白字</param>
/// <param name="osdData">叠加 OSD 数据 RGB</param>
/// <param name="osdRect">叠加 OSD 参数</param>
/// <param name="drawLineStep">叠加 OSD 行大小</param>
/// <returns></returns>
__global__ void OSDNearestKernel_Border(unsigned char* yData, unsigned char* uvData, int imageWidth, bool isWhite,
	unsigned char* osdData, Rect osdRect, int drawLineStep)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	int n = 2;

	if (tidx < osdRect.width && tidy < osdRect.height)
	{
		int srcX = tidx + osdRect.x;
		int srcY = tidy + osdRect.y;

		int idx_in_y = srcY * imageWidth + srcX;
		int idx_in_uv = srcY / 2 * imageWidth + srcX;

		int idx_osd = drawLineStep * (osdRect.height - tidy - 1) + tidx * 3;

		if (osdData[idx_osd] > 0)
		{
			if (isWhite)
			{
				yData[idx_in_y] = 255;
				uvData[srcX % 2 == 0 ? idx_in_uv : idx_in_uv - 1] = 128;
				uvData[srcX % 2 == 0 ? idx_in_uv + 1 : idx_in_uv] = 128;
			}
			else
			{
				yData[idx_in_y] = 0;
			}
		}
		else
		{
			// 绘制外边框
			for (int i = -n; i <= n; i++)
			{
				for (int j = -n; j <= n; j++)
				{
					int idx_osd_border = idx_osd + i * drawLineStep + j * 3;
					if (idx_osd_border > 0 && idx_osd_border < drawLineStep * osdRect.height && osdData[idx_osd_border] > 0)
					{
						if (isWhite)
						{
							yData[idx_in_y] = 0;
						}
						else
						{
							yData[idx_in_y] = 255;
							uvData[srcX % 2 == 0 ? idx_in_uv : idx_in_uv - 1] = 128;
							uvData[srcX % 2 == 0 ? idx_in_uv + 1 : idx_in_uv] = 128;
						}
						return;
					}
				}
			}
		}
	}
}

/// <summary>
/// 叠加 OSD 核函数
/// 反色
/// </summary>
/// <param name="yData">图像 Y 分量</param>
/// <param name="uvData">图像 UV 分量</param>
/// <param name="imageWidth">图像宽度</param>
/// <param name="imageHeight">图像高度</param>
/// <param name="osdData">叠加 OSD 数据 RGB</param>
/// <param name="osdRect">叠加 OSD 参数</param>
/// <param name="drawLineStep">叠加 OSD 行大小</param>
/// <returns></returns>
__global__ void OSDNearestKernel_Inverse(unsigned char* yData, unsigned char* uvData, int imageWidth, int imageHeight,
	unsigned char* osdData, Rect osdRect, int drawLineStep)
{
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;

	int n = 15;

	if (tidx < osdRect.width && tidy < osdRect.height)
	{
		int srcX = tidx + osdRect.x;
		int srcY = tidy + osdRect.y;

		int idx_in_y = srcY * imageWidth + srcX;
		int idx_in_uv = srcY / 2 * imageWidth + srcX;

		int idx_osd = drawLineStep * (osdRect.height - tidy - 1) + tidx * 3;

		if (osdData[idx_osd] > 0)
		{
			int dark = 0;
			int bright = 0;

			for (int i = -n; i <= n; i++)
			{
				for (int j = -n; j <= n; j++)
				{
					int idx_osd_y = (srcY + i) * imageWidth + (srcX + j);
					if (idx_osd_y > 0 && idx_osd_y < imageWidth * imageHeight)
					{
						if (yData[idx_osd_y] > 128)
						{
							bright++;
						}
						else
						{
							dark++;
						}
					}
				}
			}

			if (bright < dark)
			{
				yData[idx_in_y] = 255;
				uvData[srcX % 2 == 0 ? idx_in_uv : idx_in_uv - 1] = 128;
				uvData[srcX % 2 == 0 ? idx_in_uv + 1 : idx_in_uv] = 128;
			}
			else
			{
				yData[idx_in_y] = 0;
			}
		}
	}
}

/// <summary>
/// 叠加 OSD
/// </summary>
/// <param name="yData">图像 Y 分量</param>
/// <param name="uvData">图像 UV 分量</param>
/// <param name="imageWidth">图像宽度</param>
/// <param name="imageHeight">图像高度</param>
/// <param name="osdData">叠加 OSD 数据 RGB</param>
/// <param name="osdRect">叠加 OSD 参数</param>
/// <param name="drawLineStep">叠加 OSD 行大小</param>
/// <param name="osdStyleType">OSD 样式类型</param>
/// <returns></returns>
extern "C" bool OSDNearest(unsigned char* yData, unsigned char* uvData, int imageWidth, int imageHeight,
	unsigned char* osdData, Rect osdRect, int drawLineStep, OSDStyleType osdStyleType)
{
	dim3 block(32, 32);
	dim3 grid((osdRect.width + block.x - 1) / block.x, (osdRect.height + block.y - 1) / block.y);

	switch (osdStyleType)
	{
	case TransparentBackgroundWhiteFont:
		// 半透明底白字
		OSDNearestKernel_Background << <grid, block >> > (yData, uvData, imageWidth, true, osdData, osdRect, drawLineStep);
		break;
	case TransparentBackgroundBlackFont:
		// 半透明底黑字
		OSDNearestKernel_Background << <grid, block >> > (yData, uvData, imageWidth, false, osdData, osdRect, drawLineStep);
		break;
	case WhiteFontWithBlackBorder:
		// 白字黑边框
		OSDNearestKernel_Border << <grid, block >> > (yData, uvData, imageWidth, true, osdData, osdRect, drawLineStep);
		break;
	case BlackFontWithWhiteBorder:
		// 黑字白边框
		OSDNearestKernel_Border << <grid, block >> > (yData, uvData, imageWidth, false, osdData, osdRect, drawLineStep);
		break;
	case BlackandWhiteBrightnessInversion:
		// 黑白亮度反色
		OSDNearestKernel_Inverse << <grid, block >> > (yData, uvData, imageWidth, imageHeight, osdData, osdRect, drawLineStep);
		break;
	default:
		// 默认：半透明底白字
		OSDNearestKernel_Background << <grid, block >> > (yData, uvData, imageWidth, true, osdData, osdRect, drawLineStep);
	}

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

#### 测试调用代码
``` cpp
#include "OverlayOSD.h"

int width = 3840;
int height = 2160;

auto video_y_size = sizeof(unsigned char) * width * height;
auto video_uv_size = sizeof(unsigned char) * width * (height / 2);

unsigned char* outYData = nullptr;
cudaMalloc(&outYData, video_y_size);

unsigned char* outUVData = nullptr;
cudaMalloc(&outUVData, video_uv_size);
cudaMemset(outUVData, 0x80, video_uv_size);

string strFontFamily = "黑体";
string strFixedText = "测试视频 OSD";
vector<OverlayOSD> osds;

for (size_t i = 0; i < 5; i++)
{
	OSDConfig osdConfigDateTime;
	osdConfigDateTime.osdType = DateTime;
	osdConfigDateTime.dateTimeType = 0;

	switch (i)
	{
	case 0:osdConfigDateTime.osdStyleType = TransparentBackgroundWhiteFont; break;
	case 1:osdConfigDateTime.osdStyleType = TransparentBackgroundBlackFont; break;
	case 2:osdConfigDateTime.osdStyleType = WhiteFontWithBlackBorder; break;
	case 3:osdConfigDateTime.osdStyleType = BlackFontWithWhiteBorder; break;
	case 4:osdConfigDateTime.osdStyleType = BlackandWhiteBrightnessInversion; break;
	default:
		osdConfigDateTime.osdStyleType = TransparentBackgroundWhiteFont; break;
	}

	osdConfigDateTime.fontSize = 80;
	osdConfigDateTime.fontFamily = (char*)strFontFamily.c_str();
	osdConfigDateTime.verticalAlignment = Top;
	osdConfigDateTime.verticalCoordinate = 50 + (i * 100);
	osdConfigDateTime.horizontalAlignment = Left;
	osdConfigDateTime.horizontalCoordinate = 50;

	OverlayOSD osdDateTime = OverlayOSD();
	osdDateTime.InitOSD(osdConfigDateTime, width, height);
	osds.push_back(osdDateTime);
}

for (size_t i = 0; i < 5; i++)
{
	OSDConfig osdConfigFixedText;
	osdConfigFixedText.osdType = FixedText;
	osdConfigFixedText.fixedText = (char*)strFixedText.c_str();

	switch (i)
	{
	case 0:osdConfigFixedText.osdStyleType = TransparentBackgroundWhiteFont; break;
	case 1:osdConfigFixedText.osdStyleType = TransparentBackgroundBlackFont; break;
	case 2:osdConfigFixedText.osdStyleType = WhiteFontWithBlackBorder; break;
	case 3:osdConfigFixedText.osdStyleType = BlackFontWithWhiteBorder; break;
	case 4:osdConfigFixedText.osdStyleType = BlackandWhiteBrightnessInversion; break;
	default:
		osdConfigFixedText.osdStyleType = TransparentBackgroundWhiteFont; break;
	}

	osdConfigFixedText.fontSize = 80;
	osdConfigFixedText.fontFamily = (char*)strFontFamily.c_str();
	osdConfigFixedText.verticalAlignment = Bottom;
	osdConfigFixedText.verticalCoordinate = 50 + (i * 100);
	osdConfigFixedText.horizontalAlignment = Right;
	osdConfigFixedText.horizontalCoordinate = 50;

	OverlayOSD osdFixedText = OverlayOSD();
	osdFixedText.InitOSD(osdConfigFixedText, width, height);
	osds.push_back(osdFixedText);
}

while (true)  
{  
    ...
	// 叠加 OSD
	for (size_t i = 0; i < osds.size(); i++)
	{
		osds[i].OverlayText(outYData, outUVData);
	}
    ...
}

// 释放 OSD 资源
for (size_t i = 0; i < osds.size(); i++)
{
	osds[i].Dispose();
}
```