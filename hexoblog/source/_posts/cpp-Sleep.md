---
title: C++ Sleep 精度不够
date: 2023-08-12 12:45:52
tags: [c++]
categories: C++
---
### 解决 Sleep 精度不够问题
<!-- more -->
### 简介
在处理媒体视频时需要阻塞等待控制间隔，例如监控视频为 25FPS，就需要设置每帧间隔为 40 毫秒，在阻塞等待函数不够精准的情况下，就会出现帧数达不到的情况，可以在代码中提高帧数也就是降低间隔时间给函数留出精度不够的误差，但这似乎不是一个好办法，参考文章 [Windows几种sleep精度的测试，结果基于微秒](https://blog.csdn.net/liuhengxiao/article/details/99641539) 做的一些测试，根据项目需要选择更适合的方法。

### 核心代码
#### 测试代码
使用以下代码作为测试，运行时间越趋近于等待函数设置的值，精度越高。
``` cpp
#include <iostream>
#include <chrono>
#include <thread>
#include <Windows.h>
#include <mmsystem.h>
#include <timeapi.h>

using namespace std;
using namespace std::this_thread;
using namespace chrono;

int main()
{
	int time = 40;
	for (;;)
	{
		steady_clock::time_point start = steady_clock::now();

		// 等待函数

		steady_clock::time_point end = steady_clock::now();
		cout << "运行时间：" << duration_cast<nanoseconds>(end - start).count() << endl;
	}
}
```

#### sleep_for
使用 c++ 11 自带函数设置 sleep 间隔为 40 毫秒。

``` cpp
int main()
{
	int time = 40;
	for (;;)
	{
		steady_clock::time_point start = steady_clock::now();

		sleep_for(milliseconds(time));

		steady_clock::time_point end = steady_clock::now();
		cout << "运行时间：" << duration_cast<nanoseconds>(end - start).count() << endl;
	}
}
```

精度相差 2 - 8 毫秒，对于有精度需求的程序无法使用。

``` cmd
运行时间：44549900
运行时间：47305000
运行时间：46158900
运行时间：46949000
运行时间：45595100
运行时间：45570100
运行时间：45340200
运行时间：45241500
运行时间：45490600
运行时间：45625900
```

#### sleep_for + timeBeginPeriod
依旧使用 sleep_for,额外使用 timeBeginPeriod 与 timeEndPeriod 设置系统时钟最小周期为 1 毫秒。

``` cpp
/// <summary>
/// 等待函数
/// 基于 sleep_for 并且使用 timeBeginPeriod 提高系统时钟精度，依旧不够精准
/// </summary>
/// <param name="time">毫秒</param>
static void wait_sleep_for(int64_t time)
{
	timeBeginPeriod(1);
	sleep_for(milliseconds(time));
	timeEndPeriod(1);
}

int main()
{
	int time = 40;
	for (;;)
	{
		steady_clock::time_point start = steady_clock::now();

		wait_sleep_for(time);

		steady_clock::time_point end = steady_clock::now();
		cout << "运行时间：" << duration_cast<nanoseconds>(end - start).count() << endl;
	}
}
```

精度相差 0.1 - 0.9 毫秒，如果读取文件不被察觉，如果是推流的话，会明显看到帧数达不到，25FPS 的设置只能达到 24.6FPS 左右。并且 timeBeginPeriod 为系统全局设置，就算及时关闭也不排除会对其他程序有影响。

``` cmd
运行时间：40689800
运行时间：40542900
运行时间：40048500
运行时间：40933100
运行时间：40406200
运行时间：40570000
运行时间：40478500
运行时间：40930100
运行时间：40372600
运行时间：40375400
```

#### QueryPerformanceCounter
基于高精度计时器循环判断实现阻塞等待。

``` cpp
/// <summary>
/// 等待函数
/// 基于 QueryPerformanceCounter 高精度计时器多媒体时钟轮询，较为精准，但是会大量占用 CPU 处理轮询
/// </summary>
/// <param name="time">毫秒</param>
static void wait_sleep_perform(int64_t time)
{
	LARGE_INTEGER perfCnt, start, now;

	QueryPerformanceFrequency(&perfCnt);
	QueryPerformanceCounter(&start);

	do {
		QueryPerformanceCounter((LARGE_INTEGER*)&now);
	} while ((now.QuadPart - start.QuadPart) / float(perfCnt.QuadPart) * 1000 * 1000 < time * 1000);
}

int main()
{
	int time = 40;
	for (;;)
	{
		steady_clock::time_point start = steady_clock::now();

		wait_sleep_perform(time);

		steady_clock::time_point end = steady_clock::now();
		cout << "运行时间：" << duration_cast<nanoseconds>(end - start).count() << endl;
	}
}
```

精度相差 0.001 - 0.003 毫秒，几乎很完美的实现阻塞等待，但是存在一个问题，就是参考文章中提到的占用 CPU 问题，可能会使用一个 CPU 核心来处理，我这里测试确实有看到 CPU 的占用提升，但是还在可接受的范围内，因为视频大部分都是用 GPU 处理。

``` cmd
运行时间：40001900
运行时间：40002400
运行时间：40003300
运行时间：40003000
运行时间：40002500
运行时间：40002100
运行时间：40001800
运行时间：40002200
运行时间：40002000
运行时间：40003800
```