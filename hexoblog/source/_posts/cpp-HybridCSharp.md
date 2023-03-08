---
title: C#/C++ 混合编程
date: 2018-08-01 10:06:18
tags: [c++,c#,depends]
categories: C++
---
### 使用C++开发算法（全景拼接、人脸识别、超分辨率重建）/使用C#开发服务端业务逻辑和UI
<!-- more -->
### 简介
现工作中作由于 C++ 的 UI（[MFC](https://baike.baidu.com/item/MFC/2530850?fr=aladdin)、[QT](https://baike.baidu.com/item/qt)）开发界面比较难看，定制用户控件复杂且样式一般。而 C# 又不擅长于开发算法逻辑，效率不如 C++。所以现在大部分公司都会选用 C#/C++ 混合编程。

### 性能分析
使用 C# 做界面要比 C++ 高效的多，但是存在算法逻辑的时候由于性能问题不得不把部分模块交给 C++ 处理，C++ 可以使用高效的栈内存对象（CCalc），而 C# 所有对象只能放在托管堆中。测试 C# 调用 C++ 类库使用[托管](https://baike.baidu.com/item/%E6%89%98%E7%AE%A1/3967693)方式性能得到了一定程度的提升，但比起单纯的 C++ 项目，还是差了很多；测试 C# 调用 C++ 类库使用 [DllImport](https://docs.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.dllimportattribute?redirectedfrom=MSDN&view=netframework-4.8) [Attribute](https://docs.microsoft.com/en-us/dotnet/csharp/programming-guide/concepts/attributes/) 混合方式由[非托管动态链接库](https://baike.baidu.com/item/%E9%9D%9E%E6%89%98%E7%AE%A1/7967564)效率与单独运行 C++ 相差无几。

### 简单基础参数传递
例子：
1.最基础的加法运算；
2.传入图片地址，通过 OpenCV 处理后返回图片地址；
3.传入图片地址，通过 OpenCV 把彩色图像转换为灰度图像，然后返回给 C#；
方法声明为 C++ 方法时，[DllImport](https://docs.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.dllimportattribute?redirectedfrom=MSDN&view=netframework-4.8) 引用被不明方法加密，调用时需使用Depends工具拷贝对应方法的Function名字粘贴到EntryPoint。详情查看[Depends 使用介绍](/blog/2018/08/01/software-Depends/)

### 代码
#### .h
``` cpp
#pragma once

#ifdef TEST_EXPORTS
#define Test_API __declspec(dllexport)
#else
#define Test_API __declspec(dllimport)
#endif // TEST_RXPORTS

#ifdef __cplusplus
# define CEXTERN extern "C"
#else
# define CEXTERN
#endif

#define EXPORT_DLL CEXTERN Test_API

EXPORT_DLL int Add(int a, int b);

EXPORT_DLL char* FilePath(char* filePath);
```

#### .cpp
``` cpp
int Add(int a, int b)
{
	return a + b;
}

char* FilePath(char* filePath)
{
	char* resurlt;
	resurlt = filePath;
	return resurlt;
}
```

#### .cs
``` CSharp
private const string dllName = @"Dll.dll";

[DllImport(dllName, EntryPoint = "Add", CallingConvention = CallingConvention.Cdecl)]
public static extern int Add(int a, int b);

[DllImport(dllName, EntryPoint = "FilePath", CallingConvention = CallingConvention.Cdecl)]
public static extern IntPtr FilePath(IntPtr filePath);

static void Main(string[] args)
{
	int iAdd = Add(8, 12);
	string strFile = Marshal.PtrToStringAnsi(FilePath(Marshal.StringToHGlobalAnsi(@"1.JPG")));
}
```

### 注意事项
#### 引用方法乱码
使用 C++ 声明方法会导致引用是乱码，引用名称可以通过 [Depends](/blog/2018/08/01/software-Depends/) 查询。改为 C 声明后可正常引用方法，建议所有方法使用 C 声明，例如使用 C++ 需要引用方法："?Add@@YAHHH@Z"，而使用 C 声明只需引用："Add"。

#### 报错：无法加载 DLL“xxx.dll”: 找不到指定的模块
推荐使用 [Depends](/blog/2018/08/01/software-Depends/) 工具检测缺少引用类库

#### 报错：调用 Dll "试图加载格式不正确的程序。(异常来自 HRESULT:0x8007000B)
调用64位类库需要把项目改为64位
桌面端修改：属性 → 生成 → 目标平台设为："Any Cpu"，取消勾选"首选32位"。
Web 端修改：工具 → 选项 → 项目和解决方案 → Web 项目 → 勾选："对网站和项目使用 IIS Express 的 64 位版"

#### 报错：引发的异常: 0xC0000005: 执行位置 0x0000000000027DD4 时发生访问冲突。
c++ 以：应用程序(.exe) 的方式同样可以导出函数，最选择这样使用的原因是想核心函数即可以被其他程序以非动态链接库的方式调用，也可以单独使用控制台程序调用，导出的普通方法调用没有太大区别，但是在申请内存空间时会报错，代码不变改为：动态库(.dll)，报错消失，经测试，目前没有办法解决，能搜到的文章都已没人会使用应用程序导出函数结束。
报错的分配内存方法例如：
``` cpp
char* chars = new char[1024];
char* chars = (char*)malloc(sizeof(char) * 1024);
```

#### 回调函数
c# 以非动态链接库调用 c++ 后，c++ 以回调函数的方式返回给 c#。

##### .cs
``` csharp
[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public delegate void LogCALLBACK(string log);

[DllImport(dllName, EntryPoint = "CallbackTest")]
public static extern void CallbackTest(LogCALLBACK log);

static void Main(string[] args)
{
	logCALLBACK = Action_logCALLBACK;
	CallbackTest(logCALLBACK);
}

private static void Action_logCALLBACK(string log)
{
	Console.WriteLine(log);
}
```

##### .h
``` cpp
typedef void(*LogCALLBACK)(char* log);

EXPORT_DLL void CallbackTest(LogCALLBACK logCALLBACK);
```

##### .cpp
``` cpp
void CallbackTest(LogCALLBACK logCALLBACK)
{
	for (size_t i = 0; i < 1000; i++)
	{
		string strLog = "hello";
		char* log = new char[1024];
		strcpy(log, strLog.c_str());
		if (logCALLBACK != NULL)
		{
			logCALLBACK(log);
		}
		delete log;
	}
}
```