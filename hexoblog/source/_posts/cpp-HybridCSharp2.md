---
title: C#/C++ 混合编程无法分配内存空间与回调
date: 2023-02-28 11:03:55
tags: [c++,c#,depends]
categories: C++
---
### 调用非动态链接库无法分配内存空间，回调 C# 方法返回数据
<!-- more -->
### 简介
使用 [C#/C++ 混合编程](https://sadness96.github.io/blog/2017/05/27/csharp-NPOIHelper/) 方式开发，新遇到的问题。

### 处理
#### 报错：引发的异常: 0xC0000005: 执行位置 0x0000000000027DD4 时发生访问冲突。
c++ 以：应用程序(.exe) 的方式同样可以导出函数，最选择这样使用的原因是想核心函数即可以被其他程序以非动态链接库的方式调用，也可以单独使用控制台程序调用，导出的简单方法调用没有太大区别，但是在申请内存空间时会报错，代码不变，仅修改项目类型为：动态库(.dll)，报错消失，经测试，目前没有办法解决，能搜到的文章都已没人会使用应用程序导出函数结束。
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