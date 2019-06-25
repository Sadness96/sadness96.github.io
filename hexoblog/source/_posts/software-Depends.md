---
title: Depends 使用介绍
date: 2018-08-01 12:13:07
tags: [software,depends]
categories: Software
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Depends/depends.jpg"/>
<!-- more -->
#### 简介
[Depends](http://www.dependencywalker.com/)：可以扫描任何32位或64位Windows模块（exe，dll，ocx，sys等），并构建所有相关模块的分层树形图。对于找到的每个模块，它列出了该模块导出的所有函数，以及其他模块实际调用了哪些函数。另一个视图显示所需文件的最小集合，以及每个文件的详细信息，包括文件的完整路径，基本地址，版本号，机器类型，调试信息等。对于解决与加载和执行模块相关的系统错误也非常有用。Dependency Walker检测到许多常见的应用程序问题，例如缺少模块，模块无效，导入/导出不匹配，循环依赖性错误，模块的机器类型不匹配以及模块初始化失败。
工作中需要使用到 [C#/C++ 混合编程](/blog/2018/08/01/cpp-HybridCSharp/)，或者安装一个软件后提示丢失某些类库导致无法运行时，Depends是最好的选择。
#### 使用
##### 引用类库缺少
软件安装包或开发环境提示错误：无法加载 DLL“xxx.dll”: 找不到指定的模块。使用Depends检查缺少模块，模块无效的库。
下图：[OpenCV](https://opencv.org/) 使用 [Visual Studio](https://visualstudio.microsoft.com/zh-hans/downloads/) 2015 开发缺少mfc120d.dll、msvcr120d.dll、msvcp120d.dll库
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Depends/LackDLL.png"/>
##### C/C++开发
方法声明为C++方法时，[DllImport](https://docs.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.dllimportattribute?redirectedfrom=MSDN&view=netframework-4.8) 引用被不明方法加密，调用时需拷贝对应方法的Function名字粘贴到EntryPoint。
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Depends/C-CPP.jpg"/>
``` CSharp
[DllImport(@"CPP_Demo.dll", EntryPoint = "?filePath@@YAPADPAD@Z", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode)]
public static extern IntPtr filePath(IntPtr filePath);
```