---
title: C#/C++ 混合编程
date: 2018-08-01 10:06:18
tags: [c++,c#,depends]
categories: C++
---
### 使用C++开发算法逻辑（全景拼接、人脸识别、超分辨率重建）/使用C#开发服务端业务逻辑和客户端界面设计（WPF）
<!-- more -->
#### 简介
现工作中作由于C++的UI（MFC、QT）开发界面比较难看，定制用户控件复杂且样式一般。而C#又不擅长于开发算法逻辑，效率不如C++。所以现在大部分公司都会选用C#/C++混合编程。
#### 性能分析
使用C#做界面要比C++高效的多，但是存在算法逻辑的时候由于性能问题不得不把部分模块交给C++处理，C++可以使用高效的栈内存对象（CCalc），而C#所有对象只能放在托管堆中。测试C#调用C++类库使用[托管](https://baike.baidu.com/item/%E6%89%98%E7%AE%A1/3967693)方式性能得到了一定程度的提升，但比起单纯的C++项目，还是差了很多；测试C#调用C++类库使用 [DllImport](https://docs.microsoft.com/en-us/dotnet/api/system.runtime.interopservices.dllimportattribute?redirectedfrom=MSDN&view=netframework-4.8) 混合方式由[非托管动态链接库](https://baike.baidu.com/item/%E9%9D%9E%E6%89%98%E7%AE%A1/7967564)效率与单独运行C++相差无几。
#### 简单基础参数传递

#### 错误及处理
##### 报错：无法加载 DLL“xxx.dll”: 找不到指定的模块
推荐使用 [Depends](http://www.dependencywalker.com/) 工具检测缺少引用类库
opencv使用vs15开发缺少mfc120d.dll/msvcr120d.dll/msvcp120d.dll
##### 报错：调用 Dll "试图加载格式不正确的程序。(异常来自 HRESULT:0x8007000B)
调用64位类库需要把项目改为64位
桌面端修改：属性→生成→目标平台设为"Any Cpu"，取消勾选"首选32位"。
Web端修改：工具→选项→项目和解决方案→Web项目→勾选"对网站和项目使用 IIS Express 的 64 位版"