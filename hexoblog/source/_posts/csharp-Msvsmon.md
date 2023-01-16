---
title: Visual Studio 远程调试
date: 2019-02-16 15:37:28
tags: [c#]
categories: C#.Net
---
### 在没有开发环境的电脑远程调试程序
<!-- more -->
### 简介
在实际开发中，经常出现开发环境运行正常，生产环境报错的情况，但是由于异常捕获或是打印日志不能直接定位到问题，生产环境又不适宜安装庞大的开发环境，所以远程调试是不错的选择，以 [Visual Studio 2019](https://visualstudio.microsoft.com/) 为例。

### 调试方式
1. 确保两台电脑在同一网络中。
1. 拷贝远程调试工具 msvsmon 到生产环境，目录为：..\Microsoft Visual Studio\2019\Professional\Common7\IDE\Remote Debugger
1. 以管理员方式运行 msvsmon.exe
    <img src="https://sadness96.github.io/images/blog/csharp-Msvsmon/1.jpg"/>

1. 工具 -> 选项 -> 修改身份验证模式：如果环境安全的情况下可以设置为无身份验证，更方便连接。
    <img src="https://sadness96.github.io/images/blog/csharp-Msvsmon/2.jpg"/>
    <img src="https://sadness96.github.io/images/blog/csharp-Msvsmon/3.jpg"/>

1. 在开发电脑中打开 Visual Studio 选择开发的项目，点击菜单中：调试 -> 附加到进程，选择连接类型为远程，点击查找连接目标，搜索到开启局域网调试工具的电脑，附加到本机代码，在可用进程中搜索生产环境运行的主程序，点击附加。
    <img src="https://sadness96.github.io/images/blog/csharp-Msvsmon/4.jpg"/>
    <img src="https://sadness96.github.io/images/blog/csharp-Msvsmon/5.jpg"/>

1. 生产环境中的 msvsmon 程序显示 xxx 已连接，即可远程调试。
    <img src="https://sadness96.github.io/images/blog/csharp-Msvsmon/6.jpg"/>