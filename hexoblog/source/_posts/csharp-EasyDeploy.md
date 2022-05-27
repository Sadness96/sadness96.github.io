---
title: Easy Deploy
date: 2022-05-05 16:08:00
tags: [c#,wpf]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-EasyDeploy/MainWindow.jpg"/>

<!-- more -->
#### 简介
控制台程序轻松部署：[EasyDeploy](https://github.com/iceelves/EasyDeploy)
做这个东西是解决项目中实际遇到的一个问题，那就是内部在实际项目应用中，有很多的应用由于不需要 UI 界面，所以简便的使用控制台程序开发，而控制台程序在实际部署的时候，又存在很多问题，例如图标使用开发语言默认样式，或使用控制台默认图标，导致程序异常崩溃后不易排查缺少了哪些，或是需要配置开机自启，虽然公司已有解决方案，但还是按照自己理解做了一个版本。
放弃了以服务方式启动而是以控制台程序启动，这样方便捕获输出的信息，监控进程 PID 以更方便控制。

#### 初衷
做这个程序源于跟朋友的聊天
<img src="https://sadness96.github.io/images/blog/csharp-EasyDeploy/ChatRecord.jpg"/>

#### 功能
##### 主要功能
首先最主要的功能是部署控制台程序，可选相对路径或绝对路径，相对路径更适合附带整个程序一起打包，可以像 [XAMPP](https://www.apachefriends.org/index.html) 一样配置一些固定服务，完整的控制程序的生命周期，确保崩溃后重启，也或者连带整个主程序开机自启，右侧方便加载控制台程序的打印内容，兼容 Ansi 文字显示。

* 如果控制台程序以输入结尾，可能会存在崩溃。
* 系统自带的 UI 程序，例如 Calc 可能会存在异常。

##### 其余功能
兼容语言：（简体中文，English）。
可配置控制台默认背景颜色文字颜色。