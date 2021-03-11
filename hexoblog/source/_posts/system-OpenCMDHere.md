---
title: Win10 在此处打开命令窗口
date: 2017-01-26 10:23:19
tags: [windows,cmd,registry]
categories: System
---
### Win10 鼠标右键在此处打开CMD窗口
<!-- more -->
#### 简介
[Windows 10](https://baike.baidu.com/item/Windows%2010?fromtitle=win10&fromid=10936225) 一直想使用 [Power Shell](https://baike.baidu.com/item/Windows%20Power%20Shell/693789?fromtitle=PowerShell&fromid=1061700&fr=aladdin) 代替原有的 [CMD](https://baike.baidu.com/item/%E5%91%BD%E4%BB%A4%E6%8F%90%E7%A4%BA%E7%AC%A6/998728?fromtitle=CMD&fromid=1193011&fr=aladdin)，功能更加强大的同时也伴随很多弊病，例如：内存占用（测试CMD占用10.9MB内存，PowerShell占用37.2MB内存）、启动和执行速度较慢（高配电脑打扰了）、蓝色的底也看起来并不怎么舒服（虽然都能改）、多按两下Backspace还滴滴滴滴滴。
#### 解决方案
主要修改地址在Win10的文件夹处Shift+鼠标右键原有的"在此处打开命令窗口(M)"被修改为"在此处打开 Powershell 窗口(S)"，通过修改注册表的方式在鼠标右键添加CMD的打开方式（当前目录）。
复制以下内容到文本中存为 [reg文件](https://baike.baidu.com/item/reg%E6%96%87%E4%BB%B6/549755)，双击运行添加到注册表中即可。
``` REG
Windows Registry Editor Version 5.00

[HKEY_CLASSES_ROOT\Directory\Background\shell\OpenCMDHere]
"ShowBasedOnVelocityId"=dword:00639bc8

[HKEY_CLASSES_ROOT\Directory\Background\shell\OpenCMDHere\command]
@="cmd.exe /s /k pushd \"%V\""
```
#### 更简单的方案
2020年3月1日更新：在学习 .Net Core EF 时查阅 MSDN 才发现了一个更简单的方式。
[教程：使用迁移功能 - ASP.NET MVC 和 EF Core](https://docs.microsoft.com/zh-cn/aspnet/core/data/ef-mvc/migrations?view=aspnetcore-2.0)
直接在文件夹中地址栏输入：“cmd” 或 “powershell” 即可。
<img src="https://docs.microsoft.com/zh-cn/aspnet/core/data/ef-mvc/migrations/_static/open-command-window.png?view=aspnetcore-2.0"/>