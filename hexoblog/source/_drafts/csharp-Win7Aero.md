---
title: Windows 7 下无法开启 Aero 主题
date: 2021-06-02 10:39:45
tags: [c#,windows 7,aero]
categories: C#.Net
---
### 在 Windwos 7 中开启 Aero 主题
<!-- more -->
#### 简介
[Aero 主题](https://baike.baidu.com/item/Windows%20Aero/6845089?fromtitle=Aero&fromid=3554670&fr=aladdin) 仅仅是一个受人追捧的毛玻璃效果而已，但是在项目实际使用的时候却发现与 [DirectX](https://www.microsoft.com/zh-cn/download/details.aspx?id=35) 渲染效率相关，在未开启 Aero 主题的情况下 [Device.Present() 方法](https://docs.microsoft.com/en-us/previous-versions/bb324100(v=vs.85)) 延迟约在 00:00:00.1258071 相比开启了 Aero 主题的延迟约在 00:00:00.0000365，千倍的时间差。

#### 测试结果
换个主题并不是难事，但是面对一个早已不受微软支持的操作系统，对多屏幕的支持不是很好，目前也没什么解决办法，实际测试中 3 块 1080p 显示器及以上无法被动开启 Aero 主题，2 块 4k 显示器及以上无法被动开启 Aero 主题，试过多型号显卡（Quadro P1000、Quadro P2000、Quadro P4000、Gtx 1080、Rtx 2070、Rtx 4000），排除显卡性能问题，试过多版本驱动，但是不排除显卡驱动与 Windows 7 兼容不好。

#### 系统设置上的比对
##### 桌面右键菜单个性化
<img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Win7个性化设置-正常.png"/>

<img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Win7个性化设置-异常.png"/>