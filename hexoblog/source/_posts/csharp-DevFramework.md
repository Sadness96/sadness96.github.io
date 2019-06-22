---
title: c/s端开发基础框架
date: 2016-12-20 10:30:21
tags: [c#,wpf,helper,devexpress]
categories: C#.Net
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DevFramework/Sadness_MainRibbon.png"/>
### 基于 Wpf + Prism + DevExpress 制作的插件式开发基础框架
<!-- more -->
工作已经临近半年了，日常在工作中有所积累，打算写一套自己的帮助类，后逐步发展为一个桌面端快速开发的框架。使用 Wpf + Prism + DevExpress 作为基础，Ribbon样式的插件式开发方式。同时又基于帮助类实现了几个简单的功能，后续再博客中会逐步记录帮助类。
#### 注册工具
获取计算机硬件信息（网卡MAC地址、CPU-ID、硬盘序列号、内存序列号、主板序列号、BIOS序列号、显卡信息），拼接加密生成唯一序列号，可用于软件激活使用。
详细请查阅：[电脑硬件信息帮助类](/blog/2017/06/06/csharp-PCInformationHelper/)
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DevFramework/%E6%B3%A8%E5%86%8C%E5%B7%A5%E5%85%B7.png"/>
#### 生成二维码
基于 ZXing.QrCode 库制作的横条码、二维码生成工具，可添加LOGO，也可动态识别横条码、二维码。
详细请查阅：[二维码帮助类](/blog/2017/06/06/csharp-QRCodeHelper/)
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DevFramework/%E7%94%9F%E6%88%90%E4%BA%8C%E7%BB%B4%E7%A0%81.png"/>
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DevFramework/%E8%AF%86%E5%88%AB%E4%BA%8C%E7%BB%B4%E7%A0%81.png"/>
#### 加密解密工具
提供几种对称密钥加密与非对称加密算法，以及哈希算法，也可以加密文件夹。
详细请查阅：[加密解密帮助类](/blog/2018/01/10/csharp-EncryptionHelper/)
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DevFramework/%E5%8A%A0%E5%AF%86%E8%A7%A3%E5%AF%86%E5%B7%A5%E5%85%B7.png"/>
#### 文件共享
通过调用Windows API接口实现文件共享可视化操作（需管理员权限）。
详细请查阅：[文件共享帮助类](/blog/2017/05/23/csharp-FileSharingHelper/)
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DevFramework/%E6%96%87%E4%BB%B6%E5%85%B1%E4%BA%AB.png"/>
#### 数据库转换工具
工作中大量需要多种不同数据格式互相转换，通过ADO.NET实现可视化数据转换工具，目前支持关系型数据库SqlServer、Oracle、MySql、Access、SQLite。
详细请查阅：[ADO.NET 帮助类](/blog/2016/12/21/csharp-ADOHelper/)
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DevFramework/%E6%95%B0%E6%8D%AE%E5%BA%93%E8%BD%AC%E6%8D%A2%E5%B7%A5%E5%85%B7.png"/>
