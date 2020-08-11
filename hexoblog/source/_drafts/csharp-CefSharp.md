---
title: 使用 CEF 在 PC 客户端中加载网页
date: 2020-8-11 13:03:21
tags: [c#,wpf,cef]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-CefSharp/CEFLogo.png"/>

### 在 WPF 中调用 CEF 加载网页
<!-- more -->
#### 简介
最近有需求在 wpf 中加载一个网页，尝试使用 CEF 加载 Chrome 内核浏览器显示。
#### 引用库介绍
NuGet 中引用 [CefSharp.Wpf](https://github.com/cefsharp/cefsharp)
#### 设置
需要在项目 .csproj 文件中增加代码
``` XML
<ItemGroup>
    <Reference Update="CefSharp">
        <Private>true</Private>
    </Reference>
    <Reference Update="CefSharp.Core">
        <Private>true</Private>
    </Reference>
    <Reference Update="CefSharp.Wpf">
        <Private>true</Private>
    </Reference>
</ItemGroup>
```
#### 使用
.xaml 文件中 增加引用和样式
``` XML
xmlns:chrome="clr-namespace:CefSharp.Wpf;assembly=CefSharp.Wpf"
```
``` XML
<Grid>
    <chrome:ChromiumWebBrowser/>
</Grid>
```
.xaml.cs 文件中增加代码
``` CSharp
var webView = new ChromiumWebBrowser();
this.Content = webView;
webView.Address = @"https://www.baidu.com/";
```
即可显示
<img src="https://sadness96.github.io/images/blog/csharp-CefSharp/CEFDemo.png"/>