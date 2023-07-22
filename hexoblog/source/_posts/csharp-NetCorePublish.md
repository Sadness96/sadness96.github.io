---
title: NetCore 应用发布
date: 2023-07-26 23:55:00
tags: [c#,netcore]
categories: C#.Net
---
### 使用发布时生成单个文件找不到目录下日志与配置文件
<!-- more -->
#### 简介
现在使用 C# 开发软件，微软提供了两种框架，一种是用于构建在 Windows 上运行的应用程序 [.NET Framework](https://dotnet.microsoft.com/en-us/download/dotnet-framework) 框架 与 自由跨平台开源的 [.NET Core](https://dotnet.microsoft.com/en-us/download) 框架，[.NET Core](https://dotnet.microsoft.com/en-us/download) 框架在 3.1 版本后直接命名为 .NET 5.0 、 .NET 6.0 、 .NET 7.0，此文以 .NET Core 3.1 为例。

#### 编译与发布
编译是将源代码转换为可执行文件或库的过程。在开发过程中，可以使用 Debug 配置进行编译，以便在调试时获得更多的信息和功能。这样编译出的可执行文件通常会比较大，并包含了调试符号和其他调试相关的信息，以方便调试和排查问题。

发布是将编译后的代码进行优化和精简，以减小文件大小并提高执行效率。发布版本通常使用 Release 配置进行编译，这会启用各种优化选项，如代码压缩、去除调试符号、启用编译器优化等。发布版本的可执行文件通常更小，执行速度更快，适合部署到生产环境中使用。

#### 异常问题
##### 使用发布时生成单个文件找不到目录下日志与配置文件
<img src="https://sadness96.github.io/images/blog/csharp-NetCorePublish/发布配置.jpg"/>

在发布时选择部署模式为独立时，可以不依赖于系统环境独立运行，否则需要安装对应的 .NET Core 运行时环境，选择独立后提供一个生成单个文件的选项，勾选生成单个文件会使生成的项目仅剩一个 .exe 可执行程序与 .pdb 调试符号文件，显得很干爽，但这只是一个错觉。

<img src="https://sadness96.github.io/images/blog/csharp-NetCorePublish/dnSpy反编译.jpg"/>

使用 [dnSpy](https://github.com/dnSpy/dnSpy) 反编译得到的结果反而无法解析。

<img src="https://sadness96.github.io/images/blog/csharp-NetCorePublish/ILSpy反编译.jpg"/>

使用 [ILSpy](https://github.com/icsharpcode/ILSpy) 反编译得到的结果可以看到，所有的 NuGet 引用库以及所有 .NET Core 依赖库都在一个 .exe 可执行程序中，但是实际运行时却发现，[NLog](https://github.com/NLog/NLog) 生成的日志文件与软件编写生成在根目录的配置文件都不见了，但是却实实在在的能够读取配置，根据排查，测试运行发布的程序时打印 AppDomain.CurrentDomain.BaseDirectory，正常会打印应用程序所在目录，但是却打印了以下由系统生成的临时目录，所以运行程序后，程序会自动把引用库以及文件解压到以下文件夹中，而应用程序则是映射到这个目录。

``` cmd
C:\Users\{系统登录用户}\AppData\Local\Temp\.net\{应用程序名称}\{系统根据目录随机生成的文件夹}
```

所有的日志以及配置文件也都存在这里，这会对查找日志以及配置造成干扰，所以建议取消勾选生成单个文件，虽然安装目录文件夹中会以一种稀碎的方式存放那么多文件。
