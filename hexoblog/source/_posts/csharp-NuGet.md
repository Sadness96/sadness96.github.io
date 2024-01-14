---
title: NuGet 包管理
date: 2024-01-14 12:32:00
tags: [c#,nuget]
categories: C#.Net
---
### 创建 NuGet 包以及搭建离线服务
<!-- more -->
### 简介
[NuGet](https://www.nuget.org/) 是 .NET 的包管理器，由于微软服务器做的十分到位，不像其他语言的包管理器 maven、pip、npm 等还要搬梯子或是映射国内镜像源，所以我一直不理解在公司内搭建 NuGet 服务的需求，但是最近有一些库需要给多个项目使用，确实使用包管理的方式要比每个项目集成代码或者拷贝 DLL 好得多，但是有些库适合提交到微软的包管理仓库，有些不适合。

### NuGet 服务器
#### 简介
[BaGet](https://loic-sharma.github.io/BaGet/) 是一个轻量级 NuGet 服务器，[开源 GitHub](https://github.com/loic-sharma/BaGet)
<img src="https://user-images.githubusercontent.com/737941/50140219-d8409700-0258-11e9-94c9-dad24d2b48bb.png"/>

#### 部署
项目为跨平台应用，根据自己的习惯部署即可。
我直接使用代码编译运行，部署后访问地址为：http://localhost:50557

#### 上传
创建好 NuGet 包后使用命令上传 .nupkg 即可，IP 端口根据实际部署地址为准
``` cmd
nuget push -Source http://localhost:50557/v3/index.json package.nupkg
```

#### 项目安装
1. 项目中右键点击管理 NuGet 程序包
1. 在右上角程序包源中配置新增一个包源 http://localhost:50557/v3/index.json
1. 正常使用安装 NuGet 程序包

### 创建 NuGet 程序包
#### 简介
[NuGetPackageExplorer](https://github.com/NuGetPackageExplorer/NuGetPackageExplorer) 是一个使用 GUI 创建、更新和部署 NuGet 包的工具，推荐使用 [Microsoft Store](https://www.microsoft.com/store/apps/9wzdncrdmdm3) 安装。

<img src="https://raw.githubusercontent.com/NuGetPackageExplorer/NuGetPackageExplorer/main/images/screenshots/PackageView.png"/>

#### 创建 NuGet 项目
创建较为简单，不多介绍了，配置好名称版本库信息，库添加到 lib 下对应的版本中。

#### 遇到的问题
##### NuGet 中包含的文件无法拷贝到项目生成目录
参考 [Create a package using the nuget.exe CLI](https://learn.microsoft.com/en-us/nuget/create-packages/creating-a-package) 所有内容复制到项目根目录的文件，放到 content 下，但是实际测试并没有拷贝到项目根目录下，参考其他博客文章，几种方式都没有起效果。
1. 使用 NuGetPackageExplorer 编辑 Files 标签无法保存，不清楚是什么原因导致的。
1. 二进制文件放到 contentFiles\ 目录下，然后使用 contentFiles 标签配置，但是安装 NuGet 包时报错只能拷贝文本文件。
1. 使用 init.sp1 脚本执行拷贝，测试安装后脚本并没有执行。

所以目前我只能手动在安装 NuGet 包后手动添加配置或把文件手动把文件拷贝到项目根目录，编辑项目配置文件。
例如： NuGet 包名为 Test，版本号为 1.0.0，要拷贝的文件包含在 content\lib 下，拷贝到项目生成目录 \lib。
``` xml
<ItemGroup>
    <PackageReference Include="Test" Version="1.0.0" />
</ItemGroup>
<!--需要添加的配置-->
<ItemGroup>
    <Content Include="$(NuGetPackageRoot)\Test\1.0.0\content\lib\**">
        <LinkBase>lib</LinkBase>
        <CopyToOutputDirectory>Always</CopyToOutputDirectory>
        <LinkBaseTargetFramework></LinkBaseTargetFramework>
        <LinkBaseSpecificVersion>false</LinkBaseSpecificVersion>
    </Content>
</ItemGroup>
```