---
title: Jenkins 使用介绍
date: 2019-12-26 11:05:38
tags: [software,jenkins]
categories: Software
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-jenkins/jenkins.png"/>

<!-- more -->
### 简介
[Jenkins](https://jenkins.io/zh/) 是一个持续集成（CI&CD）工具，用以构建、部署、自动化。
### 运行流程
以部署PC客户端软件为例：
1.连接 GitLab 仓库 pull 最新代码
2.使用 NuGet 还原引用库
3.使用 MSBuild 编译项目工程
4.使用 NSIS 打包软件为安装包
5.以邮件方式将打包文件发送
### 软件部署
软件安装参考 [官方文档](https://jenkins.io/zh/doc/pipeline/tour/getting-started/)
#### 遇到的问题
##### 插件安装失败
登录重启页重启后重试
http://localhost:8081/restart
##### 需安装 Jenkins 插件
Git、MSBuild、NuGet
##### 配置系统环境变量
Path 下增加 MSBuild 路径：
D:\Software\Microsoft Visual Studio\2019\Preview\MSBuild\Current\Bin\amd64
##### NuGet 控制台程序下载
https://www.nuget.org/downloads
下载后拷贝至 Path 环境变量中
#### 构建
##### Pull Git 代码
填写 Git 地址以及分支名称即可
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-jenkins/%E6%BA%90%E7%A0%81%E7%AE%A1%E7%90%86.png"/>

##### 还原 NuGet 包
构建中选择：执行 Windows 批处理程序
``` cmd
nuget restore project.sln
```
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-jenkins/nuget.png"/>

##### 编译代码
构建中选择：Build a Visual Studio project or solution using MSBuild
| function | value |
| ---- | ---- |
| MSBuild Version | Default |
| MSBuild Build File | project.sln |
| Command Line Arguments | /t:Build /p:Configuration=Release;VisualStudioVersion=16.3 |
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-jenkins/msbuild.png"/>

##### 拷贝或删除多余文件
``` cmd
:: 拷贝文件
md .\bin\Release\MediaAccessSDK\
xcopy /s/c/h/y .\9.Reference\MediaAccessSDK\Release .\bin\Release\MediaAccessSDK

:: 删除多余的pdb和xml
del /s bin\Release\*.pdb
del /s bin\Release\*.xml
```
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-jenkins/copydel.png"/>

##### 程序打包
构建中选择：执行 Windows 批处理程序
[NSIS](https://nsis.sourceforge.io/Main_Page) 使用参考：[使用介绍](http://sadness96.github.io/blog/2018/11/24/software-Nsis/)
``` cmd
makensis PanoramaClientSetup.nsi
```
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-jenkins/nsis.png"/>

#### 构建后操作
##### 归档成品
在归档成品中直接写入打包好的安装包名称，会在构建结束后在结果中显示并可以直接下载
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-jenkins/归档成品.png"/>

<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-jenkins/结果.png"/>