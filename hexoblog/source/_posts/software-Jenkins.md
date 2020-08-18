---
title: Jenkins 使用介绍
date: 2019-12-26 11:05:38
tags: [software,jenkins]
categories: Software
---
<img src="https://sadness96.github.io/images/blog/software-Jenkins/jenkins.png"/>

<!-- more -->
### 简介
[Jenkins](https://jenkins.io/zh/) 是一个持续集成（CI&CD）工具，用以构建、部署、自动化。
### 运行流程
以部署PC客户端软件为例：
1.连接 [GitLab](https://about.gitlab.com/) 仓库 pull 最新代码
2.使用 [NuGet](https://www.nuget.org/) 还原引用库
3.使用 [MSBuild](https://msdn.microsoft.com/zh-CN/library/dd393574.aspx) 编译项目工程
4.使用 [NSIS](https://nsis.sourceforge.io/Main_Page) 打包软件为安装包
5.以邮件方式将打包文件发送(未完成)
### 软件部署
软件安装参考 [官方文档](https://jenkins.io/zh/doc/pipeline/tour/getting-started/)
#### 遇到的问题
##### 插件安装失败
登录重启页重启后重试
http://localhost:8081/restart
##### 需安装 Jenkins 插件
Git、MSBuild、NuGet、PowerShell
##### 配置系统环境变量
Path 下增加 MSBuild 路径：
..\Microsoft Visual Studio\2019\Preview\MSBuild\Current\Bin\amd64
##### NuGet 控制台程序下载
https://www.nuget.org/downloads
下载后拷贝至 Path 环境变量中
#### 构建
##### Pull Git 代码
填写 Git 地址以及分支名称即可
<img src="https://sadness96.github.io/images/blog/software-Jenkins/%E6%BA%90%E7%A0%81%E7%AE%A1%E7%90%86.png"/>

###### 设置 Git 用户名密码
如果本地 Git 记录用户无权限访问则会报错：
``` cmd
Failed to connect to repository : Command "git.exe ls-remote -h -- http://192.168.5.188:9090/***/***.git HEAD" returned status code 128:
stdout:
stderr: remote: HTTP Basic: Access denied
fatal: Authentication failed for 'http://192.168.5.188:9090/***/***.git/'
```
选择：凭据→系统→全局凭据
<img src="https://sadness96.github.io/images/blog/software-Jenkins/CredentialConfig1.png"/>

选择：添加凭据
<img src="https://sadness96.github.io/images/blog/software-Jenkins/CredentialConfig2.png"/>

配置：Git 用户名密码
<img src="https://sadness96.github.io/images/blog/software-Jenkins/CredentialConfig3.png"/>

配置：Credentials 中选择添加的凭据
<img src="https://sadness96.github.io/images/blog/software-Jenkins/CredentialConfig4.png"/>

##### 还原 NuGet 包
构建中选择：执行 Windows 批处理程序
``` cmd
:: 清空项目中多余文件
git checkout . && git clean -xdf
:: nuget 引用
nuget restore project.sln
```
<img src="https://sadness96.github.io/images/blog/software-Jenkins/nuget.png"/>

##### 编译代码(客户端)
构建中选择：Build a Visual Studio project or solution using MSBuild

| function | value |
| ---- | ---- |
| MSBuild Version | Default |
| MSBuild Build File | project.sln |
| Command Line Arguments | /t:Build /p:Configuration=Release;VisualStudioVersion=16.3 |

<img src="https://sadness96.github.io/images/blog/software-Jenkins/msbuild.png"/>

##### 编译代码(服务端)
调用发布文件 .\Properties\PublishProfiles\FolderProfile.pubxml
测试 MSBuild 命令中加入 VisualStudioVersion=16.3 会导致不会生成发布目录
``` XML
<?xml version="1.0" encoding="utf-8"?>
<!--
此文件由 Web 项目的发布/打包过程使用。可以通过编辑此 MSBuild 文件
自定义此过程的行为。为了解与此相关的更多内容，请访问 https://go.microsoft.com/fwlink/?LinkID=208121。 
-->
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <PropertyGroup>
    <WebPublishMethod>FileSystem</WebPublishMethod>
    <PublishProvider>FileSystem</PublishProvider>3
    <LastUsedBuildConfiguration>Release</LastUsedBuildConfiguration>
    <LastUsedPlatform>Any CPU</LastUsedPlatform>
    <SiteUrlToLaunchAfterPublish />
    <LaunchSiteAfterPublish>True</LaunchSiteAfterPublish>
    <ExcludeApp_Data>False</ExcludeApp_Data>
    <publishUrl>.\bin\Release\PublishOutput</publishUrl>
    <DeleteExistingFiles>True</DeleteExistingFiles>
  </PropertyGroup>
</Project>
```

| function | value |
| ---- | ---- |
| MSBuild Version | Default |
| MSBuild Build File | project.sln |
| Command Line Arguments | /t:Build /p:Configuration=Release /p:DeployOnBuild=True /p:PublishProfile=FolderProfile |
<img src="https://sadness96.github.io/images/blog/software-Jenkins/msbuildasp.png"/>

##### 拷贝或删除多余文件
``` cmd
:: 拷贝文件
xcopy /s/c/h/y .\9.Reference\MediaAccessSDK\Release .\bin\Release\MediaAccessSDK\Release\

:: 删除多余的pdb和xml
del /s bin\Release\*.pdb
del /s bin\Release\*.xml
```
<img src="https://sadness96.github.io/images/blog/software-Jenkins/copydel.png"/>

##### 程序打包
构建中选择：执行 Windows 批处理程序
[NSIS](https://nsis.sourceforge.io/Main_Page) 使用参考：[使用介绍](http://sadness96.github.io/blog/2018/11/24/software-Nsis/)
``` cmd
:: 调用 makensis 命令构建 NSI
makensis PanoramaClientSetup.nsi
```
<img src="https://sadness96.github.io/images/blog/software-Jenkins/nsis.png"/>

##### 压缩文件打包
构建中选择：PowerShell
``` cmd
# 调用 PowerShell 命令压缩文件

# 压缩文件
Compress-Archive -Path .\test -DestinationPath .\test.zip
# 解压缩文件
Expand-Archive -Path .\test.zip -DestinationPath .\test
```
<img src="https://sadness96.github.io/images/blog/software-Jenkins/zipfile.png"/>

#### 构建后操作
##### 归档成品
在归档成品中直接写入打包好的安装包名称，会在构建结束后在结果中显示并可以直接下载
<img src="https://sadness96.github.io/images/blog/software-Jenkins/归档成品.png"/>

<img src="https://sadness96.github.io/images/blog/software-Jenkins/结果.png"/>

<span id='SonarQube'/>

#### 集成 SonarQube
安装使用 SonarQube 参考博客：[SonarQube 使用介绍](http://sadness96.github.io/blog/2020/01/16/software-SonarQube/)
##### SonarQube 创建 Jenkins ToKen
<img src="https://sadness96.github.io/images/blog/software-Jenkins/SonarQubeJenkinsToKens.png"/>

##### 下载插件 SonarQube Scanner
<img src="https://sadness96.github.io/images/blog/software-Jenkins/SonarQubeScanner.png"/>

##### Jenkins 配置 SonarQube
添加凭据：
<img src="https://sadness96.github.io/images/blog/software-Jenkins/SonarQubeCredentials.png"/>

##### 配置插件 SonarQube Scanner
系统管理→系统配置→SonarQube servers
<img src="https://sadness96.github.io/images/blog/software-Jenkins/ConfigSonarQubeServers1.png"/>

##### 配置插件 SonarQube Scanner
系统管理→全局工具配置→SonarQube Scanner
<img src="https://sadness96.github.io/images/blog/software-Jenkins/ConfigSonarQubeServers2.png"/>

##### 配置插件 SonarScanner for MSBuild
系统管理→全局工具配置→SonarScanner for MSBuild
<img src="https://sadness96.github.io/images/blog/software-Jenkins/ConfigSonarScannerForMSBuild.png"/>

##### 配置插件 JDK
系统管理→全局工具配置→JDK
<img src="https://sadness96.github.io/images/blog/software-Jenkins/ConfigJDK.png"/>

##### 构建项目
增加配置：
构建中选择：SonarScanner for MSBuild - Begin Analyoio
配置项 Project key 与 Project name 与 创建 SonarQube 中的 Project key 与 Project name 对应
在 MSBuild 后选择： SonarScanner for MSBuild - End Analyoio
<img src="https://sadness96.github.io/images/blog/software-Jenkins/SonarScannerforMSBuild.png"/>