---
title: SonarQube 使用介绍
date: 2020-01-16 10:50:00
tags: [software,sonarqube]
categories: Software
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-SonarQube/sonarqube.png"/>

<!-- more -->
### 简介
[SonarQube](https://www.sonarqube.org/) 是一个用于管理代码质量和安全的开源平台。
### 软件部署
软件安装参考 [官方文档](https://docs.sonarqube.org/latest/setup/get-started-2-minutes/)
#### 下载社区版本
下载地址：[https://www.sonarqube.org/downloads/](https://www.sonarqube.org/downloads/)
#### 解压后运行批处理文件
按顺序运行：
``` cmd
.\sonarqube\bin\windows-x86-64\InstallNTService.bat
.\sonarqube\bin\windows-x86-64\StartNTService.bat
.\sonarqube\bin\windows-x86-64\StartSonar.bat
```
#### 访问网页
[http://localhost:9000/](http://localhost:9000/)
默认用户名密码：admin:admin
#### Download SonarScanner for MSBuild
[SonarScanner for MSBuild](https://sonarcloud.io/documentation/analysis/scan/sonarscanner-for-msbuild/)
解压后配置环境变量 Path
#### Create new project
创建新项目：
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-SonarQube/CreateNewProject.png"/>

生成 ToKen 令牌：
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-SonarQube/GenerateToken.png"/>

项目根目录运行：
``` cmd
SonarScanner.MSBuild.exe begin /k:"#ProjectName" /d:sonar.host.url="http://localhost:9000" /d:sonar.login="#ToKen"
MsBuild.exe /t:Rebuild
SonarScanner.MSBuild.exe end /d:sonar.login="#ToKen"
```
查看质检结果：
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-SonarQube/QualityGate.png"/>

#### Jenkins 中集成 SonarQube
参考：[集成 SonarQube](http://sadness96.github.io/blog/2019/12/26/software-Jenkins/#SonarQube)