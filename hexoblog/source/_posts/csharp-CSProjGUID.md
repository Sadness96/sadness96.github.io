---
title: 类库项目中新增 WPF 窗口
date: 2018-05-27 13:10:20
tags: [c#,wpf]
categories: C#.Net
---
### 新创建类库只能添加 WPF 用户控件，无法添加 WPF 窗体
<!-- more -->
通过修改 .csproj 配置文件使类库可以创建 WPF 窗体
### 修改方法
新增节点 ProjectTypeGuids 至 .csproj 配置文件 Project.PropertyGroup 下
``` xml
<ProjectTypeGuids>{60dc8134-eba5-43b8-bcc9-bb4bc16c2548};{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}</ProjectTypeGuids>
```
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-CSProjGUID/csproj.png"/>