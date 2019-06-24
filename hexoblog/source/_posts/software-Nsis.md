---
title: Nsis 使用介绍
date: 2018-11-24 15:59:51
tags: [software,nsis]
categories: Software
---
### 基于 NSIS 的 Windows 桌面端打包程序
<!-- more -->
#### 简介
现工作中作为全栈开发工程师，不光写服务端B/S也要写桌面端C/S程序，在部署B/S的时候一般是拷贝文件或自动部署到服务器，但是桌面端程序普遍是打包为[安装包(Install pack)](https://baike.baidu.com/item/%E5%AE%89%E8%A3%85%E5%8C%85/7693150?fr=aladdin)在官网提供下载或是直接发送给用户安装升级。一般由质检部门打包测试，最后没有BUG的版本发布。
[NSIS（Nullsoft Scriptable Install System）](https://nsis.sourceforge.io/Main_Page)是一个专业的开源系统，用于创建Windows安装程序。它的设计尽可能小巧灵活，因此非常适合互联网分发。
