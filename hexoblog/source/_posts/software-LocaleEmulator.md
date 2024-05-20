---
title: Locale-Emulator 使用介绍
date: 2021-12-28 00:45:00
tags: [software, localeemulator]
categories: Software
---
<img src="https://camo.githubusercontent.com/065f1a3eb69a0e787f7a7fdaae6a2032230cbe575ceaef82f559813735761a51/68747470733a2f2f692e696d6775722e636f6d2f45344771796c792e706e67"/>

<!-- more -->
### 简介
近日旧友邀请一起玩新改版的[彩虹岛物语(台服)](https://la.mangot5.com/la/index)，运行之后就遇到了问题，由于系统语言时区与程序不符导致的乱码，虽然可以通过修改系统默认语言和时区达到显示正常，但是显然不是长久之计，而且很麻烦，最后找到开源项目：[Locale-Emulator](https://github.com/xupefei/Locale-Emulator) 解决。

[Locale-Emulator](https://github.com/xupefei/Locale-Emulator) 是一种类似于 MS AppLocale 和 NTLEA 的工具，提供了一种模拟功能，可以使应用程序将您的操作系统识别为使用不同于真实语言的语言。当您尝试玩特定国家/地区的游戏时，它非常有用。

### 使用方法
1. 运行 LEInstaller.exe 并按“安装/升级”按钮。
1. 配置程序：对需要模拟的程序或快捷方式右键菜单 → Locale Emulator → 修改此程序的配置。例：设置为台服修改繁体与时区。
    <img src="https://sadness96.github.io/images/blog/software-LocaleEmulator/LeguiConfig.png"/>

1. 运行程序：对需要模拟的程序或快捷方式右键菜单 → Locale Emulator → 以此程序配置运行。
    <img src="https://sadness96.github.io/images/blog/software-LocaleEmulator/LeguiRun.png"/>

### 效果
#### 修改前
<img src="https://sadness96.github.io/images/blog/software-LocaleEmulator/style1.jpg"/>

#### 修改后
<img src="https://sadness96.github.io/images/blog/software-LocaleEmulator/style2.jpg"/>