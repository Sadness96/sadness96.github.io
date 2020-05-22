---
title: 命令启动程序
date: 2016-08-10 10:55:11
tags: [windows,cmd]
categories: Windows
---
### 通过运行或CMD命令启动软件
<!-- more -->
#### 简介
作为一个强迫症来说，电脑桌面当然是越精简越好，杂乱的文件不能有，甚至想干掉所有图标…Windows的强大足以完全用快捷键操作系统了，鼠标用得少了效率也提高不少。
之前有想过用 [Mac OS](https://baike.baidu.com/item/Mac%20OS/2840867?fr=aladdin) 的 Dock 的工具栏样式显示软件图标，但是没有MAC总觉得少点什么。
通过 [运行（Win+R）](https://baike.baidu.com/item/%E8%BF%90%E8%A1%8C/13132909?fr=aladdin) 的方式启动软件或许是个不错的方式，系统有集成很多自带的软件或者安装打包有环境变量的软件都可以输入名称运行（例如：calc、notepad、mspaint、mstsc）。
#### 设置
##### 创建启动目录，存放想要启动的快捷图标，修改快捷方式启动的名称
<img src="https://sadness96.github.io/images/blog/windows-QueryCmd/1.jpg"/>

##### 配置环境变量（在环境变量Path下增加启动目录）
###### Win7
<img src="https://sadness96.github.io/images/blog/windows-QueryCmd/2.jpg"/>

###### Win10
<img src="https://sadness96.github.io/images/blog/windows-QueryCmd/3.png"/>

##### 通过 Win+R 打开运行窗口，输入启动名称即可。
例：运行PS
<img src="https://sadness96.github.io/images/blog/windows-QueryCmd/4.png"/>

#### 2019年7月16日补充
记录一个更好的替代或是搭配使用的工具
Wox：[官方网站](http://www.wox.one/)（An effective launcher for windows）
GitHub：[https://github.com/Wox-launcher/Wox](https://github.com/Wox-launcher/Wox)
<img src="http://www.wox.one/images/wox_preview.jpg"/>