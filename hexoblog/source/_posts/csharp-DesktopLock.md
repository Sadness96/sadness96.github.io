---
title: 电脑挂机锁
date: 2016-05-31 23:50:56
tags: [c#,wpf,ini]
categories: C#.Net
---
<img style="width:400px;height:300px" src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DesktopLock/main.png"/>
### 初学C#开发，Windows平台电脑挂机锁
<!-- more -->
项目介绍：大学临毕业前想要做出点成品，漂洋过海跑到北京作为面试的资本。没有什么好点子，是一个同学的主意，本着学习的目的，还是比较实用的功能（没人会用的…）就做这个了。
曾经一度认为只有桌面端软件才叫软件，奈何学校教学只有基础的语法还有偏向于WEB开发，有试过用java的awt和swing设计图形化界面，但是效果并不是很理想，后来有了解到微软平台MFC、QT、Winform,但是界面的画风显得很古老，使用异形窗体后会有很严重的锯齿，最后选用WPF作为主界面。
项目开源地址：[https://github.com/Sadness96/DesktopLock](https://github.com/Sadness96/DesktopLock)
贴吧发布地址：[http://tieba.baidu.com/p/4584097900](http://tieba.baidu.com/p/4584097900)
项目虽然简单，BUG还有很多，甚至不如Windows自带的Win+L锁屏好使，但是经过几天的边学边做，真的给我明确了未来的方向。

#### 设置菜单界面
可通过三种方式设置挂机锁，1.密码解锁；2.时间解锁（根据系统当前时间拼接作为密码）；3.U盘解锁（通过写入加密数据到U盘，插入U盘时即可解锁）；
<img style="width:540px;height:330px" src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DesktopLock/config.png"/>
#### 修改锁屏壁纸
默认三种锁屏图片可选或自定义图片。
<img style="width:540px;height:330px" src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DesktopLock/style.png"/>
#### 定时锁屏关机界面
可定时多长时间后锁屏或关机。
<img style="width:540px;height:330px" src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DesktopLock/timing.png"/>
#### USB写入密码
插入U盘后写入加密秘钥，可通过设置U盘解锁方式，在插入U盘后系统自动解锁。
<img style="width:540px;height:330px" src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DesktopLock/usb.png"/>
#### 关于
<img style="width:540px;height:330px" src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DesktopLock/about.png"/>
#### 锁屏
可调节透明度，挂机时显示桌面运行的程序。
<img style="width:800px;height:450px" src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-DesktopLock/lock.png"/>