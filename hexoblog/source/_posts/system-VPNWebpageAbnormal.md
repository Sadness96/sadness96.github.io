---
title: VPN 导致的网页显示异常
date: 2020-01-28 11:42:00
tags: [windows,ie,vpn]
categories: System
---
### 未启动 VPN 程序时网络正常，网页显示异常
<!-- more -->
### 简介
电脑装有梯子软件，但是有时存在未启动程序时网络正常，但是网页显示异常的情况，由于安装过 VPN 后软件设置了默认代理导致。

#### 解决方法
1. 打开 Internet 选项，可通过 IE 设置中打开或通过运行输入打开
    ``` cmd
    inetcpl.cpl
    ```

1. 选择菜单：
    连接 → 局域网(LAN)设置 → 局域网设置(L)
1. 取消勾选：
    代理服务器 → 为 LAN 使用代理服务器(这些设置不用于拨号或 VPN 连接)(X)
    <img src="https://sadness96.github.io/images/blog/system-VPNWebpageAbnormal/SetUpProxy.png"/>

1. 确认应用保存