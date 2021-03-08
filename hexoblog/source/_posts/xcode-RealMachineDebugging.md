---
title: Xcode IOS 真机调试
date: 2020-10-23 20:20:00
tags: [ios,xcode]
categories: IOS
---
<img src="https://sadness96.github.io/images/blog/xcode-RealMachineDebugging/xcode_logo.jpg"/>

<!-- more -->
### 简介
IOS 开发使用真机有线调试和无线 Wi-Fi 调试

### 调试环境
电脑：MacBook pro 16
电脑系统：macOS Catalina 10.15.7
手机：iPhone 6s
手机系统：IOS 13.6.1
开发软件：Xcode 12.1

### 配置方式
#### 有线调试
##### 设置证书
1. 打开用户菜单
Xcode → Preferences
<img src="https://sadness96.github.io/images/blog/xcode-RealMachineDebugging/选择菜单Preferences.png"/>

1. 添加用户
菜单中选择 Accounts 添加一个用户
<img src="https://sadness96.github.io/images/blog/xcode-RealMachineDebugging/添加账户.png"/>

1. 添加证书
点击右下角 Manage Certificates… → 添加 Apple Development 证书
<img src="https://sadness96.github.io/images/blog/xcode-RealMachineDebugging/添加证书.png"/>

1. 证书管理
如果需要删除证书，在 keychain access 程序中删除
<img src="https://sadness96.github.io/images/blog/xcode-RealMachineDebugging/钥匙串管理.png"/>

1. 在应用中添加组织
编辑 .xcodeproj 配置 → Signing & Capabilities → 选择刚才添加的用户
<img src="https://sadness96.github.io/images/blog/xcode-RealMachineDebugging/添加组织.png"/>

1. 手机数据线链接至电脑
1. iPhone 中设置信任
设置 → 通用 → 设备管理
<img src="https://sadness96.github.io/images/blog/xcode-RealMachineDebugging/ios设备管理.png"/>

1. 运行程序选择真机设备
如需登录输入钥匙串密码，输入系统密码即可
<img src="https://sadness96.github.io/images/blog/xcode-RealMachineDebugging/钥匙串密码.png"/>

#### 无线调试（需完成有线调试步骤）
1. 打开链接设备菜单
Window → DevicesAndSimulators
<img src="https://sadness96.github.io/images/blog/xcode-RealMachineDebugging/选择菜单DevicesAndSimulators.png"/>

1. 开启远程 Wi-Fi 调试
保证电脑与手机在同一网络下 → 勾选 Connect via network → 左侧设备中显示一个地球则可以 Wi-Fi 调试
<img src="https://sadness96.github.io/images/blog/xcode-RealMachineDebugging/无线调试.png"/>