---
title: Android Studio 真机调试
date: 2020-10-23 20:10:00
tags: [android,android studio]
categories: Android
---
<img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/android_logo.png"/>

<!-- more -->
### 简介
Android 开发使用真机有线调试和无线 wifi 调试

### 配置方式
#### 有线调试
Android 手机数据线链接电脑并开启 USB 调试
大部分 Android 默认不显示 USB 调试，多次点击系统版本号开启开发者模式
##### Windows
###### 安装 Google USB Driver
1. 选择 File → Setting…
<img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/选择菜单Windows_Settings.png"/>

1. 选择 Appearance & Behavior → System Settings → Android SDK → SDK Tools 勾选 Android SDK Tools 选项安装
<img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/GoogleUSBDriver.png"/>

##### Mac OS
###### 获取 Android ADB 路径
1. 选择 File → Project Structure…
<img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/选择菜单ProjectStructure.png"/>

1. 选择 SDK Location → 选中位置为 Android ADB 路径
<img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/ProjectStructure.png"/>

###### 配置 Android ADB 环境变量
在 ～/.bash_profile 文件中配置

1. 创建 .bash_profile 文件（如果不存在）
    ``` shell
    cd ~
    touch .bash_profile
    ```

1. 编辑 .bash_profile 文件，ANDROID_HOME 参数为上一步获取的 ADB 路径
    ``` shell
    open .bash_profile
    ```
    <img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/bash_profile文件.png"/>

1. 保存文件后执行配置立即生效命令，输入 adb version 检查是否配置成功
    ``` shell
    source .bash_profile
    adb version
    ```

###### 配置手机可被识别
1. 执行命令获取 usb 接入信息，查询信息中会显示链接的 Android 信息，记录 Vendor ID 备用
    ``` shell
    system_profiler SPUSBDataType
    ```
    <img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/查询usb连接命令.png"/>

1. 在 ～/.android/.adb_usb 文件中配置，目录如下
<img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/adb_usb目录.png"/>

1. 创建 .adb_usb 文件（如果不存在）
    ``` shell
    cd ~
    touch .android/.adb_usb
    ```

1. 编辑 .adb_usb 文件，填写上一步获取的 Vendor ID 并保存
<img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/adb_usb文件.png"/>

1. 执行 adb 命令，显示出链接的手机信息后可以真机调试
    ``` shell
    adb devices
    ```
    <img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/判断是否连接成功.png"/>

#### 无线调试（需完成有线调试步骤）
##### Windows

##### Mac OS
1. 使用命令检测 Android 手机是否链接正常（有线连接）
    ``` shell
    adb devices
    ```

1. 创建 adb 端口
    使用命令 adb tcpip [port] 让手机的某个端口处于监听状态
    服务器端通过扫描5555到5585之间的奇数端口来寻找模拟器或设备实例并与找到的建立链接。
    ``` shell
    adb topic 5555
    ```
    返回 restarting in TCP mode port:5555 为创建成功

1. 创建 adb 无线连接
    保证手机与电脑在一个网络中
    使用命令行 adb connect [ip-address]:[port-num] 连接手机
    命令中 ip 地址为手机在网络中的 ip 地址，端口号为上一步创建的监听端口
    ``` shell
    adb connect 192.168.1.101:5555
    ```
    返回 connected to 192.168.1.101:5555 为创建成功，此时可以断开有线连接

1. 重新执行命令检查 adb 链接情况
    ``` shell
    adb devices
    ```
    返回 {ip 地址}:{端口号} 正确则可以进行无线调试

1. 断开 Wi-Fi 链接
    使用命令 adb disconnect [ip-address]:[port-num] 来中断连接
    ``` shell
    adb disconnect 192.168.1.101:5555
    ```
    返回 disconnected 192.168.1.101:5555 断开成功

1. 命令展示
    <img src="https://sadness96.github.io/images/blog/android-RealMachineDebugging/无线调试.png"/>