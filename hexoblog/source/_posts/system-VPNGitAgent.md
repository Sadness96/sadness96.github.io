---
title: Git 使用 VPN 代理加速
date: 2020-01-30 23:49:00
tags: [windows,macos,git,vpn]
categories: System
---
### 解决国内 Github 访问过慢
<!-- more -->
### 简介
开了 VPN 之后使用 Git 连接 github 的 clone pull push 命令依旧很慢，是由于 git 默认不使用代理导致，配置 git 代理后可提升速度。

### 配置方法
#### 查找 VPN 本地代理端口
以 [MonoCloud](https://mymonocloud.com/) 为例，由于不同 vpn 软件或安装的随机性导致每台机器的端口号并不一致，以显示为准。
##### MacOS
当前显示 http 端口为：8118；socks5 端口为：8119；
<img src="https://sadness96.github.io/images/blog/system-VPNGitAgent/MacOSMonoPort.png"/>

##### Windows
当前显示 http 与 socks5 端口为：7078；
<img src="https://sadness96.github.io/images/blog/system-VPNGitAgent/WindowsMonoPort.png"/>

#### 配置 Git 代理
``` cmd
git config --global http.proxy http://127.0.0.1:{port}
git config --global https.proxy http://127.0.0.1:{port}

或

git config --global http.proxy socks5://127.0.0.1:{port}
git config --global https.proxy socks5://127.0.0.1:{port}
```

<img src="https://sadness96.github.io/images/blog/system-VPNGitAgent/GitConfigProxy.png"/>

配置成功后可尝试查询配置或重新使用 git 命令

#### 查询配置
``` cmd
git config --global --list

或

git config --global --get http.proxy
git config --global --get https.proxy
```
<img src="https://sadness96.github.io/images/blog/system-VPNGitAgent/GitConfigList.png"/>

#### 取消 Git 代理配置
``` cmd
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### 异常错误
#### 报错 Connection refused
``` cmd
fatal: unable to access 'https://github.com/*/*.git/': Failed to connect to 127.0.0.1 port 7071: Connection refused
```
VPN 的本地映射端口配置错误，检查映射端口配置的正确性，或取消代理。