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
##### mac
当前显示 http 端口为：8118；socks5 端口为：8119；
<img src="https://sadness96.github.io/images/blog/system-VPNGitAgent/MonoPort.png"/>

#### 配置 Git 代理
``` cmd
git config --global http.proxy http://127.0.0.1:{port}
git config --global https.proxy http://127.0.0.1:{port}

或

git config --global http.proxy socks5://127.0.0.1:{port}
git config --global https.proxy socks5://127.0.0.1:{port}
```

#### 查询配置
``` cmd
git config --global --list

或

git config --global --get http.proxy
git config --global --get https.proxy
```

#### 取消 Git 代理配置
``` cmd
git config --global --unset http.proxy
git config --global --unset https.proxy
```