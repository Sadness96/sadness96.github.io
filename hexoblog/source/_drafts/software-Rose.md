---
title: Rose 使用介绍
date: 2020-05-11 17:50:00
tags: [software,rose]
categories: Software
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/rose_log.png"/>

### 本地灾备、远程容灾，快速恢复数据和业务
<!-- more -->
#### 简介
[RoseReplicator](http://www.rosedata.com/index.php/Prodetail/index/proid/24)是基于网络的实时数据容灾复制以及业务连续性保护产品，实现生产数据的本地/远程实时容灾备份，保证数据的安全。实时监测应用资源运行状态，实现资源故障时自动/手动应急切换，解决软、硬件的单点故障，提供高级别的容灾保护。

#### 前置条件
1.虚拟机（VMware）测试，两台（Windows Server 2016）
2.两台机器部署 MySQL，版本 5.7.29
3.两台机器部署 RoseReplicatorPlus，版本 5.8.0
4.Rose 客服人员微信号：cathy_shen1001
5.两台虚拟机IP地址：
  主机：192.168.5.210;192.168.6.210;
  备机：192.168.5.215;192.168.6.215;
  虚拟地址：192.168.5.212;

#### 创建管理软件账户（主备都要创建）
1.进入安装目录 C:\Program Files\ReplicatorPlus\bin 以管理员方式运行（否则会报 File open error 错误） account.exe 应用程序。
2.根据需求创建不同权限的用户。
3.输入用户名密码，显示 Change successfully! 注册成功。
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/注册.png"/>

#### 向导安装
1.选择：系统→配置向导
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/向导1.png"/>

2.选择活动主机登录
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/向导2.png"/>

3.选择备用主机登录
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/向导3.png"/>

4.添加客服手里要来的注册码
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/向导4.png"/>

5.创建主机关系，创建心跳网(心跳要求与局域网IP不同网段，添加一个新网段后可在主机列表更新系统信息后重新创建心跳，测试使用时把心跳IP网段改为了6网段)
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/向导5.png"/>

6.选择应用服务类型，类型中不包含MySQL，客服建议选择USERDEF（用户自定义）
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/向导6.png"/>

7.应用服务数据→勾选绑定数据→添加一对心跳包网段的IP对，定制用于复制的数据集合点击修改→选择mysql数据库data目录（Rose会同步MySQL的Data目录）。
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/向导7.png"/>

8.设置IP资源：勾选主机网卡中局域网IP对，活动IP中新增一个局域网中未规划的IP。
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/向导8.png"/>

9.仲裁磁盘：跳过
10.选择共享卷：跳过
11.选择NT服务：添加MySQL57服务，弹出确定将非手动启动类型的NT服务修改为手动启动类型选择是。
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/向导9.png"/>

12.设置文件共享：跳过
13.进程：跳过，这个主要是针对第三方开发的应用程序，通过exe启动，没有在Windows服务列表中注册服务的，如果是mysql应用不需要配置。
14.设置代理参数：跳过，这个主要是针对bat启动的应用
15.配置详细信息：确认无误点击完成开始配置。
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/向导10.png"/>

16.关闭后咨询是否带入，点击否。

#### 后续操作
1.手动停止两台电脑MySQL，手动备份Data目录。
2.在RoseReplicatorPlus 控制中心点击中间方块右键带入。
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Rose/主界面.png"/>

3.修改数据库内容后，数据同步正常。
4.在方块上右键转移，连接数据库同步后同步正常。
5.关闭一台机器后，另一台电脑 MySQL 服务正常启动。