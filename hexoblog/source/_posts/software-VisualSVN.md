---
title: VisualSVN 使用介绍
date: 2017-08-20 12:23:55
tags: [software,svn]
categories: Software
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-VisualSVN/VisualSVN-Server.png"/>
### 使用 SVN 作为版本管理工具
<!-- more -->
#### 简介
[SVN](https://baike.baidu.com/item/SVN) 是一个开放源代码的[版本控制系统](https://baike.baidu.com/item/%E7%89%88%E6%9C%AC%E6%8E%A7%E5%88%B6/3311252)，现大部分公司还是使用SVN作为代码托管服务，我曾经提议公司将版本控制替换为 [GIT](https://baike.baidu.com/item/GIT/12647237)，但是项目经理有考虑员工学习成本，最后没有使用。
#### 部署服务端（VisualSVN Server）
[VisualSVN Server](https://www.visualsvn.com/) 使Subversion服务器在Windows上安装和管理变得简单方便。[下载地址](https://www.visualsvn.com/server/download/) 安装即可。
#### 设置
##### 创建成员
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-VisualSVN/CreateNewUser.png"/>
##### 创建项目库
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-VisualSVN/CreateNewRepository.png"/>
##### 设置项目名
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-VisualSVN/CreateNewRepositoryName.png"/>
##### 设置项目结构
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-VisualSVN/CreateNewRepositoryStructure.png"/>
##### 设置项目访问权限
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-VisualSVN/CreateNewRepositoryAccess.png"/>
##### 设置项目成员
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-VisualSVN/AddUser.png"/>
#### 客户端连接
[TortoiseSVN](https://tortoisesvn.net/downloads.html) 是一个Apache ™ Subversion（SVN）&reg;客户端，实现为Windows外壳扩展。它直观且易于使用，因为它不需要Subversion命令行客户端运行。[下载地址](https://tortoisesvn.net/downloads.html) 安装即可。
右键 SVN Checkout… 连接项目库
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-VisualSVN/Checkout.jpg"/>