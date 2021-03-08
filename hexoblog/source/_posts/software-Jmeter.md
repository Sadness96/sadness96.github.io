---
title: Jmeter 使用介绍
date: 2020-12-25 23:10:00
tags: [software,jmeter]
categories: Software
---
<img src="https://sadness96.github.io/images/blog/software-Jmeter/JMeterLogo.png"/>

<!-- more -->
### 简介
[Apache JMeter](https://jmeter.apache.org/) 是Apache组织开发的基于Java的压力测试工具。用于对软件做压力测试，它最初被设计用于Web应用测试，但后来扩展到其他测试领域。 
对测试并不是很熟悉，跟朋友聊天时提到了这个软件，正好也想对自己原来做的东西做一下压力测试，结果实实在在的发现了不少问题。

### 搭建环境
#### 前置条件
[Java 8+](https://www.oracle.com/java/technologies/)

#### 下载安装 Jmeter
[apache jmeter](https://jmeter.apache.org/download_jmeter.cgi)
解压文件

#### 配置环境变量
| 变量名 | 变量值 |
| ---- | ---- |
| JMETER_HOME | 解压路径 |
| Path | %JMETER_HOME%\bin |

#### 控制台输入 jmeter 即可运行
``` cmd
PS C:\Users\Administrator> jmeter
================================================================================
Don't use GUI mode for load testing !, only for Test creation and Test debugging.
For load testing, use CLI Mode (was NON GUI):
   jmeter -n -t [jmx file] -l [results file] -e -o [Path to web report folder]
& increase Java Heap to meet your test requirements:
   Modify current env variable HEAP="-Xms1g -Xmx1g -XX:MaxMetaspaceSize=256m" in the jmeter batch file
Check : https://jmeter.apache.org/usermanual/best-practices.html
================================================================================
```

#### 初步使用
##### 添加线程组
可在线程组中设置压力测试线程数量以及重复次数
<img src="https://sadness96.github.io/images/blog/software-Jmeter/添加线程组.png"/>

##### 添加自定义变量
创建自定义变量，在后续的配置中使用可随时调用
<img src="https://sadness96.github.io/images/blog/software-Jmeter/添加自定义变量.png"/>

##### 添加信息头管理器
用于请求默认信息头，例如：
content-type：application/json; charset=UTF-8
<img src="https://sadness96.github.io/images/blog/software-Jmeter/添加信息头管理器.png"/>

##### http 请求
可直接用 "${}" 的方式直接使用前面添加的自定义变量
<img src="https://sadness96.github.io/images/blog/software-Jmeter/http请求_用户登录.png"/>

##### 正则表达式提取 Token
正常情况下多数接口都需要通过 Token 认证在能访问，但是 Token 通常从登录接口中返回，通过正则表达式匹配到 Token 数据存为自定义变量使用
<img src="https://sadness96.github.io/images/blog/software-Jmeter/正则提取器_提取token存为变量.png"/>

##### 提取到的 Token 变量存入信息头管理器
Token 的使用方法可能不仅限于通过参数传递，也可能通过信息头来传递
<img src="https://sadness96.github.io/images/blog/software-Jmeter/Token添加至头信息管理器.png"/>

##### 调用带 Token 认证的 http 请求
<img src="https://sadness96.github.io/images/blog/software-Jmeter/需要Token认证的http请求.png"/>

##### 添加结果树与报告
可在线程组下添加结果树与报告树图标等统计结果，点击顶部绿色三角等待完成压力测试
<img src="https://sadness96.github.io/images/blog/software-Jmeter/添加结果树与报告.png"/>

#### 使用 Badboy 录制
个人感觉配置 Jmeter 是一件很麻烦的事情，可以使用第三方工具 [Badboy](http://www.badboy.com.au/) 录制操作并导出为 .jmx 格式后由 Jmeter 测试。
官方网站打不开可以从 [softonic](https://badboy.en.softonic.com/) 下载