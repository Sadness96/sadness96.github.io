---
title: ActiveMQ Demo
date: 2019-12-31 17:50:00
tags: [c#,activemq]
categories: C#.Net
---
### ActiveMQ 消息队列使用介绍
<!-- more -->
#### 简介
[Apache ActiveMQ](http://kafka.apache.org/) 是一个开放源代码的消息中间件。
#### 安装部署
请参阅[官方文档](https://activemq.apache.org/)
##### Docker 部署
``` cmd
安装官方镜像
docker pull webcenter/activemq
启动 RabbitMQ 默认账户密码为 admin/admin
docker run -d --name myactivemq -p 61617:61616 -p 8162:8161 webcenter/activemq
WEB 端登录
http://localhost:8162/
```
#### C#代码调用
引用 [Apache.NMS.ActiveMQ](https://cwiki.apache.org/confluence/display/NMS/Apache.NMS.ActiveMQ) 库
