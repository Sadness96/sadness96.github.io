---
title: Clumsy 使用介绍
date: 2020-03-20 14:50:00
tags: [software,clumsy]
categories: Software
---
<img src="http://jagt.github.io/clumsy/clumsy-demo.gif"/>

### 用于模拟极端网络环境的网络干扰软件
<!-- more -->
#### 简介
[Clumsy](https://github.com/jagt/clumsy) makes your network condition on Windows significantly worse, but in a controlled and interactive manner.
公司工作需要，使用该软件模拟测试极端网络情况下的软件稳定性。
#### 使用
官方手册：[http://jagt.github.io/clumsy/manual.html](http://jagt.github.io/clumsy/manual.html)

##### 配置过滤条件
参考 [WinDivert](https://reqrypt.org/windivert-doc.html#filter_language) 配置文档

##### 干扰方式

| Function | 翻译 | 功能说明 |
| ---- | ---- | ---- |
| Lag | 滞后 | 将数据包保留一小段时间以模拟网络滞后 |
| Dorp | 丢弃 | 随机丢弃数据包 |
| Throttle | 节流阀 | 在给定的时间段内阻塞流量，然后分批发送 |
| Duplicate | 复制 | 将克隆后的数据包立即发送到原始数据包 |
| Out of order | 乱序 | 重新排列数据包的顺序 |
| Tamper | 篡改 | 微调数据包内容的位 |