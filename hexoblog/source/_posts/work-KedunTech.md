---
title: 科盾科技股份有限公司
date: 2019-06-12 12:05:00
tags: [c#,c++,asp,mysql,axure]
categories: Work
---
<img src="https://sadness96.github.io/images/blog/work-KedunTech/%E4%B8%BB%E7%A8%8B%E5%BA%8F.png"/>

### 人像识别重建系统
<!-- more -->
#### 个人简介
2018年07月12日 - 2019年08月09日 就职于 [科盾科技股份有限公司](http://www.kedun.com/) 软件研发工程师岗位。
#### 公司简介
[科盾科技股份有限公司](http://www.kedun.com/) 成立于1999年，公司总部和研发中心位于北京中关村，试验验证展示基地在昌平区，生产基地在山东莱阳，在济南、深圳、武汉和南京等地设有办事处。目前拥有员工六百余人，是国内领先的安防和夜视产品专业供应商。（股票代码：835902）。

由于拖欠工资，已申请劳动仲裁，强制执行中，拖欠工资差额 40293 元，拖欠公积金 10 个月。
仲裁编号：京海劳人仲字[2019]第 20462 号
案件编号：(2020)京0108执5448号

#### 项目介绍（由于签署保密协议，仅介绍大致工作内容）
人像识别重建系统，主要用于公安部门使用，当前视频监控得到了迅速发展。视频监控普遍存在的问题是：人脸图像分辨率低下。对于低分辨率的人脸图像，需要清晰化目标人的人脸图像和确定监控中目标人的真实身份，以便于公安部门提高破案率。现模拟画像破案率提高30% 指纹提高破案率10%。以及对于寻找失踪人口及儿童得到极大帮助！
##### 声明
该项目为：第八届国际警用装备及反恐技术装备展览会 展出产品，功能及介绍均为宣传内容，如有兴趣请联系科盾科技股份有限公司。
<img src="https://sadness96.github.io/images/blog/work-KedunTech/20190521_164057.jpg"/>

##### 项目背景
该项目由清华大学 [苏光大](https://baike.baidu.com/item/%E8%8B%8F%E5%85%89%E5%A4%A7/4797223) 教授多项国家发明专利为基础研发。主要使用技术有 [OpenCV](https://baike.baidu.com/item/opencv/10320623?fr=aladdin)、[深度学习](https://baike.baidu.com/item/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/3729729?fr=aladdin)、[DNN（深度神经网络）](https://baike.baidu.com/item/DNN/19974079?fr=aladdin)、[Caffe（卷积神经网络框架）](https://baike.baidu.com/item/Caffe/16301044?fr=aladdin) 构成。

##### 参与内容
在该项目中负责完整的架构设计、系统原型设计（[Axure RP](https://baike.baidu.com/item/axure%20rp/9653646?fromtitle=axure&fromid=5056136&fr=aladdin)）、系统流程设计（[Visio](https://baike.baidu.com/item/Microsoft%20Office%20Visio/7180347?fromtitle=VISIO&fromid=357215)）、系统UI设计（[Photoshop](https://baike.baidu.com/item/Adobe%20Photoshop/2297297?fromtitle=PS&fromid=13323&fr=aladdin)、[Illustrator](https://baike.baidu.com/item/Adobe%20Illustrator/2297548?fromtitle=AI&fromid=1753722#viewPageContent)）、编码开发（不包含算法部分）。

##### 设计概要
系统现由服务端与客户端组成：
服务端采用 ASP.NET MVC，后转 [.NET Core](https://baike.baidu.com/item/.net%20core/20130686?fr=aladdin) 框架。
客户端采用 [WPF MVVM](https://baike.baidu.com/item/MVVM/96310?fr=aladdin)、[Prism](https://github.com/PrismLibrary/)、[NSIS](/blog/2018/11/24/software-Nsis/) 架构；
其他技术使用：
数据库：[MySQL](https://www.mysql.com/)
高速缓存：[Redis](https://redis.io/)
消息队列：[RabbitMQ](https://www.rabbitmq.com/)
##### 重建流程
<img src="https://sadness96.github.io/images/blog/work-KedunTech/%E4%BA%BA%E5%83%8F%E8%AF%86%E5%88%AB%E9%87%8D%E5%BB%BA%E6%B5%81%E7%A8%8B.png"/>

##### 功能介绍
###### 超分辨率重建
使监控拍摄到的模糊人像通过算法优化重建出意向人脸，增加案件破获概率。
1、加载监控拍摄到的图像（正脸角度60以内）
<img src="https://sadness96.github.io/images/blog/work-KedunTech/%E9%87%8D%E5%BB%BA%E7%9B%91%E6%8E%A7%E6%8B%8D%E6%91%84%E8%A7%86%E9%A2%91%E6%88%AA%E5%9B%BE.jpg"/>

2、归一化图像480*360（自动处理）；调整直方图均衡及噪点；超分辨率重建
<img src="https://sadness96.github.io/images/blog/work-KedunTech/%E8%B6%85%E5%88%86%E8%BE%A8%E7%8E%87%E9%87%8D%E5%BB%BA.png"/>

###### 人脸识别
使超分辨率重建得到的意向人脸或清晰人脸照片匹配数据库中相似的人像数据，从而排查可疑人员。
###### 模拟画像
通过人脸识别到的人像自由添加替换五官部件、调整，达到更趋向于犯罪嫌疑人的画像。
<img src="https://sadness96.github.io/images/blog/work-KedunTech/%E6%A8%A1%E6%8B%9F%E7%94%BB%E5%83%8F.png"/>