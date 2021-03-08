---
title: ArcGIS API for JavaScript 使用介绍
date: 2019-11-4 11:05:38
tags: [software,arcgis]
categories: Software
---
<img src="https://sadness96.github.io/images/blog/software-ArcGisForJS/ArcGISForJavaScript.png"/>

<!-- more -->
### 简介
现工作中 GIS 地图使用客户提供的 AutoCAD 图纸提取图层到 ArcGIS 中导出图片模拟 GIS 显示，存在一定弊端（1.不包含经纬度数据，需要进行图像标定坐标转换；2.图像中边线粗的越放大越粗，边线细的缩放时不足一像素颜色减淡。），尝试以客户端加载 WebJS 的方式使用 GIS。
### 搭建环境
基于 ArcGIS 提供的桌面端（用于编辑地图），服务端（用于发布地图服务）以及 ArcGIS API for JavaScript（开发 WebJS）。
#### 搭建 ArcGIS Server 10.2
[参考资料](https://blog.csdn.net/qq_36213352/article/details/80646940)
##### 安装后默认值
地图服务地址：http://localhost:6080/arcgis/manager/
地图服务账号：siteadmin
#### 搭建 ArcGIS Desktop 10.2
[参考资料](https://blog.csdn.net/bigemap/article/details/81131840)
##### 地图服务发布
###### 1.编辑好的地图保存为 .mxd 格式
<img src="https://sadness96.github.io/images/blog/software-ArcGisForJS/SaveMxd.png"/>

###### 2.在 ArcMap 目录中选择 GIS 服务器 → 添加 ArcGIS Server → 发布 GIS 服务
选择发布 GIS 服务
<img src="https://sadness96.github.io/images/blog/software-ArcGisForJS/PublishingService1.png"/>

设置服务器 URL 与用户名密码
<img src="https://sadness96.github.io/images/blog/software-ArcGisForJS/PublishingService2.png"/>

###### 4.在 ArcMap 目录中选择 .mxd 文件右键选择：共享为服务(S)…
选择发布服务
<img src="https://sadness96.github.io/images/blog/software-ArcGisForJS/ShareForService1.png"/>

设置服务名称
<img src="https://sadness96.github.io/images/blog/software-ArcGisForJS/ShareForService2.png"/>

默认发布为根
<img src="https://sadness96.github.io/images/blog/software-ArcGisForJS/ShareForService3.png"/>

1.点击分析，解决错误（例：图层坐标系异常）；2.发布
<img src="https://sadness96.github.io/images/blog/software-ArcGisForJS/ShareForService4.png"/>

选择发布图层等待服务发布
<img src="https://sadness96.github.io/images/blog/software-ArcGisForJS/ShareForService5.png"/>

[ArcGIS Server Manager](http://localhost:6080/arcgis/manager/) 中查看服务
<img src="https://sadness96.github.io/images/blog/software-ArcGisForJS/ShareForService6.png"/>

#### 搭建 ArcGIS API for JavaScript
未完成
