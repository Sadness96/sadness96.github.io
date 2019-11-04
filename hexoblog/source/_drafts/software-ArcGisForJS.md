---
title: ArcGIS API for JavaScript 使用介绍
date: 2019-11-4 11:05:38
tags: [software,arcgis]
categories: Software
---
# ArcGIS API for JavaScript 搭建使用
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
