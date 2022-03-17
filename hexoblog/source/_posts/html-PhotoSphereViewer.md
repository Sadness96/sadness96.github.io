---
title: VR 全景图展示
date: 2022-03-16 20:15:00
tags: [html,vr,stitching]
categories: Html
---
### 通过 Photo Sphere Viewer 展示全景图
<!-- more -->
### 简介
VR 图片广泛应用于房产、车辆、景区等展示推销和宣传作用，介于 360 全景相机较为昂贵，收费模式竟然按平米收取渲染费，国内这些厂家就很离谱。所以先以最低成本尝试实现一个 Demo，以后逐步完善成一个成品。

### 实现
#### 下楼拍一组照片
下楼拍摄一组照片，全景相机的原理也是多个相机同时拍摄后拼接到一起，但是全景相机内使用的应该是鱼眼镜头，拥有超大广角，减少镜头数量就可以减少成本，如果使用手机拍摄就需要多拍一些了，把周身一圈 360 度拍满。
<img src="https://sadness96.github.io/images/blog/html-PhotoSphereViewer/PhoneCamera.jpg"/>

#### 拼接图像
图像的拼接流程大概是：提取特征点 -> 特征点匹配 -> 对图片进行圆柱投影 -> 图片拼接 -> 色差矫正。
不过这里才不会实现这些内容呢，大部分软件都是集成了 [OpenCV](https://opencv.org/) 的 [Stitching](https://docs.opencv.org/4.x/d8/d19/tutorial_stitcher.html) 库。
可以使用以下软件：
1. [AutoStitch](http://matthewalunbrown.com/autostitch/autostitch.html) （推荐）
1. [PTGui](https://ptgui.com/) （效果应该更好，但是付费）
1. [Photoshop](www.photoshop.com/) Photomerge (可能是由于版本问题，效果不尽人意)

以 [AutoStitch](http://matthewalunbrown.com/autostitch/autostitch.html) 举例，简单的添加所有图片后等待即可获得拼接后图片
<img src="https://sadness96.github.io/images/blog/html-PhotoSphereViewer/AutoStitch.jpg"/>

<img src="https://sadness96.github.io/images/blog/html-PhotoSphereViewer/pano.jpg"/>

#### 使用 PS 修复缺失部分（可选）
由于手动拍摄难免出现缝隙，或者上下存在黑洞，就算使用全景相机架着三脚架拍摄，也难免把三脚架拍摄在内，可通过 PS 的 3D 功能修复。
PS 打开全景图片 -> 3D -> 球面全景 -> 通过选中的图层新建全景图图层 -> 拖拽到瑕疵的区域覆盖或修复（就像普通的P图一样） -> 导出全景图（PS会把内容自动抻展为平面）
<img src="https://sadness96.github.io/images/blog/html-PhotoSphereViewer/PSRestore.jpg"/>

#### 展示
使用 [Photo Sphere Viewer](https://photo-sphere-viewer.js.org/) 库渲染

##### 代码
``` html
<!DOCTYPE html>
<html>

<head>
  <!-- for optimal display on high DPI devices -->
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/photo-sphere-viewer@4/dist/photo-sphere-viewer.min.css"/>
</head>
<script src="https://cdn.jsdelivr.net/npm/three/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/uevent@2/browser.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/photo-sphere-viewer@4/dist/photo-sphere-viewer.min.js"></script>

<body>

  <div id="viewer" style="width: 100vw; height: 100vh;"></div>

  </div>
  <script>
    const viewer = new PhotoSphereViewer.Viewer({
      container: document.querySelector('#viewer'),
      panorama: 'image/pano.jpg',
    });
  </script>
  <style>
    html,
    body,
    #viewer {
      margin: 0;
      width: 100%;
      height: 100%;
    }
  </style>
</body>

</html>
```

##### 预览
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/photo-sphere-viewer@4/dist/photo-sphere-viewer.min.css"/>
<script src="https://cdn.jsdelivr.net/npm/three/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/uevent@2/browser.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/photo-sphere-viewer@4/dist/photo-sphere-viewer.min.js"></script>
<div id="viewer" style="margin: 0;width: 100%; height: 75vh;"></div>
<script>
    const viewer = new PhotoSphereViewer.Viewer({
        container: document.querySelector('#viewer'),
        panorama: '/images/blog/html-PhotoSphereViewer/pano.jpg',
    });
</script>
