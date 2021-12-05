---
title: Html Video 显示到 Canvas 中
date: 2021-12-02 11:00:00
tags: [html,video,canvas]
categories: Html
---
### Video 视频画面加载到 Canvas 中并限制显示区域与缩放
<!-- more -->
### 简介
由于项目需要在视频上放大指定区域播放、视频滚轮缩放、绘制特定内容，H5 原生的 [Video](https://www.w3school.com.cn/tags/tag_video.asp) 控件无法满足需求。但是同样的 Canvas 也有不如 Video 的弊端，比如全屏。

### 方法
#### requestVideoFrameCallback
[HTMLVideoElement.requestVideoFrameCallback()](https://wicg.github.io/video-rvfc/) 用于注册回调，在渲染一帧图像时触发。
参考博客：[The requestVideoFrameCallback API](https://blog.tomayac.com/2020/05/15/the-requestvideoframecallback-api/)

##### 回调播放核心代码
``` js
const video = document.querySelector("video");
const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");

const updateCanvas = (now, metadata) => {
  ctx.drawImage(video, 0, 0, width, height);
  video.requestVideoFrameCallback(updateCanvas);
};

video.requestVideoFrameCallback(updateCanvas);
```

##### 回调播放演示
<video width="640px" height="360px" controls playsinline></video>
<canvas width="640px" height="360px" style="border: 1px solid Gray;"></canvas>

<style>
  video,canvas {
    max-width: 100%;
    height: auto;
 }
</style>

<script>
  const startDrawing = () => {
    const video = document.querySelector("video");
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    
    let width = canvas.width;
    let height = canvas.height;

    const updateCanvas = (now, metadata) => {
      ctx.drawImage(video, 0, 0, width, height);
      video.requestVideoFrameCallback(updateCanvas);
    };
    
    video.src = "../../../../../video/test.mp4";
    video.muted = true;
    video.loop = 'loop';
    video.requestVideoFrameCallback(updateCanvas);
    video.play();
  };

  window.addEventListener('load', startDrawing);
</script>

#### 基于回调的区域裁切
通过创建 Canvas 控件时的 drawImage 方法控制裁切显示区域
参考：[Web Api drawImage](https://developer.mozilla.org/zh-CN/docs/Web/API/CanvasRenderingContext2D/drawImage)
例如：裁切起始坐标 50,50，裁切大小 160x90，显示坐标 0,0，显示大小 192x108

##### 核心代码

``` js
const startDrawingCutting = () => {
  const video = document.querySelector("video");
  const canvas = document.getElementById('videoCutting');
  const ctx = canvas.getContext("2d");

  const updateCanvasCutting = (now, metadata) => {
    ctx.drawImage(video, 50, 50, 160, 90, 0, 0, 192, 108);
    video.requestVideoFrameCallback(updateCanvasCutting);
  };
  video.requestVideoFrameCallback(updateCanvasCutting);
};
window.addEventListener('load', startDrawingCutting);
```

##### 回调裁切演示

<canvas id="videoCutting" ></canvas>

<script>
  const startDrawingCutting = () => {
    const video = document.querySelector("video");
    const canvas = document.getElementById('videoCutting');
    const ctx = canvas.getContext("2d");
  
    const updateCanvasCutting = (now, metadata) => {
      ctx.drawImage(video, 50, 50, 160, 90, 0, 0, 192, 108);
      video.requestVideoFrameCallback(updateCanvasCutting);
    };
    video.requestVideoFrameCallback(updateCanvasCutting);
  };
  window.addEventListener('load', startDrawingCutting);
</script>

#### Konva.js
[Konva.js](https://konvajs.org/) 是适用于桌面/移动端应用的 HTML5 2d canvas 库，将视频添加到 Konva 的舞台中，更适合后期操作。
参考：[VideoOnCanvas](https://konvajs.org/docs/sandbox/Video_On_Canvas.html) 将视频加载到 Konva Canvas 中

##### Konva 播放核心代码

``` js
var stage = new Konva.Stage({
  container: 'container',
  width: width,
  height: height,
});
var layer = new Konva.Layer();
stage.add(layer);
var video = document.createElement('video');
var image = new Konva.Image({
  image: video,
  draggable: true,
  x: 0,
  y: 0,
});
layer.add(image);
var anim = new Konva.Animation(function () {
  // do nothing, animation just need to update the layer
}, layer);
// update Konva.Image size when meta is loaded
video.addEventListener('loadedmetadata', function (e) {
  image.width(width);
  image.height(height);
});
```

##### Konva 播放演示

<script src="https://cdnjs.cloudflare.com/ajax/libs/konva/8.3.0/konva.min.js"></script>

<div id="container"></div>

<style>
  #container {
    width: 640px;
    height: 360px;
    border: 1px solid Gray;
 }
</style>

<script>
  var width = 640;
  var height = 360;

  var stage = new Konva.Stage({
    container: 'container',
    width: width,
    height: height,
  });
  var layer = new Konva.Layer();
  stage.add(layer);
  var video = document.createElement('video');
  video.src = '../../../../../video/test.mp4';
  var image = new Konva.Image({
    image: video,
    draggable: false,
    x: 0,
    y: 0,
  });
  layer.add(image);
  var anim = new Konva.Animation(function () {
    // do nothing, animation just need to update the layer
  }, layer);
  // update Konva.Image size when meta is loaded
  video.addEventListener('loadedmetadata', function (e) {
    image.width(width);
    image.height(height);
  });

  video.muted = true;
  video.loop = 'loop';
  video.play();
  anim.start();
</script>

#### 基于 Konva.js 的拖拽和鼠标滚轮缩放
1. 拖拽：创建 Konva 对象时设置 draggable: true 即可拖动
    参考 [复杂的拖拽区域](http://konvajs-doc.bluehymn.com/docs/drag_and_drop/Complex_Drag_and_Drop.html) 可以设置更为详细的拖拽规则
1. 缩放：监听 wheel 方法进行缩放操作

##### 鼠标滚轮缩放核心代码

``` js
var scaleBy = 1.04;
stageZoom.on('wheel', e => {
  e.evt.preventDefault();
  var oldScale = stageZoom.scaleX();
  var mousePointTo = {
    x: stageZoom.getPointerPosition().x / oldScale - stageZoom.x() / oldScale,
    y: stageZoom.getPointerPosition().y / oldScale - stageZoom.y() / oldScale
  };
  var newScale =
    e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;
  stageZoom.scale({ x: newScale, y: newScale });
  var newPos = {
    x:
      -(mousePointTo.x - stageZoom.getPointerPosition().x / newScale) *
      newScale,
    y:
      -(mousePointTo.y - stageZoom.getPointerPosition().y / newScale) *
      newScale
  };
  stageZoom.position(newPos);
  stageZoom.batchDraw();
});
```

##### 拖拽和鼠标滚轮缩放演示

<div id="containerZoom"></div>

<style>
  #containerZoom {
    width: 640px;
    height: 360px;
    border: 1px solid Gray;
 }
</style>

<script>
  var widthZoom = 640;
  var heightZoom = 360;

  var stageZoom = new Konva.Stage({
    container: 'containerZoom',
    width: widthZoom,
    height: heightZoom,
  });
  var layerZoom = new Konva.Layer();
  stageZoom.add(layerZoom);
  var videoZoom = document.createElement('video');
  videoZoom.src = '../../../../../video/test.mp4';
  var imageZoom = new Konva.Image({
    image: videoZoom,
    draggable: true,
    x: 0,
    y: 0,
  });
  layerZoom.add(imageZoom);
  var animZoom = new Konva.Animation(function () {
    // do nothing, animation just need to update the layer
  }, layerZoom);
  // update Konva.Image size when meta is loaded
  videoZoom.addEventListener('loadedmetadata', function (e) {
    imageZoom.width(widthZoom);
    imageZoom.height(heightZoom);
  });

  videoZoom.muted = true;
  videoZoom.loop = 'loop';
  videoZoom.play();
  animZoom.start();

  var scaleBy = 1.04;
  stageZoom.on('wheel', e => {
    e.evt.preventDefault();
    var oldScale = stageZoom.scaleX();
    var mousePointTo = {
      x: stageZoom.getPointerPosition().x / oldScale - stageZoom.x() / oldScale,
      y: stageZoom.getPointerPosition().y / oldScale - stageZoom.y() / oldScale
    };
    var newScale =
      e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;
    stageZoom.scale({ x: newScale, y: newScale });
    var newPos = {
      x:
        -(mousePointTo.x - stageZoom.getPointerPosition().x / newScale) *
        newScale,
      y:
        -(mousePointTo.y - stageZoom.getPointerPosition().y / newScale) *
        newScale
    };
    stageZoom.position(newPos);
    stageZoom.batchDraw();
  });
</script>