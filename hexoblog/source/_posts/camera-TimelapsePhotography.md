---
title: Canon 相机延时摄影
date: 2022-08-28 02:46:55
tags: [c++,canon]
categories: Camera
---
<img src="https://sadness96.github.io/images/blog/camera-TimelapsePhotography/mmexport1619697203114.jpg"/>

<!-- more -->
### 简介
[延时摄影](https://en.wikipedia.org/wiki/Time-lapse_photography) 是以一种将时间压缩的拍摄技术。其拍摄的通常是一组照片，后期通过将照片串联合成视频，把几分钟、几小时甚至是几天的过程压缩在一个较短的时间内以视频的方式播放。在一段延时摄影视频中，物体或者景物缓慢变化的过程被压缩到一个较短的时间内，呈现出平时用肉眼无法察觉的奇异精彩的景象。

### 延时设置
[Canon EOS 5D Mark IV](https://cweb.canon.jp/eos/lineup/5dmk4/index.html) 提供两种延时拍摄方式，一种为延时短片，一种为定时间隔拍照。
#### 延时短片
佳能提供以常规录像方式的延时摄影，有点是设置简单，拍摄完成直接输出视频文件，但缺点是仅支持 1080P 的分辨率。需要切换到视频模式，并且关闭 WIFI 后才可设置。
<img src="https://sadness96.github.io/images/blog/camera-TimelapsePhotography/延时短片1.jpg"/>

<img src="https://sadness96.github.io/images/blog/camera-TimelapsePhotography/延时短片2.jpg"/>

<img src="https://sadness96.github.io/images/blog/camera-TimelapsePhotography/延时短片3.jpg"/>

#### 定时间隔拍照
定时间隔拍照为常规延时视频制作方式，优点是每一帧的画质都可以设置为相机拍照原本的画质，测试 5d4 合成视频上传后可以显示为 8K 超高清，缺点是需要使用软件合成视频，并且拍摄时不断的触发快门，对相机产生消耗，或许微单比单反更适合这样拍摄。
<img src="https://sadness96.github.io/images/blog/camera-TimelapsePhotography/延时拍照1.jpg"/>

<img src="https://sadness96.github.io/images/blog/camera-TimelapsePhotography/延时拍照2.jpg"/>

<img src="https://sadness96.github.io/images/blog/camera-TimelapsePhotography/延时拍照3.jpg"/>

### Adobe Premiere 导入
使用 [Adobe Premiere](https://www.adobe.com/products/premiere.html) 导入素材，导入媒体选择第一张图片，勾选：图像序列 即可，拖入时间轴即可编辑视频
<img src="https://sadness96.github.io/images/blog/camera-TimelapsePhotography/PrImport.jpg"/>

<img src="https://sadness96.github.io/images/blog/camera-TimelapsePhotography/PrTimeAxis.jpg"/>

### 演示实拍
由于实拍时处于阴天，效果并不是很理想，第一次实拍在曝光的设置上有些问题，导致黑天时过暗，下次在好好调。
Bilibili 外链限制清晰度，打开原视频最高可看 8K 超高清。
<iframe src="https://player.bilibili.com/player.html?aid=644908659&page=1&high_quality=1&danmaku=0&allowfullscreen=true" width="100%" height="500px" scrolling="no" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>

### 拍摄地点
<div id="allmap"></div>

<style type="text/css">
    #allmap {
        width: 100%;
        height: 400px;
        overflow: hidden;
        margin: 0;
    }
</style>

<script type="text/javascript" src="https://api.map.baidu.com/api?v=3.0&ak=tgELalGNumraHZdVurYllitGmvd7RC6R"></script>
<script>
    window.onload = function () {
        //加载百度地图
        var map = new BMap.Map("allmap");
        var point = new BMap.Point(116.398636, 40.025161);
        map.centerAndZoom(point, 17);
        map.enableScrollWheelZoom();
        var marker = new BMap.Marker(point);
        map.addOverlay(marker);
    }
</script>