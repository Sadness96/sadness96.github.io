---
title: 网页使用 iframe 嵌入部分其他网页
date: 2021-01-02 17:01:00
tags: [html,iframe]
categories: Html
---
### 在网页中嵌入其他网页中的一部分
<!-- more -->
### 简介
[iframe](https://www.w3school.com.cn/tags/tag_iframe.asp) 元素会创建包含另外一个文档的内联框架，平时使用不多，但是也确实蛮好用的，比如在[博客关于页面](https://sadness96.github.io/blog/about/)中嵌入了一个[网易云音乐的播放插件](https://music.163.com/#/outchain/2/28445602/)，但是有时在嵌入其他页面时直想截取部分嵌入，再此解决。

### 如何使用
#### 最简单的加载
加载一个中国天气网中北京的天气预报，看起来很简单，伴随这一篇没用的广告，不认真看甚至根本不敢相信这是中国天气预报的官方网站。

##### 代码
``` html
<iframe src="http://www.weather.com.cn/weather/101010100.shtml" width="100%" height="400px" sframeborder="0"></iframe>
```

##### 示例
<iframe src="http://www.weather.com.cn/weather/101010100.shtml" width="100%" height="400px" sframeborder="0"></iframe>

#### 仅截取天气预报地图部分并嵌入在网页中
##### 原理
1. 加载一个 iframe 标签，设置网页长度宽度拉伸网页确保样式符合预期，重要内容无广告遮挡。
    <img src="https://sadness96.github.io/images/blog/html-IframeEmbed/iframe1.jpg"/>
1. iframe 外层添加 div 标签，用于移动 iframe 嵌入网页的坐标，宽度高度为实际选取内容宽度高度，参考 [CSS margin 属性](https://www.w3school.com.cn/cssref/pr_margin.asp) 设置偏移量
1. div 外层在添加一层 div 标签作为遮罩层，用于遮罩偏移量产生的多余信息，宽度高度为实际选取内容宽度高度，设置边框 0 与溢出隐藏 [CSS overflow 属性](https://www.w3school.com.cn/css/pr_pos_overflow.asp)
    <img src="https://sadness96.github.io/images/blog/html-IframeEmbed/iframe2.jpg"/>

##### 代码
``` html
<div style="width:680px;height:640px;overflow:hidden;border:0px;"> 
  <div style="width:680px;height:640px;margin:-145px 0px 0px -140px;"> 
   <iframe src="http://www.weather.com.cn/weather/101010100.shtml" height="850" width="1280" frameborder="0"></iframe> 
  </div> 
</div>
```

##### 示例
<div style="width:680px;height:640px;overflow:hidden;border:0px;"> 
  <div style="width:680px;height:640px;margin:-145px 0px 0px -140px;"> 
   <iframe src="http://www.weather.com.cn/weather/101010100.shtml" height="850" width="1280" frameborder="0"></iframe> 
  </div> 
</div>

#### 其他示例(嵌入可交互网页)
另一篇博客 [Luminox 8821 更换电池](https://sadness96.github.io/blog/2021/01/02/repair-Luminox8821/) 中嵌入日本官网(瑞士官网仅有文档美国官网被跳转成某东了)中的可交互页面
##### 代码
``` html
<div style="width:100%;height:820px;overflow:hidden;border:0px;">
  <div style="width:100%;height:820px;margin:-90px 0px 0px 0px;">
   <iframe src="https://luminox.jp/watch-collection/land/recon-point-man-8820-series-ref8821-km/" scrolling="no" height="900" width="767" frameborder="0"></iframe>
  </div>
</div>
```

##### 示例
<div style="width:100%;height:820px;overflow:hidden;border:0px;">
  <div style="width:100%;height:820px;margin:-90px 0px 0px 0px;">
   <iframe src="https://luminox.jp/watch-collection/land/recon-point-man-8820-series-ref8821-km/" scrolling="no" height="900" width="767" frameborder="0"></iframe>
  </div>
</div>