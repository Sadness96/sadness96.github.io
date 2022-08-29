---
title: 网页嵌入 Bilibili 视频
date: 2022-08-30 00:00:20
tags: [html,iframe,bilibili]
categories: Html
---
### 网页使用 iframe 嵌入 Bilibili 视频
<!-- more -->
### 简介
[网页使用 iframe 嵌入部分其他网页](https://sadness96.github.io/blog/2021/01/02/html-IframeEmbed/)
嵌入 [Bilibili](https://www.bilibili.com) 视频需要使用增加额外的宽高才能显示，并且有一些参数可以配置。通过视频下的 转发 -> 嵌入代码 可以获取到嵌入链接，修改参数后嵌入网页。

### 参数
| key | 说明 |
| --- | --- |
| aid | 视频ID |
| cid | 没什么用 |
| page | 第几个视频, 起始下标为 1 (默认值也是为1) |
| as_wide | 是否宽屏 1: 宽屏, 0: 小屏 |
| high_quality | 是否高清 1: 高清, 0: 最低视频质量(默认) |
| danmaku | 是否开启弹幕 1: 开启(默认), 0: 关闭 |
| allowfullscreen | 是否全屏 true：全屏 |

### 代码
``` html
<iframe src="https://player.bilibili.com/player.html?aid=706&page=1&high_quality=1&danmaku=0&allowfullscreen=true" width="100%" height="500px" scrolling="no" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>
```

### 示例
<iframe src="https://player.bilibili.com/player.html?aid=706&page=1&high_quality=1&danmaku=0&allowfullscreen=true" width="100%" height="500px" scrolling="no" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>