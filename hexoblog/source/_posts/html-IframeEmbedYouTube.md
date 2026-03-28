---
title: 网页嵌入 YouTube 视频
date: 2026-03-26 17:07:20
tags: [html,iframe,youtube]
categories: Html
---
### 网页使用 iframe 嵌入 YouTube 视频
<!-- more -->
### 简介
相似功能参考 [网页嵌入 Bilibili 视频](https://sadness96.github.io/blog/2022/08/30/html-IframeEmbedBilibili/)。

### 参数
#### 播放控制类
| 参数         | 说明       | 取值              |
| ---------- | -------- | --------------- |
| `autoplay` | 是否自动播放   | `1` 自动 / `0` 关闭 |
| `mute`     | 是否静音播放   | `1` 静音 / `0` 有声 |
| `start`    | 从第几秒开始播放 | 秒数（如 `10`）      |
| `end`      | 播放到第几秒结束 | 秒数              |
| `loop`     | 是否循环播放   | `1` 循环          |

#### 界面控制类
| 参数               | 说明            | 取值              |
| ---------------- | ------------- | --------------- |
| `controls`       | 显示播放控制条       | `1` 显示 / `0` 隐藏 |
| `modestbranding` | 减少 YouTube 标志 | `1`             |
| `rel`            | 结束后推荐视频来源     | `0` 仅当前频道       |
| `fs`             | 是否允许全屏        | `1` 允许 / `0` 禁用 |

#### 兼容性类
| 参数            | 说明           | 取值                       |
| ------------- | ------------ | ------------------------ |
| `playsinline` | iOS 内联播放     | `1` 推荐使用                 |
| `enablejsapi` | 允许 JS 控制播放器  | `1`                      |
| `origin`      | 安全域名（SEO/安全） | `https://yourdomain.com` |

#### 时间控制类
| 参数      | 说明       |
| ------- | -------- |
| `start` | 从指定秒开始播放 |
| `end`   | 播放到指定秒停止 |

### 代码
``` html
<iframe width="560" height="315" src="https://www.youtube.com/embed/FtutLA63Cp8?si=ND5fBLJxG31yzFgL" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
```

### 示例
<iframe width="560" height="315" src="https://www.youtube.com/embed/FtutLA63Cp8?si=ND5fBLJxG31yzFgL" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>