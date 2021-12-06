---
title: Html Video 截图和录像
date: 2021-11-26 14:24:00
tags: [html,video,canvas]
categories: Html
---
### Video 视频画面截图与录像
<!-- more -->
### 简介
项目需要在视频上截图与录像，功能已实现，我在迁移时整理一遍。

### 演示视频
<video width="640px" height="360px" crossorigin="anonymous" controls playsinline></video>

<script>
  const startDrawing = () => {
    const video = document.querySelector("video");
    video.src = "../../../../../video/test.mp4";
    video.muted = true;
    video.loop = 'loop';
    video.play();
  };

  window.addEventListener('load', startDrawing);
</script>

### 截图
#### 代码
``` js
// 截图
function Screenshot() {
  const video = document.querySelector("video");
  var RecordCanvas = document.createElement('canvas');
  RecordCanvas.width = video.videoWidth;
  RecordCanvas.height = video.videoHeight;
  RecordCanvas.getContext("2d").drawImage(
    video,
    0,
    0,
    RecordCanvas.width,
    RecordCanvas.height
  );
  var img = document.createElement("img");
  img.src = RecordCanvas.toDataURL("image/png");
  var savename = "img_" + new Date().toLocaleString();
  DownloadBase64ImageFile(img.src, savename)
}

// 下载 Base64 图片
function DownloadBase64ImageFile(content, fileName) {
  var base64ToBlob = function (code) {
  const parts = code.split(";base64,");
  const contentType = parts[0].split(":")[1];
  const raw = window.atob(parts[1]);
  const rawLength = raw.length;
  const uInt8Array = new Uint8Array(rawLength);
  for (let i = 0; i < rawLength; ++i) {
    uInt8Array[i] = raw.charCodeAt(i);
  }
  return new Blob([uInt8Array], {
      type: contentType,
    });
  };
  const aLink = document.createElement("a");
  const blob = base64ToBlob(content);
  const evt = document.createEvent("HTMLEvents");
  evt.initEvent("click", true, true);
  aLink.download = fileName;
  aLink.href = URL.createObjectURL(blob);
  aLink.click();
}
```

#### 演示
<button type="button" onclick="Screenshot()">截图</button>

<script>
function Screenshot() {
  const video = document.querySelector("video");
  var RecordCanvas = document.createElement('canvas');
  RecordCanvas.width = video.videoWidth;
  RecordCanvas.height = video.videoHeight;
  RecordCanvas.getContext("2d").drawImage(
    video,
    0,
    0,
    RecordCanvas.width,
    RecordCanvas.height
  );
  var img = document.createElement("img");
  img.src = RecordCanvas.toDataURL("image/png");
  var savename = "img_" + new Date().toLocaleString();
  DownloadBase64ImageFile(img.src, savename)
}

// 下载 Base64 图片
function DownloadBase64ImageFile(content, fileName) {
  var base64ToBlob = function (code) {
  const parts = code.split(";base64,");
  const contentType = parts[0].split(":")[1];
  const raw = window.atob(parts[1]);
  const rawLength = raw.length;
  const uInt8Array = new Uint8Array(rawLength);
  for (let i = 0; i < rawLength; ++i) {
    uInt8Array[i] = raw.charCodeAt(i);
  }
  return new Blob([uInt8Array], {
      type: contentType,
    });
  };
  const aLink = document.createElement("a");
  const blob = base64ToBlob(content);
  const evt = document.createEvent("HTMLEvents");
  evt.initEvent("click", true, true);
  aLink.download = fileName;
  aLink.href = URL.createObjectURL(blob);
  aLink.click();
}
</script>

### 录像
#### 代码
``` js

```

#### 演示
<button type="button" onclick="Videotape()">录像</button>

<script>
function Videotape() {
  const video = document.querySelector("video");

}
</script>