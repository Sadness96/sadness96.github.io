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
#### 简介
视频的一帧图像写入临时的 canvas 中，通过 [toDataURL](https://developer.mozilla.org/zh-CN/docs/Web/API/HTMLCanvasElement/toDataURL) 方法转换为 Base64 数据，下载 Base64 即可截图。
注：toDataURL 方法引用外部文件会报 CORS 跨域错误，需要单独解决 CORS 跨域。

#### 代码
``` js
// 截图
function Screenshot() {
  const video = document.querySelector("video");
  var imgRecordCanvas = document.createElement('canvas');
  imgRecordCanvas.width = video.videoWidth;
  imgRecordCanvas.height = video.videoHeight;
  imgRecordCanvas.getContext("2d").drawImage(
    video,
    0,
    0,
    imgRecordCanvas.width,
    imgRecordCanvas.height
  );
  var img_base64 = imgRecordCanvas.toDataURL("image/png");
  var savename = "img_" + new Date().getTime();
  DownloadBase64ImageFile(img_base64, savename)
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
// 截图
function Screenshot() {
  const video = document.querySelector("video");
  var imgRecordCanvas = document.createElement('canvas');
  imgRecordCanvas.width = video.videoWidth;
  imgRecordCanvas.height = video.videoHeight;
  imgRecordCanvas.getContext("2d").drawImage(
    video,
    0,
    0,
    imgRecordCanvas.width,
    imgRecordCanvas.height
  );
  var img_base64 = imgRecordCanvas.toDataURL("image/png");
  var savename = "img_" + new Date().getTime();
  DownloadBase64ImageFile(img_base64, savename)
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
#### 简介
视频录制与截图方式差不多，把图像缓存到 canvas 标签，然后通过接口 [MediaRecorder](https://developer.mozilla.org/zh-CN/docs/Web/API/MediaRecorder/MediaRecorder) 录制视频。

#### 代码
``` js
var isVideotape = false;
var videoRecordCanvas = null;
var videoCanvasContext = null;
var videoRecorder = null;
var videoFrameId = null;
var videoChunks = [];
const video = document.querySelector("video");
var StartVideotape = document.getElementById('StartVideotape');
var StopVideotape = document.getElementById('StopVideotape');
// 开始录像
function VideotapeStart() {
  StartVideotape.disabled = "disabled";
  StopVideotape.disabled = "";
  isVideotape = true;
  // 开始录制
  videoRecordCanvas = document.createElement('canvas');
  videoRecordCanvas.width = video.videoWidth;
  videoRecordCanvas.height = video.videoHeight;
  videoCanvasContext = videoRecordCanvas.getContext("2d");
  videoCanvasContext.fillStyle = "deepskyblue";
  videoCanvasContext.fillRect(0, 0, videoRecordCanvas.videoWidth, videoRecordCanvas.videoHeight);
  // 创建MediaRecorder，设置媒体参数
  var frameRate = 60;
  var stream = videoRecordCanvas.captureStream(frameRate);
  videoRecorder = new MediaRecorder(stream, {
    mimeType: "video/webm;codecs=vp8",
  });
  // 收集录制数据
  videoRecorder.ondataavailable = (e) => {
    videoChunks.push(e.data);
  };
  videoRecorder.start(10);
  // 播放视频
  DrawFrame();
}
// 播放视频
function DrawFrame() {
  if (videoCanvasContext && videoRecordCanvas) {
    videoCanvasContext.drawImage(
      video,
      0,
      0,
      video.videoWidth,
      video.videoHeight
    );
    videoFrameId = requestAnimationFrame(this.DrawFrame);
  }
}
// 停止录像
function VideotapeStop() {
  StartVideotape.disabled = "";
  StopVideotape.disabled = "disabled";
  isVideotape = false;
  // 停止录制
  videoRecorder.stop();
  cancelAnimationFrame(videoFrameId);
  // 下载录制内容
  if (videoChunks.length > 0) {
    const blob = new Blob(videoChunks);
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = new Date().getTime() + ".mp4";
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();
    const lenght = videoChunks.length;
    for (let i = 0; i <= lenght; i++) {
      videoChunks.pop();
    }
  }
}
```

#### 演示
<div>
<button type="button" id="StartVideotape" onclick='VideotapeStart()'>录像</button>
<button type="button" id="StopVideotape" disabled='disabled' onclick='VideotapeStop()'>停止</button>
</div>

<script>
var isVideotape = false;
var videoRecordCanvas = null;
var videoCanvasContext = null;
var videoRecorder = null;
var videoFrameId = null;
var videoChunks = [];
const video = document.querySelector("video");
var StartVideotape = document.getElementById('StartVideotape');
var StopVideotape = document.getElementById('StopVideotape');
// 开始录像
function VideotapeStart() {
  StartVideotape.disabled = "disabled";
  StopVideotape.disabled = "";
  isVideotape = true;
  // 开始录制
  videoRecordCanvas = document.createElement('canvas');
  videoRecordCanvas.width = video.videoWidth;
  videoRecordCanvas.height = video.videoHeight;
  videoCanvasContext = videoRecordCanvas.getContext("2d");
  videoCanvasContext.fillStyle = "deepskyblue";
  videoCanvasContext.fillRect(0, 0, videoRecordCanvas.videoWidth, videoRecordCanvas.videoHeight);
  // 创建MediaRecorder，设置媒体参数
  var frameRate = 60;
  var stream = videoRecordCanvas.captureStream(frameRate);
  videoRecorder = new MediaRecorder(stream, {
    mimeType: "video/webm;codecs=vp8",
  });
  // 收集录制数据
  videoRecorder.ondataavailable = (e) => {
    videoChunks.push(e.data);
  };
  videoRecorder.start(10);
  // 播放视频
  DrawFrame();
}
// 播放视频
function DrawFrame() {
  if (videoCanvasContext && videoRecordCanvas) {
    videoCanvasContext.drawImage(
      video,
      0,
      0,
      video.videoWidth,
      video.videoHeight
    );
    videoFrameId = requestAnimationFrame(this.DrawFrame);
  }
}
// 停止录像
function VideotapeStop() {
  StartVideotape.disabled = "";
  StopVideotape.disabled = "disabled";
  isVideotape = false;
  // 停止录制
  videoRecorder.stop();
  cancelAnimationFrame(videoFrameId);
  // 下载录制内容
  if (videoChunks.length > 0) {
    const blob = new Blob(videoChunks);
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = 'video_' + new Date().getTime() + ".mp4";
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();
    const lenght = videoChunks.length;
    for (let i = 0; i <= lenght; i++) {
      videoChunks.pop();
    }
  }
}
</script>