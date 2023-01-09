---
title: Html 中嵌入 Office 文件
date: 2017-08-05 23:23:00
tags: [html,office]
categories: Html
---
### Html 静态网页中嵌入 Word、Excel、PPT、PDF 文件
<!-- more -->
#### 简介
整理了两篇博客文章，想要把两个 Excel 文件嵌入到网页中显示，遇到了一点小麻烦。
由于博客文件比较精简，尽量避开项目中调用大量脚本。
[数据分析-手机号](https://sadness96.github.io/blog/2017/08/01/data-PhoneNumber/)
[数据分析-身份证号码](https://sadness96.github.io/blog/2017/08/01/data-IdNumber/)

#### Office Web Apps Viewer
使用微软提供的 [Office Online](https://docs.microsoft.com/zh-cn/office365/servicedescriptions/office-online-service-description/office-online-service-description) 实现 Office 文档的在线查看

##### 使用方法
``` html
http://view.officeapps.live.com/op/view.aspx?src=[OFFICE_FILE_URL]
```

##### 嵌入方法
``` html
<iframe src="http://view.officeapps.live.com/op/view.aspx?src=[OFFICE_FILE_URL]" style="width:100%; height:1500px;" frameborder="0"></iframe>
```

##### 存在问题
Microsoft 限制文件大小 Worder、PPT 文件上限为 10MB，Excel 文件上限为 5MB。
由于我保存的 Excel 文件由 MySQL 数据库导出纯文本，没有压缩的空间。
解决：Excel 文件另存为 .xlsb 后缀(二进制工作簿要比正常格式大小小得多)

#### Google Docs Viewer
##### 使用方法
``` html
https://docs.google.com/viewer?url=[OFFICE_FILE_URL]
```

##### 嵌入方法
``` html
<iframe src="https://docs.google.com/viewer?url=[OFFICE_FILE_URL]" style="width:100%; height:1500px;" frameborder="0"></iframe>
```

##### 存在问题
国内对于 Google 并不太友好，有时访问可能需要个[梯子](https://baike.baidu.com/item/%E8%99%9A%E6%8B%9F%E4%B8%93%E7%94%A8%E7%BD%91%E7%BB%9C/8747869?fromtitle=VPN&fromid=382304&fr=aladdin)