---
title: Asp服务端文件下载
date: 2019-03-11 12:57:30
tags: [asp,download]
categories: Asp.Net
---
### 服务端提供网页点击下载功能，或 WebApi 接口提供软件更新下载服务
<!-- more -->
#### 继承 Controller 接口的控制器方法下载方式
使用环境：网页端按钮点击下载
按钮脚本链接进入这个方法之后，服务端直接生成文件，浏览器自动下载

``` CSharp
/// <summary>
/// 下载文件
/// </summary>
/// <returns>是否下载成功</returns>
public ActionResult DownLoadFile()
{
    string filePath = "服务端文件路径（程序自动获取路径或根据传入参数判断文件路径）";
    FileInfo fileInfo = new FileInfo(filePath);

    if (fileInfo.Exists)
    {
        Response.Clear();
        Response.ClearContent();
        Response.ClearHeaders();
        Response.Buffer = true;

        Response.AddHeader("Content-Disposition", "attachment;filename=" + Server.UrlEncode(fileInfo.Name));
        Response.AddHeader("Content-Length", fileInfo.Length.ToString());
        Response.AddHeader("Content-Transfer-Encoding", "binary");
        Response.ContentType = "application/x-msdownload";
        Response.ContentEncoding = System.Text.Encoding.GetEncoding("gb2312");

        Response.WriteFile(fileInfo.FullName);
        Response.Flush();
        Response.End();
        return Json(true);
    }
    else
    {
        return Json(false);
    }
}
```

#### 继承 ApiController 接口的 WebApi 方法下载方式
使用环境：桌面端自动更新 提交需要更新的文件名 获取下载
服务端返回文件流，桌面端接收并保存到指定目录替换

``` CSharp
/// <summary>
/// 下载文件
/// </summary>
/// <returns>文件数据流</returns>
[HttpGet]
public byte[] DownLoadFile()
{
    string filePath = "服务端文件路径（程序自动获取路径或根据传入参数判断文件路径）";
    return File.Exists(filePath) ? File.ReadAllBytes(filePath) : null;
}
```