---
title: HttpClient 上传文件
date: 2022-05-20 21:33:17
tags: [c#,helper,webapi]
categories: C#.Net
---
### 使用 HttpClient form-data 上传文件
<!-- more -->
### 简介
用户需求，通过 multipart/form-data 类型 HTTP 请求上传文件，原帮助类代码参考 [WebApi 帮助类](https://sadness96.github.io/blog/2018/08/27/csharp-WebApiHelper/)，修改代码发送 HttpClient 表单文件上传请求。

### 代码
``` CSharp
/// <summary>
/// 创建Post请求
/// </summary>
/// <param name="url">Api访问地址</param>
/// <param name="requestUrl">详细方法路径</param>
/// <returns>Api返回值</returns>
public static string CreatePostHttpClient(string url, string requestUrl)
{
    try
    {
        string filePath = "1.jpg";
        string savePath = "/danger/reform/before/";

        HttpClient httpClient = new HttpClient();
        httpClient.BaseAddress = new Uri(url);
        // 创建使用 multipart/form-data MIME 类型进行编码的内容提供容器。
        MultipartFormDataContent form = new MultipartFormDataContent();
        // 加载文件流
        FileStream fs = File.OpenRead(filePath);
        var streamContent = new StreamContent(fs);
        // 添加到基于字节数组的 HTTP 内容
        var imageContent = new ByteArrayContent(streamContent.ReadAsByteArrayAsync().Result);
        // 设置 HTTP 响应上的 Content-Type 内容标头值为 multipart/form-data
        imageContent.Headers.ContentType = MediaTypeHeaderValue.Parse("multipart/form-data");
        // 添加文件类型数据
        form.Add(imageContent, "file", Path.GetFileName(filePath));
        // 添加其他文本类型数据
        form.Add(new StringContent(savePath), "path");

        var result = httpClient.PostAsync(requestUrl, form).Result;
        return result.Content.ReadAsStringAsync().Result;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return "";
    }
}
```