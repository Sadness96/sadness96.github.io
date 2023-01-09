---
title: C/S端调用 WebApi 帮助类
date: 2018-08-27 19:35:37
tags: [c#,helper,webapi]
categories: C#.Net
---
### 使用 HttpClient 与 HttpWebRequest 两种方式调用 WebApi 帮助类
<!-- more -->
#### 简介
现最常见的软件开发模式就是 [服务端](https://baike.baidu.com/item/%E6%9C%8D%E5%8A%A1%E7%AB%AF/6492316?fr=aladdin)（[B/S](https://baike.baidu.com/item/B%2FS%E7%BB%93%E6%9E%84/4868588?fromtitle=BS&fromid=2629117&fr=aladdin)、[WebApi](https://baike.baidu.com/item/WeBAPI)、[WebServer](https://baike.baidu.com/item/web%20server/9306055)) + [客户端](https://baike.baidu.com/item/%E5%AE%A2%E6%88%B7%E7%AB%AF)([C/S](https://baike.baidu.com/item/Client%2FServer/1504488?fromtitle=cs&fromid=2852264)、[Android](https://baike.baidu.com/item/Android/60243?fromtitle=%E5%AE%89%E5%8D%93&fromid=5389782)、[IOS](https://baike.baidu.com/item/ios/45705))。
公司有部分新项目修改为逻辑在服务端处理，所以通过两种方法封装一个C/S端调用WebApi接口的帮助类。
调试WebApi推荐使用：[Postman](https://www.getpostman.com/)。

#### 帮助类
##### HttpClient
[HttpClientHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/WebApi/HttpClientHelper.cs)

###### 创建Get请求
``` CSharp
/// <summary>
/// 创建Get请求
/// </summary>
/// <param name="url">Api访问地址</param>
/// <param name="requestUrl">详细方法路径</param>
/// <param name="parameters">请求参数</param>
/// <returns>Api返回值</returns>
public static string CreateGetHttpClient(string url, string requestUrl, IDictionary<string, string> parameters)
{
    try
    {
        StringBuilder builder = new StringBuilder();
        builder.Append(url);
        builder.Append(requestUrl);
        if (parameters != null && parameters.Count >= 1)
        {
            builder.Append("?");
            int i = 0;
            foreach (var item in parameters)
            {
                if (i > 0)
                {
                    builder.Append("&");
                }
                builder.AppendFormat("{0}={1}", item.Key, item.Value);
                i++;
            }
        }
        HttpClient httpClient = new HttpClient();
        httpClient.BaseAddress = new Uri(url);
        var result = httpClient.GetAsync(builder.ToString()).Result;
        return result.Content.ReadAsStringAsync().Result;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return "";
    }
}
```

###### 创建Post请求
``` CSharp
/// <summary>
/// 创建Post请求
/// </summary>
/// <param name="url">Api访问地址</param>
/// <param name="requestUrl">详细方法路径</param>
/// <param name="parameters">请求参数</param>
/// <returns>Api返回值</returns>
public static string CreatePostHttpClient(string url, string requestUrl, IDictionary<string, string> parameters)
{
    try
    {
        HttpClient httpClient = new HttpClient();
        httpClient.BaseAddress = new Uri(url);
        var result = httpClient.PostAsync(requestUrl, new FormUrlEncodedContent(parameters)).Result;
        return result.Content.ReadAsStringAsync().Result;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return "";
    }
}
```

##### HttpWebRequest
###### 创建Get请求
``` CSharp
/// <summary>
/// 创建Get请求
/// </summary>
/// <param name="url">Api访问地址</param>
/// <param name="requestUrl">详细方法路径</param>
/// <param name="parameters">请求参数</param>
/// <returns>Api返回值</returns>
public static string CreateGetHttpWebRequest(string url, string requestUrl, IDictionary<string, string> parameters)
{
    try
    {
        StringBuilder builder = new StringBuilder();
        builder.Append(url);
        builder.Append(requestUrl);
        if (parameters != null && parameters.Count >= 1)
        {
            builder.Append("?");
            int i = 0;
            foreach (var item in parameters)
            {
                if (i > 0)
                {
                    builder.Append("&");
                }
                builder.AppendFormat("{0}={1}", item.Key, item.Value);
                i++;
            }
        }
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create(builder.ToString());
        request.Method = "GET";
        request.ContentType = "application/x-www-form-urlencoded";
        StreamReader reader = new StreamReader(request.GetResponse().GetResponseStream(), Encoding.UTF8);
        return reader.ReadToEnd();
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return "";
    }
}

/// <summary>
/// 创建Get请求
/// </summary>
/// <param name="url">Api访问地址</param>
/// <param name="requestUrl">详细方法路径</param>
/// <param name="parameters">请求参数</param>
/// <param name="encoding">字符编码</param>
/// <param name="timout ">请求超时前等待的毫秒数,默认值是 100,000 毫秒（100 秒）</param>
/// <returns>Api返回值</returns>
public static string CreateGetHttpWebRequest(string url, string requestUrl, IDictionary<string, string> parameters, Encoding encoding, int timout)
{
    try
    {
        StringBuilder builder = new StringBuilder();
        builder.Append(url);
        builder.Append(requestUrl);
        if (parameters != null && parameters.Count >= 1)
        {
            builder.Append("?");
            int i = 0;
            foreach (var item in parameters)
            {
                if (i > 0)
                {
                    builder.Append("&");
                }
                builder.AppendFormat("{0}={1}", item.Key, item.Value);
                i++;
            }
        }
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create(builder.ToString());
        request.Method = "GET";
        request.ContentType = "application/x-www-form-urlencoded";
        request.Timeout = timout;
        StreamReader reader = new StreamReader(request.GetResponse().GetResponseStream(), encoding);
        return reader.ReadToEnd();
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return "";
    }
}
```

###### 创建Post请求
``` CSharp
/// <summary>
/// 创建Post请求
/// </summary>
/// <param name="url">Api访问地址</param>
/// <param name="requestUrl">详细方法路径</param>
/// <param name="parameters">请求参数</param>
/// <returns>Api返回值</returns>
public static string CreatePostHttpWebRequest(string url, string requestUrl, IDictionary<string, string> parameters)
{
    try
    {
        HttpWebRequest request = WebRequest.Create(url + requestUrl) as HttpWebRequest;
        request.ProtocolVersion = HttpVersion.Version10;
        request.Method = "POST";
        request.ContentType = "application/x-www-form-urlencoded";
        //如果需要POST数据
        if (!(parameters == null || parameters.Count == 0))
        {
            StringBuilder buffer = new StringBuilder();
            int i = 0;
            foreach (string key in parameters.Keys)
            {
                if (i > 0)
                {
                    buffer.AppendFormat("&{0}={1}", key, parameters[key]);
                }
                else
                {
                    buffer.AppendFormat("{0}={1}", key, parameters[key]);
                }
                i++;
            }
            byte[] data = Encoding.GetEncoding("utf-8").GetBytes(buffer.ToString());
            using (Stream stream = request.GetRequestStream())
            {
                stream.Write(data, 0, data.Length);
            }
        }
        StreamReader reader = new StreamReader(request.GetResponse().GetResponseStream(), Encoding.UTF8);
        return reader.ReadToEnd();
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return "";
    }
}

/// <summary>
/// 创建Post请求
/// </summary>
/// <param name="url">Api访问地址</param>
/// <param name="requestUrl">详细方法路径</param>
/// <param name="parameters">请求参数</param>
/// <param name="encoding">字符编码</param>
/// <param name="timout ">请求超时前等待的毫秒数,默认值是 100,000 毫秒（100 秒）</param>
/// <returns>Api返回值</returns>
public static string CreatePostHttpWebRequest(string url, string requestUrl, IDictionary<string, string> parameters, Encoding encoding, int timout)
{
    try
    {
        HttpWebRequest request = WebRequest.Create(url + requestUrl) as HttpWebRequest;
        request.ProtocolVersion = HttpVersion.Version10;
        request.Method = "POST";
        request.ContentType = "application/x-www-form-urlencoded";
        request.Timeout = timout;
        //如果需要POST数据
        if (!(parameters == null || parameters.Count == 0))
        {
            StringBuilder buffer = new StringBuilder();
            int i = 0;
            foreach (string key in parameters.Keys)
            {
                if (i > 0)
                {
                    buffer.AppendFormat("&{0}={1}", key, parameters[key]);
                }
                else
                {
                    buffer.AppendFormat("{0}={1}", key, parameters[key]);
                }
                i++;
            }
            byte[] data = encoding.GetBytes(buffer.ToString());
            using (Stream stream = request.GetRequestStream())
            {
                stream.Write(data, 0, data.Length);
            }
        }
        StreamReader reader = new StreamReader(request.GetResponse().GetResponseStream(), Encoding.UTF8);
        return reader.ReadToEnd();
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return "";
    }
}
```