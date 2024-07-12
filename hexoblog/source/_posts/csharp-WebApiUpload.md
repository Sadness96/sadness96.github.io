---
title: WebApi 上传文件
date: 2024-06-15 18:50:35
tags: [c#,helper,webapi]
categories: C#.Net
---
### 基于表单的 WebApi 文件上传服务端与客户端
<!-- more -->
### 简介
之前有写过 [HttpClient 上传文件](https://sadness96.github.io/blog/2022/05/20/csharp-WebApiFormData/)，只是作为客户端使用，现在提供服务端演示，封装完善函数。
WebApi 下载文件参考：[WebApi 下载文件](https://sadness96.github.io/blog/2019/04/19/csharp-WebApiDownload/)。

### 代码
#### 服务端代码
服务端代码因项目而异，这段代码仅为示例。
``` CSharp
/// <summary>
/// 上传文件
/// </summary>
/// <param name="formData">表单数据</param>
/// <returns></returns>
[HttpPost]
public async Task<ActionResult> UploadFile([FromForm] IFormCollection formData)
{
    try
    {
        // 打印 Text
        var vKeys = formData.Keys;
        foreach (var key in vKeys)
        {
            Console.WriteLine($"Text {key}:{formData[key]}");
        }

        // 下载 File
        var vFiles = formData.Files;
        foreach (var file in vFiles)
        {
            var vFileCompletePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, file.FileName);
            Console.WriteLine($"File {file.Name}:{file.FileName}");
            using var stream = new FileStream(vFileCompletePath, FileMode.Create);
            await file.CopyToAsync(stream);
        }

        return Ok();
    }
    catch (Exception ex)
    {
        return BadRequest($"Execute error! {ex.Message}");
    }
}
```

#### 帮助类代码
``` CSharp
/// <summary>
/// 文件上传
/// </summary>
/// <param name="baseAddress">Api访问地址</param>
/// <param name="requestUrl">请求地址</param>
/// <param name="parameters">请求参数</param>
/// <param name="parameterFiles">请求文件参数</param>
/// <param name="progressAction">进度回调</param>
/// <param name="token">Token认证</param>
/// <returns></returns>
public static async Task<T?> UploadPostAsync<T>(Uri baseAddress, string? requestUrl, IDictionary<string, object>? parameters, IDictionary<string, string>? parameterFiles, Action<object?, HttpProgressEventArgs>? progressAction = null, string? token = null)
{
    ProgressMessageHandler progress = new ProgressMessageHandler();
    progress.HttpSendProgress += (s, e) =>
    {
        progressAction?.Invoke(s, e);
    };

    var vHttpClient = HttpClientFactory.Create(progress);
    vHttpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

    // 缓存文件流，finally 时释放
    List<FileStream> files = new();
    List<StreamContent> streams = new();

    try
    {
        if (!string.IsNullOrEmpty(token))
        {
            vHttpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }
        vHttpClient.Timeout = TimeSpan.FromMinutes(20);

        //拼接参数
        var vBuilder = new StringBuilder();
        vBuilder.Append(baseAddress);
        vBuilder.Append(requestUrl);

        // 创建使用 multipart/form-data MIME 类型进行编码的内容提供容器。
        var vMultipartForm = new MultipartFormDataContent();
        // 添加文本参数
        if (parameters != null && parameters.Count >= 1)
        {
            foreach (var parameter in parameters)
            {
                vMultipartForm.Add(new StringContent(parameter.Value?.ToString() ?? string.Empty), parameter.Key);
            }
        }

        // 添加文件参数
        if (parameterFiles != null && parameterFiles.Count >= 1)
        {
            foreach (var parameterFile in parameterFiles)
            {
                string filePath = parameterFile.Value;
                var fileName = Path.GetFileName(filePath);

                var fileStream = File.OpenRead(filePath);
                files.Add(fileStream);
                var streamContent = new StreamContent(fileStream);
                streams.Add(streamContent);
                vMultipartForm.Add(streamContent, parameterFile.Key, fileName);
            }
        }

        using var vResponse = await vHttpClient.PostAsync(vBuilder.ToString(), vMultipartForm);
        if (vResponse.StatusCode == HttpStatusCode.OK)
        {
            return typeof(T).FullName.Equals("System.String") || typeof(T).FullName.Equals("System.Boolean")
                ? (T)Convert.ChangeType(await vResponse.Content.ReadAsStringAsync(), typeof(T))
                : JsonConvert.DeserializeObject<T>(await vResponse.Content.ReadAsStringAsync());
        }
        else
        {
            return default;
        }
    }
    catch (Exception)
    {
        return default;
    }
    finally
    {
        foreach (var item in files)
        {
            item.Dispose();
        }
        files.Clear();

        foreach (var item in streams)
        {
            item.Dispose();
        }
        streams.Clear();

        vHttpClient.Dispose();
        vHttpClient = null;
        GC.Collect();
    }
}
```

#### 调用上传
``` CSharp
// 基于服务端示例代码的参数
string serverUrl = "https://localhost:7252/";
string requestUrl = "api/File/UploadFile";

var vParameters = new Dictionary<string, object>
{
    { "id", 1 },
    { "name", "test" }
};
var vParameterFiles = new Dictionary<string, string>
{
    { "file1", @"1.png" },
    { "file2", @"2.png" }
};
return await HttpClientHelper.UploadPostAsync<bool>(new Uri(serverUrl), requestUrl, vParameters, vParameterFiles, null, null);
```

### 注意事项
#### 报错请求正文太大，最大请求正文大小为30000000字节
``` cmd
// 错误信息
Failed to read the request form. Request body too large. The max request body size is 30000000 bytes. 
```

需要修改服务端 WebApi。
1. 通过在 API 函数上增加属性实现，以下二选一
    ``` CSharp
    // 禁用请求大小限制
    [DisableRequestSizeLimit]
    // 请求大小限制(例如：200MB)
    [RequestSizeLimit(200_000_000)]
    ```
1. 通过在 Program.cs 中添加全局配置（Net6）
    ``` CSharp
    // Configure Kestrel server options
    builder.WebHost.ConfigureKestrel((context, options) =>
    {
        // For example, 200 MB limit
        options.Limits.MaxRequestBodySize = 200_000_000;
    });
    ```

#### 报错超过了正文长度限制 134217728
``` cmd
// 错误信息
Failed to read the request form. Multipart body length limit 134217728 exceeded.
```

需要修改服务端 WebApi。
1. 通过在 API 函数上增加属性实现
    ``` CSharp
    // 设置最大的 multipart/form-data 请求体长度为 1 GB
    [RequestFormLimits(MultipartBodyLengthLimit = 1_000_000_000)]
    ```
1. 通过在 Program.cs 中添加全局配置（Net6）
    ``` CSharp
    // Configure FormOptions for multipart form limits
    builder.Services.Configure<FormOptions>(options =>
    {
        // For example, 1 GB limit
        options.MultipartBodyLengthLimit = 1_000_000_000;
    });
    ```

#### 经过以上两项修改，可能还会在 API 接口中抛异常超过了主体长度限制 16384
``` cmd
// 错误信息
System.IO.InvalidDataException:“Multipart body length limit 16384 exceeded.”
```

需要修改服务端 WebApi。
``` CSharp
public ActionResult<bool> UploadFile()
{
    // 直接使用 Request.Form 会抛异常
    var vFormData = Request.Form;
}

public ActionResult<bool> UploadFile([FromForm] IFormCollection formData)
{
    // 改为使用参数传入
    var vFormData = formData;
}
```