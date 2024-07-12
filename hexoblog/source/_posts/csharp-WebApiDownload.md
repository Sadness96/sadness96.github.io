---
title: WebApi 下载文件
date: 2019-04-19 10:59:20
tags: [c#,helper,webapi]
categories: C#.Net
---
### WebApi 文件下载服务端与客户端
<!-- more -->
### 简介
原帮助类代码参考 [WebApi 帮助类](https://sadness96.github.io/blog/2018/08/27/csharp-WebApiHelper/)，新增下载文件方法，使用 WPF 调用下载并回调显示下载进度。显示下载文件大小以及下载速度。

### 代码
#### 服务端代码
服务端代码因项目而异，这段代码仅为示例。
``` CSharp
/// <summary>
/// 下载文件
/// </summary>
/// <param name="FileName">文件名</param>
/// <returns></returns>
[HttpGet]
public ActionResult DownloadFile(string FileName)
{
    var vFileCompletePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, FileName);
    if (!System.IO.File.Exists(vFileCompletePath))
    {
        return Unauthorized("File does not exist!");
    }

    var vStream = System.IO.File.OpenRead(vFileCompletePath);
    var vProvider = new FileExtensionContentTypeProvider();
    if (!vProvider.TryGetContentType(vFileCompletePath, out string? contentType))
    {
        contentType = "application/octet-stream";
    }
    return File(vStream, contentType);
}
```

#### 帮助类代码
``` CSharp
/// <summary>
/// 文件下载
/// </summary>
/// <param name="baseAddress">Api访问地址</param>
/// <param name="requestUrl">请求地址</param>
/// <param name="parameters">请求参数</param>
/// <param name="saveFilePath">保存文件路径</param>
/// <param name="progressAction">进度回调</param>
/// <param name="token">Token认证</param>
/// <returns>是否下载成功</returns>
public static async Task<bool> DownloadGetAsync(Uri baseAddress, string? requestUrl, IDictionary<string, object>? parameters, string saveFilePath, Action<object?, HttpProgressEventArgs>? progressAction = null, string? token = null)
{
    ProgressMessageHandler progress = new ProgressMessageHandler();
    progress.HttpReceiveProgress += (s, e) =>
    {
        progressAction?.Invoke(s, e);
    };

    var vHttpClient = HttpClientFactory.Create(progress);
    vHttpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

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
        if (parameters != null && parameters.Count >= 1)
        {
            vBuilder.Append('?');
            int i = 0;
            foreach (var item in parameters)
            {
                if (i > 0)
                {
                    vBuilder.Append('&');
                }
                vBuilder.AppendFormat("{0}={1}", item.Key, item.Value);
                i++;
            }
        }

        using var vResponse = await vHttpClient.GetAsync(vBuilder.ToString());
        if (vResponse.IsSuccessStatusCode)
        {
            //保存文件
            using var vFs = File.Create(saveFilePath);
            var vStream = await vResponse.Content.ReadAsStreamAsync();
            vStream.CopyTo(vFs);
            vStream.Close();
            vStream.Dispose();
            vFs.Close();
            vFs.Dispose();
            return true;
        }
        else
        {
            return false;
        }
    }
    catch (Exception)
    {
        return false;
    }
    finally
    {
        vHttpClient.Dispose();
        vHttpClient = null;
        GC.Collect();
    }
}
```

#### 进度条样式
``` CSharp
<Grid>
    <ProgressBar x:Name="progressBar" HorizontalAlignment="Left" Height="13" Margin="10,10,0,0" VerticalAlignment="Top" Width="100" Maximum="1" Minimum="0"/>
    <Label x:Name="number" HorizontalAlignment="Left" Margin="10,28,0,0" VerticalAlignment="Top" Width="319" Height="41"/>
    <Label x:Name="number2" HorizontalAlignment="Left" Margin="10,58,0,0" VerticalAlignment="Top" Width="319" Height="41"/>
</Grid>
```

#### 调用下载
``` CSharp
private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
{
    // 基于服务端示例代码的参数
    string serverUrl = "https://localhost:7252/";
    string requestUrl = "api/File/DownloadFile";
    string savePath = @"D:\1.png";

    var vParameters = new Dictionary<string, object>
    {
        { "FileName", "1.png" }
    };
    var vDownload = await HttpClientHelper.DownloadGetAsync(new Uri(serverUrl), requestUrl, vParameters, savePath, DownloadProgressChanged);
    MessageBox.Show($"保存{(vDownload ? "成功" : "失败")}");
}

/// <summary>
/// 下载进度发生改变
/// 更新界面下载进度
/// </summary>
/// <param name="arg1"></param>
/// <param name="arg2"></param>
private void DownloadProgressChanged(object? arg1, HttpProgressEventArgs arg2)
{
    this.Dispatcher.Invoke(new Action(() =>
    {
        if (arg2.ProgressPercentage >= 0 && arg2.ProgressPercentage <= 100)
        {
            // 显示进度条
            progressBar.Value = arg2.ProgressPercentage / 100.0;
            // 显示下载文件大小
            number.Content = $"{ByteChange(arg2.BytesTransferred)}/{ByteChange(arg2.TotalBytes)}";
            // 显示计算下载速度
            CalcDownloadSpeed(DateTime.Now, arg2.BytesTransferred);
        }
    }));
}

private DateTime lastTime = DateTime.Now;
private long BytesTransferred = 0;

/// <summary>
/// 计算下载速度
/// </summary>
/// <param name="time"></param>
/// <param name="bytesTransferred"></param>
private void CalcDownloadSpeed(DateTime time, long bytesTransferred)
{
    if (lastTime.Second != time.Second)
    {
        this.Dispatcher.Invoke(new Action(() =>
        {
            number2.Content = $"Speed: {ByteChange(bytesTransferred - BytesTransferred)}/s";
        }));

        lastTime = time;
        BytesTransferred = bytesTransferred;
    }
}

/// <summary>
/// 转换下载单位
/// </summary>
/// <param name="TotalBytes">数据大小比特</param>
/// <returns></returns>
private string ByteChange(long? TotalBytes)
{
    long kb = 1024;
    long mb = kb * kb;
    long gb = mb * kb;
    long tb = gb * kb;

    if (TotalBytes >= tb)
    {
        return $"{Math.Round((decimal)TotalBytes / tb, 2)}TB";
    }
    else if (TotalBytes >= gb)
    {
        return $"{Math.Round((decimal)TotalBytes / gb, 2)}GB";
    }
    else if (TotalBytes >= mb)
    {
        return $"{Math.Round((decimal)TotalBytes / mb, 2)}MB";
    }
    else if (TotalBytes >= kb)
    {
        return $"{Math.Round((decimal)TotalBytes / kb, 2)}KB";
    }
    else
    {
        return $"{TotalBytes}B";
    }
}
```