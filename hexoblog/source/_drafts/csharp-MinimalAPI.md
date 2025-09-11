---
title: ASP.NET Core Minimal API
date: 2025-07-18 04:42:15
tags: [c#,.net core,api]
categories: C#.Net
---
### 使用 Minimal API 替代 Owin
<!-- more -->
### 简介
在 .NET Framework 中或者 .NET Core 6 以前，在控制台或其他桌面应用程序中使用 WebApi，可以使用 [Owin](https://learn.microsoft.com/zh-cn/aspnet/web-api/overview/hosting-aspnet-web-api/use-owin-to-self-host-web-api) 来创建，但是在 .NET Core 6 以后的版本 Owin 则不再被支持，而是引入了轻量级的开发模式 [Minimal API](https://learn.microsoft.com/zh-cn/aspnet/core/fundamentals/minimal-apis?view=aspnetcore-9.0)

### 代码
#### Owin
参考 [使用 OWIN Self-Host ASP.NET Web API](https://learn.microsoft.com/zh-cn/aspnet/web-api/overview/hosting-aspnet-web-api/use-owin-to-self-host-web-api)
* NuGet 引用官方 Microsoft.AspNet.WebApi.OwinSelfHost 库。

``` CSharp
class Program
{
    static void Main()
    {
        const string url = "http://localhost:5050";

        using (WebApp.Start<Startup>(url))
        {
            Console.WriteLine($"服务已启动，访问 {url}/api/test");
            Console.ReadLine();
        }
    }
}

public class Startup
{
    public void Configuration(IAppBuilder app)
    {
        var config = new HttpConfiguration();
        config.Routes.MapHttpRoute(
            "DefaultApi",
            "api/{controller}/{id}",
            new { id = RouteParameter.Optional });
        app.UseWebApi(config);
    }
}

public class TestController : ApiController
{
    public string Get() => "Hello from OWIN Self-Host!";
}
```

#### Minimal API
参考 [最小 API 快速参考](https://learn.microsoft.com/zh-cn/aspnet/core/fundamentals/minimal-apis?view=aspnetcore-9.0)
* NuGet 引用官方 Microsoft.AspNet.WebApi.Client 库。
* NuGet 引用官方 Microsoft.AspNetCore.Authentication.JwtBearer 库。

``` CSharp
internal class Program
{
    static void Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(args);

        // 添加 MVC 服务
        builder.Services.AddControllers();

        // 日志只记录 Warning 及以上
        builder.Logging.SetMinimumLevel(LogLevel.Warning);

        builder.WebHost.UseUrls("http://localhost:5050");
        var app = builder.Build();

        // 可以选择使用极简的方式 URL 路径与处理逻辑绑定
        // app.MapGet("/", () => "Hello from Minimal API!");

        // 映射 Controller 路由
        app.MapControllers();

        // 启动 API
        var apiTask = app.RunAsync();
        Console.ReadLine();
    }
}

[Route("api/[controller]/[action]")]
[ApiController]
public class TestController : ControllerBase
{
    [HttpGet]
    public ActionResult<string> GetTest()
    {
        return Ok("Hello from Minimal API!");
    }
}
```