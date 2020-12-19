---
title: 在 Release 下输出堆栈信息
date: 2020-11-25 23:20:45
tags: [c#,.net core,stack]
categories: C#.Net
---
### 使用 CallerMemberNameAttribute 类获取堆栈信息
<!-- more -->
#### 简介
在项目中需要在打印日志同时打印堆栈信息，通常使用 [StackTrace](https://docs.microsoft.com/zh-cn/dotnet/api/system.diagnostics.stacktrace) 来捕获堆栈信息，并跟随日志一同打印，但是生产环境部署通常使用 Release 方式打包，这会导致 StackTrace 方法失效。测试改用 [CallerMemberNameAttribute](https://docs.microsoft.com/zh-cn/dotnet/api/System.Runtime.CompilerServices.CallerMemberNameAttribute) 方式可以解决。

#### 代码
##### StackTrace 方式获取堆栈信息
``` CSharp
public static void SaveError(string message)
{
    var stackTrace = new StackTrace(true);
    var stackFrame = stackTrace.GetFrame(1);
    if (stackFrame != null)
    {
        logger.Error($"[{stackFrame?.GetMethod()?.DeclaringType?.Name}][{stackFrame?.GetMethod()?.Name}] {message}");
    }
}
```

##### CallerMemberNameAttribute 改用方式获取堆栈信息
``` CSharp
public static void SaveError(string message,
[System.Runtime.CompilerServices.CallerMemberName] string memberName = "",
[System.Runtime.CompilerServices.CallerFilePath] string sourceFilePath = "",
[System.Runtime.CompilerServices.CallerLineNumber] int sourceLineNumber = 0)
{
    logger.Error($"[{System.IO.Path.GetFileName(sourceFilePath)}][{memberName}] {message}");
}
```