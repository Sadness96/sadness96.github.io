---
title: Windows 应用程序启动时打开控制台
date: 2018-06-12 14:43:45
tags: [c#,console]
categories: C#.Net
---
### 程序启动时附加控制台调试
<!-- more -->
#### 简介
在开发大型项目时会经常有数据传输，在测试时又不能及时显示出来，为了方便调试，可在运行 Windows 应用程序时同时运行控制台，打印测试信息用于调试。

#### 代码
##### 工具类
``` CSharp
public class ConsoleWindow
{
    [DllImport("kernel32.dll", EntryPoint = "AllocConsole")]
    public static extern bool Show();

    [DllImport("kernel32.dll", EntryPoint = "FreeConsole")]
    public static extern bool Close();
}
```

##### 调用
可以选择仅在 Debug 时运行，一般运行于完整生命周期，不太用得上 Close。
``` CSharp
#if DEBUG
    // 运行控制台程序
    ConsoleWindow.Show();
#endif
```