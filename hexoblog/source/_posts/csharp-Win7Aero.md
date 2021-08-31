---
title: Windows 7 下无法开启 Aero 主题
date: 2021-06-02 10:39:45
tags: [c#,windows 7,aero]
categories: C#.Net
---
### 多个高分屏导致 Aero 主题无法开启
<!-- more -->
#### 简介
[Aero 主题](https://baike.baidu.com/item/Windows%20Aero/6845089?fromtitle=Aero&fromid=3554670&fr=aladdin) 仅仅是一个受人追捧的毛玻璃效果而已，但是在项目实际使用的时候却发现与 [DirectX](https://www.microsoft.com/zh-cn/download/details.aspx?id=35) 渲染效率相关，在未开启 Aero 主题的情况下叠加透明窗体 [Device.Present() 方法](https://docs.microsoft.com/en-us/previous-versions/bb324100(v=vs.85)) 延迟约在 00:00:00.1258071 相比开启了 Aero 主题的延迟约在 00:00:00.0000365，千倍的时间差。

#### 测试结果
换个主题并不是难事，但是面对一个早已不受微软支持的操作系统，对多屏幕的支持不是很好，目前也没什么解决办法，实际测试中 3 块 1080p 显示器及以上无法被动开启 Aero 主题，2 块 4k 显示器及以上无法被动开启 Aero 主题，试过多型号显卡（Quadro P1000、Quadro P2000、Quadro P4000、Gtx 1080、Rtx 2070、Rtx 4000），排除显卡性能问题，试过多版本驱动，但是不排除显卡驱动与 Windows 7 兼容不好。

#### 系统设置上的比对
##### 桌面右键菜单个性化
异常情况左下角提示：解决透明度和其他 Aero 效果问题
证明此时 Windows 7 系统是知道显示存在问题的
<img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Win7个性化设置-正常.png"/>

<img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Win7个性化设置-异常.png"/>

##### 解决透明度和其他 Aero 效果问题
点击修复程序，结果却显示很多问题不存在，无法修复
<img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Win7Aero修复程序.png"/>

<img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Win7Aero修复程序-结果.png"/>

##### 点击个性化中的窗口颜色
此时点击个性化中的窗口颜色，弹出的窗口颜色外观界面不同，异常情况无法设置 Aero 效果
<img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Win7个性化设置窗口和外观-正常.png"/>

<img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Win7个性化设置窗口和外观-异常.png"/>

##### 系统性能选项
在系统性能选项卡中异常情况缺少 启用 Aero Peek 选项与其他重要选项
<img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Win7系统性能选项-正常.png"/>

<img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Win7系统性能选项-异常.png"/>

#### 解决方案
在 Windows 7 兼容（4块1080p显示器以下或2块4k显示器以下）允许的范围内解决显示器接入断开导致 Aero 效果失效，软件操作恢复而不用人工操作。

##### 处理方式
1. 通过 dwmapi.dll 库的 DwmIsCompositionEnabled 检测 Aero 开启状态
1. 通过系统 Environment.OSVersion.Version 方法获取 Windows 版本
1. 通过 CMD 命令切换 Aero 主题和启动服务
1. 通过 System.ServiceProcess.ServiceController 重启服务
1. 涉及到的服务主要由 UxSms，或可能与 Themes也有关
    <img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/UxSms服务.png"/>

    <img src="https://sadness96.github.io/images/blog/csharp-Win7Aero/Themes服务.png"/>

##### 代码
``` csharp
/// <summary>
/// 检测 Aero 开启状态
/// </summary>
[DllImport("dwmapi.dll", PreserveSig = false)]
public static extern bool DwmIsCompositionEnabled();

/// <summary>
/// 是否为 Windows 7
/// </summary>
/// <returns></returns>
private bool IsWindows7()
{
    return Environment.OSVersion.Version.Major == 6 && Environment.OSVersion.Version.Minor == 1;
}

/// <summary>
/// 运行 Cmd 命令
/// </summary>
/// <param name="strCmdCommand"></param>
private void RunCmd(string strCmdCommand)
{
    Process cmd = new Process();
    cmd.StartInfo.FileName = "cmd";
    cmd.StartInfo.RedirectStandardInput = true;
    cmd.StartInfo.RedirectStandardOutput = true;
    cmd.StartInfo.CreateNoWindow = true;
    cmd.StartInfo.UseShellExecute = false;
    cmd.Start();
    cmd.StandardInput.WriteLine(strCmdCommand);
    cmd.StandardInput.Flush();
    cmd.StandardInput.Close();
    cmd.Close();
    cmd.Dispose();
}

/// <summary>
/// 恢复 Aero 效果
/// </summary>
/// <returns></returns>
private bool RecoveryAero()
{
    if (IsWindows7() && !DwmIsCompositionEnabled())
    {
        // 服务未启动 启动服务
        ServiceController service = new ServiceController("UxSms");
        if (service.Status == ServiceControllerStatus.Stopped)
        {
            RunCmd("net start UxSms");
        }
        // 未设置主题 修改主题
        if (!DwmIsCompositionEnabled())
        {
            string strAeroTheme = @"C:\WINDOWS\Resources\Themes\aero.theme";
            string strCmdCommand = string.Format(@"rundll32.exe %SystemRoot%\system32\shell32.dll,Control_RunDLL %SystemRoot%\system32\desk.cpl desk,@Themes /Action:OpenTheme /file:""{0}""", strAeroTheme); //cmd命令
            RunCmd(strCmdCommand);
            Thread.Sleep(5000);
        }
        // 仍无法显示 重启服务
        if (!DwmIsCompositionEnabled())
        {
            try
            {
                service.Stop();
                service.WaitForStatus(ServiceControllerStatus.Stopped);
                service.Start();
                service.WaitForStatus(ServiceControllerStatus.Running);
            }
            catch
            {

            }
        }
    }
    return DwmIsCompositionEnabled();
}
```