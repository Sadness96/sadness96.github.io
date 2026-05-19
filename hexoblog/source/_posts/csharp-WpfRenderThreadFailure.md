---
title: WPF 渲染线程崩 0x88980406
date: 2023-12-27 23:38:00
tags: [c#]
categories: C#.Net
---
### UCEERR_RENDERTHREADFAILURE (0x88980406)
<!-- more -->
### 简介
WPF 的渲染线程崩（MILCore / D3D 渲染链路异常）
目前测试仅在两台笔记本中运行 WPF 程序调用 D3D 出现这个问题，在环境完整的情况下首先怀疑的就是笔记本的集显与独显问题。

``` text
System.Runtime.InteropServices.COMException (0x88980406): UCEERR_RENDERTHREADFAILURE (0x88980406)
at System.Windows.Media.Composition.DUCE.Channel.SyncFlush()
at System.Windows.Interop.HwndTarget.UpdateWindowSettings(Boolean enableRenderTarget, Nullable`1 channelSet)
at System.Windows.Interop.HwndTarget.UpdateWindowPos(IntPtr lParam)
at System.Windows.Interop.HwndTarget.HandleMessage(WindowMessage msg, IntPtr wparam, IntPtr lparam)
at System.Windows.Interop.HwndSource.HwndTargetFilterMessage(IntPtr hwnd, Int32 msg, IntPtr wParam, IntPtr lParam, Boolean& handled)
at MS.Win32.HwndWrapper.WndProc(IntPtr hwnd, Int32 msg, IntPtr wParam, IntPtr lParam, Boolean& handled)
at MS.Win32.HwndSubclass.DispatcherCallbackOperation(Object o)
at System.Windows.Threading.ExceptionWrapper.InternalRealCall(Delegate callback, Object args, Int32 numArgs)
at System.Windows.Threading.ExceptionWrapper.TryCatchWhen(Object source, Delegate callback, Object args, Int32 numArgs, Delegate catchHandler)
```

### 解决方案
#### 集显不能被禁用
我需要在 D3D 解析 NV12，在集显禁用的情况下会导致 NV12 不被支持，不需要 NV12 解析可忽略
使用以下代码测试，需确保 ResourceType: Surface 返回结果为 True

``` csharp
static void Main(string[] args)
{
    using var d3d = new Direct3D();

    int adapterId = 0;
    var displayMode = d3d.GetAdapterDisplayMode(adapterId);
    Console.WriteLine($"当前显示模式: {displayMode.Format}, {displayMode.Width}x{displayMode.Height}");

    Format nv12Format = (Format)842094158; // D3DFMT_NV12

    Usage[] usages =
    [
        Usage.None,
        Usage.RenderTarget,
        Usage.Dynamic
    ];

    SharpDX.Direct3D9.ResourceType[] resourceTypes =
    [
        SharpDX.Direct3D9.ResourceType.Surface,
        SharpDX.Direct3D9.ResourceType.Texture
    ];

    Console.WriteLine("检查 NV12 支持情况...\n");

    foreach (var usage in usages)
    {
        foreach (var resourceType in resourceTypes)
        {
            bool supported = d3d.CheckDeviceFormat(
                adapterId,
                DeviceType.Hardware,
                displayMode.Format,
                usage,
                resourceType,
                nv12Format
            );

            Console.WriteLine($"Usage: {usage,-20} ResourceType: {resourceType,-10} Supported: {supported}");
        }
    }

    Console.WriteLine("\n检查完成。");
}
```


#### 设置首选图形处理器
在笔记本中报这个错误的主要原因。
需要在 NVIDIA 控制面板 → 管理 3D 设置 → 全局设置 → 首选图形处理器 中选择：高性能 NVIDIA 处理器。