---
title: WPF Popup 与 ToolTip 显示位置异常
date: 2022-10-10 22:54:05
tags: [c#]
categories: C#.Net
---
### 由于平板电脑设置下的惯用手设置导致的显示位置异常
<!-- more -->
### 简介
WPF 中 [Popup](https://learn.microsoft.com/en-us/dotnet/desktop/wpf/controls/popup?view=netframeworkdesktop-4.8) 与 [ToolTip](https://learn.microsoft.com/en-us/dotnet/desktop/wpf/controls/tooltip?view=netframeworkdesktop-4.8) 控件会受到 Windows 系统设置中 平板电脑设置->惯用手 设置而显示异常，例如设置惯用右手时弹窗显示在左侧，设置惯用左手时弹窗显示在右侧。
#### 惯用右手
<img src="https://sadness96.github.io/images/blog/csharp-Handedness/惯用右手1.jpg"/>
<img src="https://sadness96.github.io/images/blog/csharp-Handedness/惯用右手2.jpg"/>

#### 惯用左手
<img src="https://sadness96.github.io/images/blog/csharp-Handedness/惯用左手1.jpg"/>
<img src="https://sadness96.github.io/images/blog/csharp-Handedness/惯用左手2.jpg"/>

### 设置惯用手
1. 运行中输入：shell:::{80F3F1D5-FECA-45F3-BC32-752C152E456E} 打开平板电脑设置
<img src="https://sadness96.github.io/images/blog/csharp-Handedness/运行.jpg"/>

1. 在其他中可以设置左右手使用习惯（一般情况下系统默认惯用左手）
<img src="https://sadness96.github.io/images/blog/csharp-Handedness/平板电脑设置.jpg"/>

### 代码修改
#### 代码修改系统为惯用左手（不建议）
参考 [SystemParametersInfoA](https://learn.microsoft.com/zh-cn/windows/win32/api/winuser/nf-winuser-systemparametersinfoa?redirectedfrom=MSDN) 函数设置，设置为惯用左手，不过既然是设置系统，免不了会与其他软件冲突。
``` CSharp
public MainWindow()
{
    InitializeComponent();
    // 设置对齐方式
    SystemParametersInfoSet(0x001C /*SPI_SETMENUDROPALIGNMENT*/, 0, 0, 0);
}

[DllImport("user32.dll", EntryPoint = "SystemParametersInfo", SetLastError = true)]
public static extern bool SystemParametersInfoSet(uint action, uint uiParam, uint vparam, uint init);
```

#### 代码修改临时为惯用左手（推荐）
使用代码修改临时为惯用左手，仅对当前运行有效，不修改系统设置。
``` CSharp
public MainWindow()
{
    InitializeComponent();
    // 设置对齐方式
    SetAlignment();
}

/// <summary>
/// 设置对齐方式
/// 设置为惯用左手 菜单出现在手的右侧
/// </summary>
public static void SetAlignment() 
{
    //获取系统是以Left-handed（true）还是Right-handed（false）
    var ifLeft = SystemParameters.MenuDropAlignment;

    if (ifLeft)
    {
        // change to false
        var t = typeof(SystemParameters);
        var field = t.GetField("_menuDropAlignment", BindingFlags.NonPublic | BindingFlags.Static);
        field.SetValue(null, false);
    }
}
```