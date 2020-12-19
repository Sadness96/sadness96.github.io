---
title: 使用 CEF 在 PC 客户端中加载网页
date: 2020-8-11 13:03:21
tags: [c#,wpf,cef]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-CefSharp/CEFLogo.png"/>

### 在 WPF 中调用 CEF 加载网页
<!-- more -->
#### 简介
最近有需求在 wpf 中加载网页，尝试使用 CEF 加载 Chrome 内核浏览器显示。

#### 引用库介绍
NuGet 中引用 [CefSharp.Wpf](https://github.com/cefsharp/cefsharp)

#### 设置
需要在项目 .csproj 文件中增加代码
``` XML
<ItemGroup>
    <Reference Update="CefSharp">
        <Private>true</Private>
    </Reference>
    <Reference Update="CefSharp.Core">
        <Private>true</Private>
    </Reference>
    <Reference Update="CefSharp.Wpf">
        <Private>true</Private>
    </Reference>
</ItemGroup>
```

#### 使用
.xaml 文件中 增加引用和样式
``` XML
xmlns:chrome="clr-namespace:CefSharp.Wpf;assembly=CefSharp.Wpf"
```
``` XML
<Grid>
    <chrome:ChromiumWebBrowser x:Name="Browser"/>
</Grid>
```
.xaml.cs 文件中增加代码
``` CSharp
Browser.Address = @"https://www.baidu.com/";
```
即可显示
<img src="https://sadness96.github.io/images/blog/csharp-CefSharp/CEFDemo.png"/>

#### CefSharp 使用优化
##### 屏蔽或修改右键菜单
``` csharp
using CefSharp;

namespace Ice.CefControl.Handler
{
    /// <summary>
    /// 关联 Cef 右键菜单
    /// </summary>
    public class RightMenuHandler : IContextMenuHandler
    {
        public void OnBeforeContextMenu(IWebBrowser chromiumWebBrowser, IBrowser browser, IFrame frame, IContextMenuParams parameters, IMenuModel model)
        {
            //主要修改代码在此处;如果需要完完全全重新添加菜单项,首先执行model.Clear()清空菜单列表即可.
            //需要自定义菜单项的,可以在这里添加按钮;
            if (model.Count > 0)
            {
                model.AddSeparator();//添加分隔符;
            }

            //清理所有右键菜单
            model.Clear();

            //打开调试
            //model.AddItem((CefMenuCommand)26501, "Show DevTools");
            //model.AddItem((CefMenuCommand)26502, "Close DevTools");
        }

        public bool OnContextMenuCommand(IWebBrowser chromiumWebBrowser, IBrowser browser, IFrame frame, IContextMenuParams parameters, CefMenuCommand commandId, CefEventFlags eventFlags)
        {
            //命令的执行,点击菜单做什么事写在这里.
            if (commandId == (CefMenuCommand)26501)
            {
                browser.GetHost().ShowDevTools();
                return true;
            }
            if (commandId == (CefMenuCommand)26502)
            {
                browser.GetHost().CloseDevTools();
                return true;
            }
            return false;
        }

        public void OnContextMenuDismissed(IWebBrowser chromiumWebBrowser, IBrowser browser, IFrame frame)
        {

        }

        public bool RunContextMenu(IWebBrowser chromiumWebBrowser, IBrowser browser, IFrame frame, IContextMenuParams parameters, IMenuModel model, IRunContextMenuCallback callback)
        {
            //return false 才可以弹出
            return false;
        }
    }
}
```

调用时引用

``` csharp
// 关联右键菜单
Browser.MenuHandler = new RightMenuHandler();
```

##### 捕获快捷键(调试使用)
``` csharp
using CefSharp;
using System;
using System.Windows.Forms;

namespace Ice.CefControl.Handler
{
    /// <summary>
    /// 捕获 Cef 快捷键
    /// </summary>
    public class KeyBoardHandler : IKeyboardHandler
    {
        public bool OnKeyEvent(IWebBrowser browserControl, IBrowser browser, KeyType type, int windowsKeyCode, int nativeKeyCode, CefEventFlags modifiers, bool isSystemKey)
        {
            if (type == KeyType.KeyUp && Enum.IsDefined(typeof(Keys), windowsKeyCode))
            {
                var key = (Keys)windowsKeyCode;
                switch (key)
                {
                    case Keys.F12:
                        browser.ShowDevTools();
                        break;
                    case Keys.F5:
                        if (modifiers == CefEventFlags.ControlDown)
                        {
                            //MessageBox.Show("ctrl+f5");
                            browser.Reload(true);
                        }
                        else
                        {
                            //MessageBox.Show("f5");
                            browser.Reload();
                        }
                        break;
                }
            }
            return false;
        }

        public bool OnPreKeyEvent(IWebBrowser browserControl, IBrowser browser, KeyType type, int windowsKeyCode, int nativeKeyCode, CefEventFlags modifiers, bool isSystemKey, ref bool isKeyboardShortcut)
        {
            return false;
        }
    }
}
```

调用时引用

``` csharp
#if DEBUG
    // 关联快捷键
    Browser.KeyboardHandler = new KeyBoardHandler();
#endif
```

##### CefSharp 以 Any CPU 平台编译并且使文件生成在子目录
###### 参考文档
[Add AnyCPU Support](https://github.com/cefsharp/CefSharp/issues/1714)
[Copy CefSharp Files](https://github.com/cefsharp/CefSharp/pull/1753)
###### 代码部分
编辑项目文件 project.csproj 加入以下内容
``` xml
<Project Sdk="Microsoft.NET.Sdk.WindowsDesktop">

  <PropertyGroup>
    <!--允许在 Any CPU 平台下允许编译-->
    <CefSharpAnyCpuSupport>true</CefSharpAnyCpuSupport>
    <!--拷贝 CefSharp 相关文件至 \CefSharp 子文件夹-->
    <!--x86 与 x64 平台下编译会生成在 \CefSharp 目录下-->
    <!--Any CPU 平台下编译会在 \CefSharp 下生成 \x86 与 \x64 子文件夹-->
    <CefSharpTargetDir>\CefSharp</CefSharpTargetDir>
  </PropertyGroup>

</Project>
```
在运行初始时执行代码 WPF 为：App.xaml.cs 文件
``` csharp
using CefSharp;
using CefSharp.Wpf;
using System;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Windows;

namespace ProjectName
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            // Add Custom assembly resolver
            AppDomain.CurrentDomain.AssemblyResolve += Resolver;

            // Any CefSharp references have to be in another method with NonInlining
            // attribute so the assembly rolver has time to do it's thing.
            InitializeCefSharp();

            // 启动主程序
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void InitializeCefSharp()
        {
            var settings = new CefSettings();

            // 不对日志进行保存
            settings.LogSeverity = LogSeverity.Disable;

            // Set BrowserSubProcessPath based on app bitness at runtime
            settings.BrowserSubprocessPath = GetCefSharpFilePath("CefSharp.BrowserSubprocess.exe");

            // Make sure you set performDependencyCheck false
            Cef.Initialize(settings, performDependencyCheck: false, browserProcessHandler: null);
        }

        // Will attempt to load missing assembly from either x86 or x64 subdir
        // Required by CefSharp to load the unmanaged dependencies when running using AnyCPU
        private static Assembly Resolver(object sender, ResolveEventArgs args)
        {
            if (args.Name.StartsWith("CefSharp"))
            {
                string assemblyName = args.Name.Split(new[] { ',' }, 2)[0] + ".dll";
                string archSpecificPath = GetCefSharpFilePath(assemblyName);
                return File.Exists(archSpecificPath) ? Assembly.LoadFile(archSpecificPath) : null;
            }
            return null;
        }

        /// <summary>
        /// 获取 CefSharp 文件路径
        /// </summary>
        /// <param name="assemblyName">文件名称</param>
        /// <returns></returns>
        private static string GetCefSharpFilePath(string assemblyName)
        {
            var vAnyCpuPath = Path.Combine(AppDomain.CurrentDomain.SetupInformation.ApplicationBase, "CefSharp", Environment.Is64BitProcess ? "x64" : "x86", assemblyName);
            var vNoAnyCpuPath = Path.Combine(AppDomain.CurrentDomain.SetupInformation.ApplicationBase, "CefSharp", assemblyName);
            return Directory.Exists(Path.GetDirectoryName(vAnyCpuPath)) ? vAnyCpuPath : vNoAnyCpuPath;
        }
    }
}
```

##### CefSharp 报错：试图加载格式不正确的程序。
默认 CefSharp 仅允许在设置为 x86 或 x64 平台下运行，修改设置即可。
但是有时会在配置了允许 Any CPU 后出现 x86 平台下正常 x64 平台下运行报同样错误，最后找到问题出现在独立创建用的于调用 Cef 库不知何时生成出一些不必要的内容，删除 project.csproj 文件下不必要的内容即可。
``` xml
<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|AnyCPU'">
  <PlatformTarget>x86</PlatformTarget>
</PropertyGroup>

<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x86'">
  <PlatformTarget>x86</PlatformTarget>
</PropertyGroup>

<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
  <PlatformTarget>x86</PlatformTarget>
</PropertyGroup>
```