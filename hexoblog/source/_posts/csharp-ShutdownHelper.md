---
title: Windows关机帮助类
date: 2017-06-06 17:48:02
tags: [c#,helper,windows,shutdown]
categories: C#.Net
---
<img src="http://www.bkill.com/u/upload/2017/08/17/172215508193.jpg"/>
### 关闭Windows计算机帮助类…鬼知道为什么我关电脑也能整理出一篇文章。
<!-- more -->
#### 简介
起初在刚接触C#时制作 [Desktop Lock](/blog/2016/05/31/csharp-DesktopLock/) 时有设置自动关机的功能，使用的是执行 [CMD](https://baike.baidu.com/item/%E5%91%BD%E4%BB%A4%E6%8F%90%E7%A4%BA%E7%AC%A6/998728?fromtitle=CMD&fromid=1193011&fr=aladdin) 的 [shutdown](https://baike.baidu.com/item/shutdown) 命令，但是极容易出现被杀毒软件误报毒或误被取消关机。然后又通过调用 [Win32 API](https://baike.baidu.com/item/Win32%20API) 的方式关闭计算机。
#### 帮助类
[ShutdownHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Shutdown/ShutdownHelper.cs)
##### CMD Shutdown
``` CSharp
public void shutdown()
{
    Process myProcess = new Process();
    myProcess.StartInfo.FileName = "cmd.exe";
    myProcess.StartInfo.UseShellExecute = false;
    myProcess.StartInfo.RedirectStandardInput = true;
    myProcess.StartInfo.RedirectStandardOutput = true;
    myProcess.StartInfo.RedirectStandardError = true;
    myProcess.StartInfo.CreateNoWindow = true;
    myProcess.Start();
    myProcess.StandardInput.WriteLine("shutdown -s -f -t 0");
}
```
##### Win32 API
``` CSharp
[StructLayout(LayoutKind.Sequential, Pack = 1)]
internal struct TokPriv1Luid
{ public int Count; public long Luid; public int Attr;}
[DllImport("kernel32.dll", ExactSpelling = true)]
internal static extern IntPtr GetCurrentProcess();
[DllImport("advapi32.dll", ExactSpelling = true, SetLastError = true)]
internal static extern bool OpenProcessToken(IntPtr h, int acc, ref IntPtr phtok);
[DllImport("advapi32.dll", SetLastError = true)]
internal static extern bool LookupPrivilegeValue(string host, string name, ref long pluid);
[DllImport("advapi32.dll", ExactSpelling = true, SetLastError = true)]
internal static extern bool AdjustTokenPrivileges(IntPtr htok, bool disall, ref TokPriv1Luid newst, int len, IntPtr prev, IntPtr relen);
[DllImport("user32.dll", ExactSpelling = true, SetLastError = true)]
internal static extern bool ExitWindowsEx(int flg, int rea);
internal const int SE_PRIVILEGE_ENABLED = 0x00000002;
internal const int TOKEN_QUERY = 0x00000008;
internal const int TOKEN_ADJUST_PRIVILEGES = 0x00000020;
internal const string SE_SHUTDOWN_NAME = "SeShutdownPrivilege";
internal const int EWX_LOGOFF = 0x00000000;
internal const int EWX_SHUTDOWN = 0x00000001;
internal const int EWX_REBOOT = 0x00000002;
internal const int EWX_FORCE = 0x00000004;
internal const int EWX_POWEROFF = 0x00000008;
internal const int EWX_FORCEIFHUNG = 0x00000010;

/// <summary>
/// 关闭Windows
/// </summary>
/// <param name="flg"></param>
private static void DoExitWin(int flg)
{
    bool ok;
    TokPriv1Luid tp;
    IntPtr hproc = GetCurrentProcess();
    IntPtr htok = IntPtr.Zero;
    ok = OpenProcessToken(hproc, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ref htok);
    tp.Count = 1; tp.Luid = 0; tp.Attr = SE_PRIVILEGE_ENABLED;
    ok = LookupPrivilegeValue(null, SE_SHUTDOWN_NAME, ref tp.Luid);
    ok = AdjustTokenPrivileges(htok, false, ref tp, 0, IntPtr.Zero, IntPtr.Zero);
    ok = ExitWindowsEx(flg, 0);
}

/// <summary>
/// 关闭计算机
/// </summary>
public static void Shutdown()
{
    try
    {
        DoExitWin(EWX_SHUTDOWN);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
    }
}

/// <summary>
/// 注销计算机
/// </summary>
public static void Logoff()
{
    try
    {
        DoExitWin(EWX_LOGOFF);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
    }
}

/// <summary>
/// 重启计算机
/// </summary>
public static void Reboot()
{
    try
    {
        DoExitWin(EWX_REBOOT);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
    }
}
```