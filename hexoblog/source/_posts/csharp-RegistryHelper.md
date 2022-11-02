---
title: Windows注册表帮助类
date: 2017-06-22 19:50:08
tags: [c#,helper,windows,registry]
categories: C#.Net
---
### 基于 Microsoft.Win32 库操作Windows注册表帮助类
<!-- more -->
#### 简介
[注册表（Registry）](https://baike.baidu.com/item/%E6%B3%A8%E5%86%8C%E8%A1%A8/101856?fr=aladdin) 作为Windows操作系统中的一个核心数据库，用于存储系统和应用程序的设置信息。修改常见的功能有：软件启动项、系统级菜单、文件默认启动程序及默认图标。常见的修改方式有Windows自带的命令regedit进入图形界面修改，或者熟悉[批处理脚本（.bat）](https://baike.baidu.com/item/%E6%89%B9%E5%A4%84%E7%90%86/1448600?fromtitle=.bat&fromid=6476412&fr=aladdin)的朋友可以更方便的修改。
#### 帮助类
[RegistryHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Registry/RegistryHelper.cs) 引用 [Microsoft.Win32](https://docs.microsoft.com/zh-cn/dotnet/api/microsoft.win32?redirectedfrom=MSDN&view=netframework-4.8) 库。
##### 设置软件启动项
``` CSharp
#region Registry Startup Items
/// <summary>
/// 创建注册表启动项
/// </summary>
/// <param name="strName">键值名称</param>
/// <param name="strSoftwarePath">启动项软件路径</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CreateStartupItems(string strName, string strSoftwarePath)
{
    try
    {
        if (string.IsNullOrEmpty(strName) || string.IsNullOrEmpty(strSoftwarePath))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.CurrentUser.OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.CurrentUser.CreateSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Run");
        }
        registryKey.SetValue(strName, strSoftwarePath);
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 删除注册表启动项
/// </summary>
/// <param name="strName">键值名称</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeleteStartupItems(string strName)
{
    try
    {
        if (string.IsNullOrEmpty(strName))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.CurrentUser.OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", true);
        if (registryKey == null)
        {
            return false;
        }
        registryKey.DeleteValue(strName, false);
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 获得注册表中所有启动项
/// </summary>
/// <returns>注册表中启动项(键值,启动路径)</returns>
public static Dictionary<string, string> GetAllStartupItems()
{
    try
    {
        Dictionary<string, string> dicAllStartupItems = new Dictionary<string, string>();
        RegistryKey registryKey = null;
        //获取HKEY_CURRENT_USER中的启动项
        registryKey = Microsoft.Win32.Registry.CurrentUser.OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", true);
        if (registryKey != null)
        {
            foreach (string strValeName in registryKey.GetValueNames())
            {
                if (!dicAllStartupItems.ContainsKey(strValeName))
                {
                    dicAllStartupItems.Add(strValeName, registryKey.GetValue(strValeName).ToString());
                }
            }
        }
        //获取HKEY_LOCAL_MACHINE中的启动项
        registryKey = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", true);
        if (registryKey != null)
        {
            foreach (string strValeName in registryKey.GetValueNames())
            {
                if (!dicAllStartupItems.ContainsKey(strValeName))
                {
                    dicAllStartupItems.Add(strValeName, registryKey.GetValue(strValeName).ToString());
                }
            }
        }
        return dicAllStartupItems;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}
#endregion
```

###### UAC 策略导致部分电脑无法开机自启
[UAC (用户帐户控制)](https://learn.microsoft.com/zh-cn/previous-versions/mt634236(v=vs.85%29) 是由微软在其 Windows Vista 及更高版本操作系统中采用的一种控制机制。其原理是通知用户是否对应用程序使用硬盘驱动器和系统文件授权，以达到帮助阻止恶意程序损坏系统的效果。
非管理员用户登录的 Windows 系统，设置注册表开机自启，要求以管理员权限运行的程序（软件图标右下角有个盾牌）无法自动启动，修改 UAC 策略禁用后即可开机自启，所有软件以管理员权限运行。
运行 -> gpedit.msc -> 计算机配置 -> Windows 设置 -> 安全设置 -> 本地策略 -> 安全选项 -> 禁用：用户账户控制：以管理员批准模式运行所有管理员
<img src="https://sadness96.github.io/images/blog/csharp-RegistryHelper/gpedit_uac.jpg"/>

##### 设置系统右键菜单
###### 桌面右键菜单项
``` CSharp
#region 注册表桌面右键菜单项
/// <summary>
/// 创建注册表桌面右键菜单项
/// </summary>
/// <param name="strName">键值名称</param>
/// <param name="strDisplayName">右键菜单显示名称,如果为空显示键值名称</param>
/// <param name="strSoftwarePath">启动软件路径</param>
/// <param name="strIcoPath">右键菜单图片路径,如果为空则不显示图片</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CreateDesktopRightClickMenu(string strName, string strDisplayName, string strSoftwarePath, string strIcoPath)
{
    try
    {
        if (string.IsNullOrEmpty(strName) || string.IsNullOrEmpty(strSoftwarePath))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"Directory\Background\shell\" + strName, true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(@"Directory\Background\shell\" + strName);
        }
        if (!string.IsNullOrEmpty(strDisplayName))
        {
            registryKey.SetValue("", strDisplayName);
        }
        if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath))
        {
            registryKey.SetValue("icon", strIcoPath);
        }
        registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"Directory\Background\shell\" + strName + @"\command", true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(@"Directory\Background\shell\" + strName + @"\command");
        }
        registryKey.SetValue("", strSoftwarePath);
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 创建注册表桌面右键菜单项(二级菜单)(WIN7-X64下测试通过)
/// </summary>
/// <param name="strName">一级菜单键值名称</param>
/// <param name="strDisplayName">一级菜单右键菜单显示名称,如果为空显示键值名称</param>
/// <param name="strIcoPath">一级菜单右键菜单图片路径,如果为空则不显示图片</param>
/// <param name="listSecondaryMenu">二级菜单配置</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CreateDesktopRightClickMenu2(string strName, string strDisplayName, string strIcoPath, List<SecondaryMenuModel> listSecondaryMenu)
{
    try
    {
        if (string.IsNullOrEmpty(strName) || listSecondaryMenu.Count < 1)
        {
            return false;
        }
        //创建一级菜单
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"Directory\Background\shell\" + strName, true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(@"Directory\Background\shell\" + strName);
        }
        if (!string.IsNullOrEmpty(strDisplayName))
        {
            registryKey.SetValue("MUIVerb", strDisplayName);
        }
        if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath))
        {
            registryKey.SetValue("icon", strIcoPath);
        }
        string strSecondaryMenuName = string.Empty;
        for (int iSecondaryMenu = 0; iSecondaryMenu < listSecondaryMenu.Count; iSecondaryMenu++)
        {
            if (iSecondaryMenu < listSecondaryMenu.Count - 1)
            {
                strSecondaryMenuName += listSecondaryMenu[iSecondaryMenu].strSecondaryMenuName;
                strSecondaryMenuName += ';';
            }
            else
            {
                strSecondaryMenuName += listSecondaryMenu[iSecondaryMenu].strSecondaryMenuName;
            }
        }
        if (!string.IsNullOrEmpty(strSecondaryMenuName))
        {
            registryKey.SetValue("SubCommands", strSecondaryMenuName);
        }
        //创建二级菜单
        foreach (SecondaryMenuModel vSecondaryMenu in listSecondaryMenu)
        {
            string strName2 = vSecondaryMenu.strSecondaryMenuName;
            string strDisplayName2 = vSecondaryMenu.strDisplayName;
            string strSoftwarePath2 = vSecondaryMenu.strSoftwarePath;
            string strIcoPath2 = vSecondaryMenu.strIcoPath;
            registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2, true);
            if (registryKey == null)
            {
                registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).CreateSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2);
            }
            if (!string.IsNullOrEmpty(strDisplayName2))
            {
                registryKey.SetValue("", strDisplayName2);
            }
            if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath2))
            {
                registryKey.SetValue("icon", strIcoPath2);
            }
            registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2 + @"\command", true);
            if (registryKey == null)
            {
                registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).CreateSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2 + @"\command");
            }
            registryKey.SetValue("", strSoftwarePath2);
        }
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 删除注册表桌面右键菜单项(二级菜单只删除一级菜单)
/// </summary>
/// <param name="strName">键值名称</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeleteDesktopRightClickMenu(string strName)
{
    try
    {
        if (string.IsNullOrEmpty(strName))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"Directory\Background\shell\", true);
        if (registryKey == null)
        {
            return false;
        }
        registryKey.DeleteSubKeyTree(strName);
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
#endregion
```

###### 文件夹右键菜单项
``` CSharp
#region 注册表文件夹右键菜单项
/// <summary>
/// 创建注册表文件夹右键菜单项
/// </summary>
/// <param name="strName">键值名称</param>
/// <param name="strDisplayName">右键菜单显示名称,如果为空显示键值名称</param>
/// <param name="strSoftwarePath">启动软件路径</param>
/// <param name="strIcoPath">右键菜单图片路径,如果为空则不显示图片</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CreateFolderRightClickMenu(string strName, string strDisplayName, string strSoftwarePath, string strIcoPath)
{
    try
    {
        if (string.IsNullOrEmpty(strName) || string.IsNullOrEmpty(strSoftwarePath))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"Folder\shell\" + strName, true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(@"Folder\shell\" + strName);
        }
        if (!string.IsNullOrEmpty(strDisplayName))
        {
            registryKey.SetValue("", strDisplayName);
        }
        if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath))
        {
            registryKey.SetValue("icon", strIcoPath);
        }
        registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"Folder\shell\" + strName + @"\command", true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(@"Folder\shell\" + strName + @"\command");
        }
        registryKey.SetValue("", strSoftwarePath);
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 创建注册表文件夹右键菜单项(二级菜单)(WIN7-X64下测试通过)
/// </summary>
/// <param name="strName">一级菜单键值名称</param>
/// <param name="strDisplayName">一级菜单右键菜单显示名称,如果为空显示键值名称</param>
/// <param name="strIcoPath">一级菜单右键菜单图片路径,如果为空则不显示图片</param>
/// <param name="listSecondaryMenu">二级菜单配置</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CreateFolderRightClickMenu2(string strName, string strDisplayName, string strIcoPath, List<SecondaryMenuModel> listSecondaryMenu)
{
    try
    {
        if (string.IsNullOrEmpty(strName) || listSecondaryMenu.Count < 1)
        {
            return false;
        }
        //创建一级菜单
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"Folder\shell\" + strName, true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(@"Folder\shell\" + strName);
        }
        if (!string.IsNullOrEmpty(strDisplayName))
        {
            registryKey.SetValue("MUIVerb", strDisplayName);
        }
        if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath))
        {
            registryKey.SetValue("icon", strIcoPath);
        }
        string strSecondaryMenuName = string.Empty;
        for (int iSecondaryMenu = 0; iSecondaryMenu < listSecondaryMenu.Count; iSecondaryMenu++)
        {
            if (iSecondaryMenu < listSecondaryMenu.Count - 1)
            {
                strSecondaryMenuName += listSecondaryMenu[iSecondaryMenu].strSecondaryMenuName;
                strSecondaryMenuName += ';';
            }
            else
            {
                strSecondaryMenuName += listSecondaryMenu[iSecondaryMenu].strSecondaryMenuName;
            }
        }
        if (!string.IsNullOrEmpty(strSecondaryMenuName))
        {
            registryKey.SetValue("SubCommands", strSecondaryMenuName);
        }
        //创建二级菜单
        foreach (SecondaryMenuModel vSecondaryMenu in listSecondaryMenu)
        {
            string strName2 = vSecondaryMenu.strSecondaryMenuName;
            string strDisplayName2 = vSecondaryMenu.strDisplayName;
            string strSoftwarePath2 = vSecondaryMenu.strSoftwarePath;
            string strIcoPath2 = vSecondaryMenu.strIcoPath;
            registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2, true);
            if (registryKey == null)
            {
                registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).CreateSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2);
            }
            if (!string.IsNullOrEmpty(strDisplayName2))
            {
                registryKey.SetValue("", strDisplayName2);
            }
            if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath2))
            {
                registryKey.SetValue("icon", strIcoPath2);
            }
            registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2 + @"\command", true);
            if (registryKey == null)
            {
                registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).CreateSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2 + @"\command");
            }
            registryKey.SetValue("", strSoftwarePath2);
        }
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 删除注册表文件夹右键菜单项(二级菜单只删除一级菜单)
/// </summary>
/// <param name="strName">键值名称</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeleteFolderRightClickMenu(string strName)
{
    try
    {
        if (string.IsNullOrEmpty(strName))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"Folder\shell\", true);
        if (registryKey == null)
        {
            return false;
        }
        registryKey.DeleteSubKeyTree(strName);
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
#endregion
```

###### 文件右键菜单项
``` CSharp
#region 注册表文件右键菜单项
/// <summary>
/// 创建注册表文件右键菜单项
/// </summary>
/// <param name="strName">键值名称</param>
/// <param name="strDisplayName">右键菜单显示名称,如果为空显示键值名称</param>
/// <param name="strSoftwarePath">启动软件路径</param>
/// <param name="strIcoPath">右键菜单图片路径,如果为空则不显示图片</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CreateFileRightClickMenu(string strName, string strDisplayName, string strSoftwarePath, string strIcoPath)
{
    try
    {
        if (string.IsNullOrEmpty(strName) || string.IsNullOrEmpty(strSoftwarePath))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"*\shell\" + strName, true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(@"*\shell\" + strName);
        }
        if (!string.IsNullOrEmpty(strDisplayName))
        {
            registryKey.SetValue("", strDisplayName);
        }
        if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath))
        {
            registryKey.SetValue("icon", strIcoPath);
        }
        registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"*\shell\" + strName + @"\command", true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(@"*\shell\" + strName + @"\command");
        }
        registryKey.SetValue("", strSoftwarePath);
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 创建注册表文件右键菜单项(二级菜单)(WIN7-X64下测试通过)
/// </summary>
/// <param name="strName">一级菜单键值名称</param>
/// <param name="strDisplayName">一级菜单右键菜单显示名称,如果为空显示键值名称</param>
/// <param name="strIcoPath">一级菜单右键菜单图片路径,如果为空则不显示图片</param>
/// <param name="listSecondaryMenu">二级菜单配置</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CreateFileRightClickMenu2(string strName, string strDisplayName, string strIcoPath, List<SecondaryMenuModel> listSecondaryMenu)
{
    try
    {
        if (string.IsNullOrEmpty(strName) || listSecondaryMenu.Count < 1)
        {
            return false;
        }
        //创建一级菜单
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"*\shell\" + strName, true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(@"*\shell\" + strName);
        }
        if (!string.IsNullOrEmpty(strDisplayName))
        {
            registryKey.SetValue("MUIVerb", strDisplayName);
        }
        if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath))
        {
            registryKey.SetValue("icon", strIcoPath);
        }
        string strSecondaryMenuName = string.Empty;
        for (int iSecondaryMenu = 0; iSecondaryMenu < listSecondaryMenu.Count; iSecondaryMenu++)
        {
            if (iSecondaryMenu < listSecondaryMenu.Count - 1)
            {
                strSecondaryMenuName += listSecondaryMenu[iSecondaryMenu].strSecondaryMenuName;
                strSecondaryMenuName += ';';
            }
            else
            {
                strSecondaryMenuName += listSecondaryMenu[iSecondaryMenu].strSecondaryMenuName;
            }
        }
        if (!string.IsNullOrEmpty(strSecondaryMenuName))
        {
            registryKey.SetValue("SubCommands", strSecondaryMenuName);
        }
        //创建二级菜单
        foreach (SecondaryMenuModel vSecondaryMenu in listSecondaryMenu)
        {
            string strName2 = vSecondaryMenu.strSecondaryMenuName;
            string strDisplayName2 = vSecondaryMenu.strDisplayName;
            string strSoftwarePath2 = vSecondaryMenu.strSoftwarePath;
            string strIcoPath2 = vSecondaryMenu.strIcoPath;
            registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2, true);
            if (registryKey == null)
            {
                registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).CreateSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2);
            }
            if (!string.IsNullOrEmpty(strDisplayName2))
            {
                registryKey.SetValue("", strDisplayName2);
            }
            if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath2))
            {
                registryKey.SetValue("icon", strIcoPath2);
            }
            registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2 + @"\command", true);
            if (registryKey == null)
            {
                registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).CreateSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2 + @"\command");
            }
            registryKey.SetValue("", strSoftwarePath2);
        }
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 删除注册表文件右键菜单项(二级菜单只删除一级菜单)
/// </summary>
/// <param name="strName">键值名称</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeleteFileRightClickMenu(string strName)
{
    try
    {
        if (string.IsNullOrEmpty(strName))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(@"*\shell\", true);
        if (registryKey == null)
        {
            return false;
        }
        registryKey.DeleteSubKeyTree(strName);
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
#endregion
```

###### 特定文件右键菜单项
``` CSharp
#region 注册表特定文件右键菜单项
/// <summary>
/// 创建注册表特定文件右键菜单项
/// </summary>
/// <param name="strFileType">特定文件类型(例:.txt|.exe)</param>
/// <param name="strName">键值名称</param>
/// <param name="strDisplayName">右键菜单显示名称,如果为空显示键值名称</param>
/// <param name="strSoftwarePath">启动软件路径</param>
/// <param name="strIcoPath">右键菜单图片路径,如果为空则不显示图片</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CreateSpecificFileRightClickMenu(string strFileType, string strName, string strDisplayName, string strSoftwarePath, string strIcoPath)
{
    try
    {
        if (string.IsNullOrEmpty(strFileType) || string.IsNullOrEmpty(strName) || string.IsNullOrEmpty(strSoftwarePath))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType, true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strFileType);
        }
        //获取(默认)中的数据
        string strDefault = registryKey.ValueCount >= 1 ? registryKey.GetValue("").ToString() : string.Empty;
        if (string.IsNullOrEmpty(strDefault))
        {
            //如果该后缀名里(默认)没有值,则创建shell写入菜单功能
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType + @"\shell\" + strName, true);
            if (registryKey == null)
            {
                registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strFileType + @"\shell\" + strName);
            }
            if (!string.IsNullOrEmpty(strDisplayName))
            {
                registryKey.SetValue("", strDisplayName);
            }
            if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath))
            {
                registryKey.SetValue("icon", strIcoPath);
            }
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType + @"\shell\" + strName + @"\command", true);
            if (registryKey == null)
            {
                registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strFileType + @"\shell\" + strName + @"\command");
            }
            registryKey.SetValue("", strSoftwarePath);
        }
        else
        {
            //如果该后缀名里(默认)存在值,读取值所在的路径创建shell写入菜单功能
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strDefault + @"\shell\" + strName, true);
            if (registryKey == null)
            {
                registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strDefault + @"\shell\" + strName);
            }
            if (!string.IsNullOrEmpty(strDisplayName))
            {
                registryKey.SetValue("", strDisplayName);
            }
            if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath))
            {
                registryKey.SetValue("icon", strIcoPath);
            }
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strDefault + @"\shell\" + strName + @"\command", true);
            if (registryKey == null)
            {
                registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strDefault + @"\shell\" + strName + @"\command");
            }
            registryKey.SetValue("", strSoftwarePath);
        }
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 创建注册表特定文件右键菜单项(二级菜单)(WIN7-X64下测试通过)
/// </summary>
/// <param name="strFileType">特定文件类型(例:.txt|.exe)</param>
/// <param name="strName">一级菜单键值名称</param>
/// <param name="strDisplayName">一级菜单右键菜单显示名称,如果为空显示键值名称</param>
/// <param name="strIcoPath">一级菜单右键菜单图片路径,如果为空则不显示图片</param>
/// <param name="listSecondaryMenu">二级菜单配置</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CreateSpecificFileRightClickMenu2(string strFileType, string strName, string strDisplayName, string strIcoPath, List<SecondaryMenuModel> listSecondaryMenu)
{
    try
    {
        if (string.IsNullOrEmpty(strFileType) || string.IsNullOrEmpty(strName) || listSecondaryMenu.Count < 1)
        {
            return false;
        }
        //创建一级菜单
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType, true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strFileType);
        }
        //获取(默认)中的数据
        string strDefault = registryKey.ValueCount >= 1 ? registryKey.GetValue("").ToString() : string.Empty;
        if (string.IsNullOrEmpty(strDefault))
        {
            //如果该后缀名里(默认)没有值,则创建shell写入菜单功能
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType + @"\shell\\" + strName, true);
            if (registryKey == null)
            {
                registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strFileType + @"\shell\\" + strName);
            }
            if (!string.IsNullOrEmpty(strDisplayName))
            {
                registryKey.SetValue("MUIVerb", strDisplayName);
            }
            if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath))
            {
                registryKey.SetValue("icon", strIcoPath);
            }
            string strSecondaryMenuName = string.Empty;
            for (int iSecondaryMenu = 0; iSecondaryMenu < listSecondaryMenu.Count; iSecondaryMenu++)
            {
                if (iSecondaryMenu < listSecondaryMenu.Count - 1)
                {
                    strSecondaryMenuName += listSecondaryMenu[iSecondaryMenu].strSecondaryMenuName;
                    strSecondaryMenuName += ';';
                }
                else
                {
                    strSecondaryMenuName += listSecondaryMenu[iSecondaryMenu].strSecondaryMenuName;
                }
            }
            if (!string.IsNullOrEmpty(strSecondaryMenuName))
            {
                registryKey.SetValue("SubCommands", strSecondaryMenuName);
            }
        }
        else
        {
            //如果该后缀名里(默认)存在值,读取值所在的路径创建shell写入菜单功能
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strDefault + @"\shell\\" + strName, true);
            if (registryKey == null)
            {
                registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strDefault + @"\shell\\" + strName);
            }
            if (!string.IsNullOrEmpty(strDisplayName))
            {
                registryKey.SetValue("MUIVerb", strDisplayName);
            }
            if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath))
            {
                registryKey.SetValue("icon", strIcoPath);
            }
            string strSecondaryMenuName = string.Empty;
            for (int iSecondaryMenu = 0; iSecondaryMenu < listSecondaryMenu.Count; iSecondaryMenu++)
            {
                if (iSecondaryMenu < listSecondaryMenu.Count - 1)
                {
                    strSecondaryMenuName += listSecondaryMenu[iSecondaryMenu].strSecondaryMenuName;
                    strSecondaryMenuName += ';';
                }
                else
                {
                    strSecondaryMenuName += listSecondaryMenu[iSecondaryMenu].strSecondaryMenuName;
                }
            }
            if (!string.IsNullOrEmpty(strSecondaryMenuName))
            {
                registryKey.SetValue("SubCommands", strSecondaryMenuName);
            }
        }
        //创建二级菜单
        foreach (SecondaryMenuModel vSecondaryMenu in listSecondaryMenu)
        {
            string strName2 = vSecondaryMenu.strSecondaryMenuName;
            string strDisplayName2 = vSecondaryMenu.strDisplayName;
            string strSoftwarePath2 = vSecondaryMenu.strSoftwarePath;
            string strIcoPath2 = vSecondaryMenu.strIcoPath;
            registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2, true);
            if (registryKey == null)
            {
                registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).CreateSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2);
            }
            if (!string.IsNullOrEmpty(strDisplayName2))
            {
                registryKey.SetValue("", strDisplayName2);
            }
            if (!string.IsNullOrEmpty(strIcoPath) && File.Exists(strIcoPath2))
            {
                registryKey.SetValue("icon", strIcoPath2);
            }
            registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).OpenSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2 + @"\command", true);
            if (registryKey == null)
            {
                registryKey = RegistryKey.OpenBaseKey(RegistryHive.LocalMachine, Environment.Is64BitOperatingSystem ? RegistryView.Registry64 : RegistryView.Registry32).CreateSubKey(@"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\CommandStore\shell\" + strName2 + @"\command");
            }
            registryKey.SetValue("", strSoftwarePath2);
        }
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 删除注册表特定文件右键菜单项(二级菜单只删除一级菜单)
/// </summary>
/// <param name="strFileType">特定文件类型(例:.txt|.exe)</param>
/// <param name="strName">键值名称</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeleteSpecificFileRightClickMenu(string strFileType, string strName)
{
    try
    {
        if (string.IsNullOrEmpty(strFileType) || string.IsNullOrEmpty(strName))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType, true);
        //获取(默认)中的数据
        string strDefault = registryKey.ValueCount >= 1 ? registryKey.GetValue("").ToString() : string.Empty;
        if (string.IsNullOrEmpty(strDefault))
        {
            //如果该后缀名里(默认)没有值,则创建shell写入菜单功能
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType + @"\shell\", true);
            if (registryKey == null)
            {
                return false;
            }
            registryKey.DeleteSubKeyTree(strName);
        }
        else
        {
            //如果该后缀名里(默认)存在值,读取值所在的路径创建shell写入菜单功能
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strDefault + @"\shell\", true);
            if (registryKey == null)
            {
                return false;
            }
            registryKey.DeleteSubKeyTree(strName);
        }
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
#endregion
```

##### 设置系统特定文件后缀默认图标
``` CSharp
#region Registry Default Icon
/// <summary>
/// 修改特定后缀文件默认图标(需重启电脑)
/// </summary>
/// <param name="strFileType">特定文件类型(例:.txt|.exe)</param>
/// <param name="strIcoPath">替换图片路径</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool ModifyDefaultIcon(string strFileType, string strIcoPath)
{
    try
    {
        if (string.IsNullOrEmpty(strFileType) || string.IsNullOrEmpty(strIcoPath) || !File.Exists(strIcoPath))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType, true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strFileType);
        }
        //获取(默认)中的数据
        string strDefault = registryKey.ValueCount >= 1 ? registryKey.GetValue("").ToString() : string.Empty;
        if (string.IsNullOrEmpty(strDefault))
        {
            //如果该后缀名里(默认)没有值,则创建shell写入菜单功能
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType + @"\DefaultIcon\", true);
            if (registryKey == null)
            {
                registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strFileType + @"\DefaultIcon\");
            }
            if (!string.IsNullOrEmpty(strIcoPath))
            {
                registryKey.SetValue("", strIcoPath);
            }
        }
        else
        {
            //如果该后缀名里(默认)存在值,读取值所在的路径创建shell写入菜单功能
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strDefault + @"\DefaultIcon\", true);
            if (registryKey == null)
            {
                registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strDefault + @"\DefaultIcon\");
            }
            if (!string.IsNullOrEmpty(strIcoPath))
            {
                registryKey.SetValue("", strIcoPath);
            }
        }
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
#endregion
```

##### 设置系统特定后缀文件默认程序
``` CSharp
#region Registry Default Programs
/// <summary>
/// 修改特定后缀文件默认程序
/// </summary>
/// <param name="strFileType">特定文件类型(例:.txt|.exe)</param>
/// <param name="strSoftwarePath">替换程序路径</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool ModifyDefaultPrograms(string strFileType, string strSoftwarePath)
{
    try
    {
        if (string.IsNullOrEmpty(strFileType) || string.IsNullOrEmpty(strSoftwarePath))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType, true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strFileType);
        }
        //获取(默认)中的数据
        string strDefault = registryKey.ValueCount >= 1 ? registryKey.GetValue("").ToString() : string.Empty;
        if (string.IsNullOrEmpty(strDefault))
        {
            //如果该后缀名里(默认)没有值,则创建shell写入菜单功能
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strFileType + @"\shell\open\command\", true);
            if (registryKey == null)
            {
                registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strFileType + @"\shell\open\command\");
            }
            if (!string.IsNullOrEmpty(strSoftwarePath))
            {
                registryKey.SetValue("", strSoftwarePath);
            }
        }
        else
        {
            //如果该后缀名里(默认)存在值,读取值所在的路径创建shell写入菜单功能
            registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strDefault + @"\shell\open\command\", true);
            if (registryKey == null)
            {
                registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strDefault + @"\shell\open\command\");
            }
            if (!string.IsNullOrEmpty(strSoftwarePath))
            {
                registryKey.SetValue("", strSoftwarePath);
            }
        }
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
#endregion
```

##### 创建 URL Protocol 协议,通过网页打开本地应用
由于公司业务需求，制作[单点登录](https://baike.baidu.com/item/%E5%8D%95%E7%82%B9%E7%99%BB%E5%BD%95/4940767?fr=aladdin)功能，该方法作为比较常见的一种，安装C/S端软件时写入注册表，B/S程序通过A标签即可打开C/S端程序并且自动登录B/S的用户名密码。
``` CSharp
#region Registry URL Protocol
/// <summary>
/// 创建 URL Protocol 协议,通过网页打开本地应用
/// </summary>
/// <param name="strName">键值名称</param>
/// <param name="strSoftwarePath">启动软件路径</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CreateURLProtocol(string strName, string strSoftwarePath)
{
    try
    {
        if (string.IsNullOrEmpty(strName) || string.IsNullOrEmpty(strSoftwarePath))
        {
            return false;
        }
        //Web端调用方法:<a href="strName://"%1"参数>URL Protocol</a>
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strName + @"\shell\open\command", true);
        if (registryKey == null)
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot.CreateSubKey(strName + @"\shell\open\command");
        }
        registryKey.SetValue("", strSoftwarePath);
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 删除 URL Protocol 协议
/// </summary>
/// <param name="strName">键值名称</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeleteURLProtocol(string strName)
{
    try
    {
        if (string.IsNullOrEmpty(strName))
        {
            return false;
        }
        RegistryKey registryKey = Microsoft.Win32.Registry.ClassesRoot.OpenSubKey(strName, true);
        if (registryKey == null)
        {
            return false;
        }
        else
        {
            registryKey = Microsoft.Win32.Registry.ClassesRoot;
            registryKey.DeleteSubKeyTree(strName);
        }
        registryKey.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
#endregion
```