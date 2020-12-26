---
title: 文件共享帮助类
date: 2017-05-23 22:34:50
tags: [c#,helper,windows,filesharing]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-DevFramework/%E6%96%87%E4%BB%B6%E5%85%B1%E4%BA%AB.png"/>

<!-- more -->
### 基于 Win32 API 与 CMD 命令制作可视化文件共享工具
#### 简介
不论是工作使用或是平时经常会出现两台电脑网线直连拷贝文件的情况（通过外网传输速度会受带宽限制，飞秋这种局域网聊天软件需要两台电脑同时安装，硬件存储设备需要拷贝两次，Ftp需要搭建环境，所以大部分情况下文件共享还是最好的解决方案），但是就像我这种对系统比较熟悉的人，每次操作也要找一会，与其记住命令不如做出一个可视化界面。
#### 核心代码
[FileSharingHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/FileIO.Helper/FileSharing/FileSharingHelper.cs)
``` CSharp
/// <summary>
/// 读取共享文件信息
/// </summary>
/// <returns>共享信息DataTable('name':'共享文件名称','path':'共享文件路径','permissions':'访问控制权限','type':'共享文件属性')</returns>
public static DataTable InquireShareFile()
{
    try
    {
        ManagementObjectSearcher searcher = new ManagementObjectSearcher("select * from win32_share");
        DataTable ShareFile = new DataTable();
        ShareFile.Columns.Add("name");
        ShareFile.Columns.Add("path");
        ShareFile.Columns.Add("permissions");
        ShareFile.Columns.Add("type");
        foreach (ManagementObject share in searcher.Get())
        {
            try
            {
                //获取共享文件信息
                string name = share["Name"].ToString();
                string path = share["Path"].ToString();
                string type = share["Type"].ToString();
                if (type == "0")
                { type = "磁盘驱动器"; }
                else if (type == "1")
                { type = "打印队列"; }
                else if (type == "2")
                { type = "设备"; }
                else if (type == "3")
                { type = "IPC"; }
                else if (type == "2147483648")
                { type = "磁盘驱动器管理"; }
                else if (type == "2147483649")
                { type = "打印队列管理"; }
                else if (type == "2147483650")
                { type = "设备管理"; }
                else if (type == "2147483651")
                { type = "IPC 管理员"; }
                //获得共享文件访问权限(通过cmd搜索)
                string Permissions = "";
                string cmd = string.Format("net share {0}", name);
                string strOutput = ImplementationCMD(cmd);
                if (strOutput.IndexOf("FULL") > -1)
                { Permissions = "完全控制"; }
                else if (strOutput.IndexOf("READ") > -1)
                { Permissions = "只读"; }
                else if (strOutput.IndexOf("CHANGE") > -1)
                { Permissions = "读取/写入"; }
                //数据写入DataTable
                DataRow dr = ShareFile.NewRow();
                dr["name"] = name;
                dr["path"] = path;
                dr["permissions"] = Permissions;
                dr["type"] = type;
                ShareFile.Rows.Add(dr);
            }
            catch (Exception ex)
            {
                TXTHelper.Logs(ex.ToString());
                return null;
            }
        }
        return ShareFile;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 新增共享
/// </summary>
/// <param name="FolderPath">共享文件路径</param>
/// <param name="ShareName">共享文件名称</param>
/// <param name="Permissions">访问控制权限('完全控制':'FULL','只读':'READ','读取/写入':'CHANGE')</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool AddShareFolder(string FolderPath, string ShareName, string Permissions)
{
    try
    {
        //输入命令NET SHARE sharename=drive:path [/GRANT:user,[READ | CHANGE | FULL]
        string cmd = string.Format(@"net share {0}={1} /grant:{2},{3}", ShareName, FolderPath, System.Environment.UserName, Permissions);
        string strOutput = ImplementationCMD(cmd);
        return strOutput.IndexOf("共享成功") > -1 ? true : false;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 新增共享
/// </summary>
/// <param name="FolderPath">共享文件路径</param>
/// <param name="ShareName">共享文件名称</param>
/// <param name="PermissionsType">访问控制权限(枚举)</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool AddShareFolder(string FolderPath, string ShareName, SharingPermissions.PermissionsType PermissionsType)
{
    try
    {
        //输入命令NET SHARE sharename=drive:path [/GRANT:user,[READ | CHANGE | FULL]
        string cmd = string.Format(@"net share {0}={1} /grant:{2},{3}", ShareName, FolderPath, System.Environment.UserName, PermissionsType);
        string strOutput = ImplementationCMD(cmd);
        return strOutput.IndexOf("共享成功") > -1 ? true : false;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 删除共享
/// </summary>
/// <param name="FolderPath">共享文件夹路径</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeleteShareFolder(string FolderPath)
{
    try
    {
        //输入命令NET SHARE sharename \\computername /DELETE
        string cmd = string.Format(@"net share {0} /delete /y", FolderPath);
        string strOutput = ImplementationCMD(cmd);
        return strOutput.IndexOf("已经删除") > -1 ? true : false;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 执行CMD命令
/// </summary>
/// <param name="cmd">cmd命令</param>
/// <returns>cmd窗口的输出信息</returns>
private static string ImplementationCMD(string cmd)
{
    try
    {
        //使用cmd命令对文件共享进行操作
        System.Diagnostics.Process p = new System.Diagnostics.Process();
        p.StartInfo.FileName = "cmd.exe";
        //是否使用操作系统shell启动
        p.StartInfo.UseShellExecute = false;
        //接受来自调用程序的输入信息
        p.StartInfo.RedirectStandardInput = true;
        //由调用程序获取输出信息
        p.StartInfo.RedirectStandardOutput = true;
        //重定向标准错误输出
        p.StartInfo.RedirectStandardError = true;
        //不显示程序窗口
        p.StartInfo.CreateNoWindow = true;
        //启动程序
        p.Start();
        //执行CMD命令
        p.StandardInput.WriteLine(cmd);
        p.StandardInput.AutoFlush = true;
        p.StandardInput.WriteLine("exit");
        //获取cmd窗口的输出信息
        string strOutput = p.StandardOutput.ReadToEnd();
        p.WaitForExit();//等待程序执行完退出进程
        p.Close();
        return strOutput;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}
```