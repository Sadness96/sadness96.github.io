---
title: FTP 共享帮助类
date: 2017-05-24 20:30:30
tags: [c#,helper,ftp]
categories: C#.Net
---
### 基于 Serv-U 搭建的 FTP 帮助类
<!-- more -->
#### 简介
[Serv-U](https://www.serv-u.com/) 是一种被广泛运用的FTP服务器端软件。
现FTP服务器还是公司企业普遍运用的文件储存方式，编写帮助类用于自动上传下载管理服务。

#### 核心代码
[FTPSharingHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/FileIO.Helper/FTPSharing/FTPSharingHelper.cs)
``` CSharp
/// <summary>
/// FTP服务器URL(统一资源定位符)
/// </summary>
public static string strFTPServerURL { get; set; }

/// <summary>
/// FTP服务器IP地址
/// </summary>
public static string strFTPServerIP { get; set; }

/// <summary>
/// FTP服务器端口号
/// </summary>
public static string strFTPServerPort { get; set; }

/// <summary>
/// FTP服务器路径
/// </summary>
public static string strFTPServerPath { get; set; }

/// <summary>
/// FTP服务器用户名
/// </summary>
public static string strFTPServerUserID { get; set; }

/// <summary>
/// FTP服务器用户密码
/// </summary>
public static string strFTPServerPassword { get; set; }

/// <summary>  
/// 实现了文件传输协议(FTP)客户端
/// </summary>  
FtpWebRequest Request = null;

/// <summary>
/// FTP连接URL
/// </summary>
/// <param name="FTPServerURL">FTP服务器URL(统一资源定位符)</param>
/// <param name="FTPServerUserID">FTP服务器用户名</param>
/// <param name="FTPServerPassword">FTP服务器用户密码</param>
public void FTPConnectionURL(string FTPServerURL, string FTPServerUserID, string FTPServerPassword)
{
    strFTPServerURL = FTPServerURL;
    strFTPServerUserID = FTPServerUserID;
    strFTPServerPassword = FTPServerPassword;
}

/// <summary>
/// FTP连接URL
/// </summary>
/// <param name="FTPServerIP">FTP服务器IP地址</param>
/// <param name="FTPServerPort">FTP服务器端口号</param>
/// <param name="FTPServerPath">FTP服务器路径</param>
/// <param name="FTPServerUserID">FTP服务器用户名</param>
/// <param name="FTPServerPassword">FTP服务器用户密码</param>
public void FTPConnectionURL(string FTPServerIP, string FTPServerPort, string FTPServerPath, string FTPServerUserID, string FTPServerPassword)
{
    strFTPServerIP = FTPServerIP;
    strFTPServerPort = FTPServerPort;
    strFTPServerPath = FTPServerPath;
    strFTPServerUserID = FTPServerUserID;
    strFTPServerPassword = FTPServerPassword;
    strFTPServerURL = "ftp://" + strFTPServerIP + ":" + strFTPServerPort + "/" + strFTPServerPath + "/";
}

/// <summary>
/// 获得URL路径下文件列表
/// </summary>
/// <returns>文件列表List</returns>
public List<string> GetFilesDetailList()
{
    try
    {
        List<string> listAllFiles = new List<string>();
        Request = (FtpWebRequest)FtpWebRequest.Create(new Uri(strFTPServerURL));
        Request.Credentials = new NetworkCredential(strFTPServerUserID, strFTPServerPassword);
        Request.Method = WebRequestMethods.Ftp.ListDirectoryDetails;
        WebResponse Response = Request.GetResponse();
        StreamReader Reader = new StreamReader(Response.GetResponseStream());
        string strLine = Reader.ReadLine();
        while (strLine != null)
        {
            if (AnalysisLISTCommand(strLine).FileType == "-")
            {
                listAllFiles.Add(AnalysisLISTCommand(strLine).FileName);
            }
            strLine = Reader.ReadLine();
        }
        Reader.Close();
        Response.Close();
        return listAllFiles;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 获得URL路径下文件夹列表
/// </summary>
/// <returns>文件列表List</returns>
public List<string> GetFoldersDetailList()
{
    try
    {
        List<string> listAllFiles = new List<string>();
        Request = (FtpWebRequest)FtpWebRequest.Create(new Uri(strFTPServerURL));
        Request.Credentials = new NetworkCredential(strFTPServerUserID, strFTPServerPassword);
        Request.Method = WebRequestMethods.Ftp.ListDirectoryDetails;
        WebResponse Response = Request.GetResponse();
        StreamReader Reader = new StreamReader(Response.GetResponseStream());
        string strLine = Reader.ReadLine();
        while (strLine != null)
        {
            if (AnalysisLISTCommand(strLine).FileType == "d")
            {
                listAllFiles.Add(AnalysisLISTCommand(strLine).FileName);
            }
            strLine = Reader.ReadLine();
        }
        Reader.Close();
        Response.Close();
        return listAllFiles;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 获得指定URL路径下指定文件属性
/// </summary>
/// <param name="strFileName">URL中指定文件名</param>
/// <returns>FTP-FTP-LIST命令返回参数数据模型,失败返回NULL</returns>
public FTPListTypeModel GetFolderOrFileType(string strFileName)
{
    try
    {
        FtpWebRequest fRequest = (FtpWebRequest)FtpWebRequest.Create(new Uri(strFTPServerURL + strFileName));
        fRequest.Credentials = new NetworkCredential(strFTPServerUserID, strFTPServerPassword);
        fRequest.Method = WebRequestMethods.Ftp.ListDirectoryDetails;
        FtpWebResponse Response = (FtpWebResponse)fRequest.GetResponse();
        Stream ftpStream = Response.GetResponseStream();
        StreamReader Reader = new StreamReader(ftpStream);
        string strLine = Reader.ReadLine();
        Reader.Close();
        Response.Close();
        return AnalysisLISTCommand(strLine);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 新建URL路径下文件夹
/// </summary>
/// <param name="strFolderName">文件夹名称</param>
/// <returns>成功返回true,失败返回false</returns>
public bool CreateFolder(string strFolderName)
{
    try
    {
        Request = (FtpWebRequest)FtpWebRequest.Create(new Uri(strFTPServerURL + strFolderName));
        Request.Method = WebRequestMethods.Ftp.MakeDirectory;
        Request.UseBinary = true;
        Request.Credentials = new NetworkCredential(strFTPServerUserID, strFTPServerPassword);
        FtpWebResponse response = (FtpWebResponse)Request.GetResponse();
        Stream ftpStream = response.GetResponseStream();
        ftpStream.Close();
        response.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 删除URL路径下指定文件
/// </summary>
/// <param name="strFileName">URL中指定文件名</param>
/// <returns>成功返回true,失败返回false</returns>
public bool DeleteFile(string strFileName)
{
    try
    {
        Request = (FtpWebRequest)FtpWebRequest.Create(new Uri(strFTPServerURL + strFileName));
        Request.Credentials = new NetworkCredential(strFTPServerUserID, strFTPServerPassword);
        Request.Method = WebRequestMethods.Ftp.DeleteFile;
        Request.KeepAlive = false;
        FtpWebResponse Response = (FtpWebResponse)Request.GetResponse();
        Stream ftpStream = Response.GetResponseStream();
        ftpStream.Close();
        Response.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 删除URL路径下指定文件夹
/// </summary>
/// <param name="strFolderName">文件夹名</param>
/// <returns>成功返回true,失败返回false</returns>
public bool DeleteFolder(string strFolderName)
{
    try
    {
        Request = (FtpWebRequest)FtpWebRequest.Create(new Uri(strFTPServerURL + strFolderName));
        Request.Credentials = new NetworkCredential(strFTPServerUserID, strFTPServerPassword);
        Request.Method = WebRequestMethods.Ftp.RemoveDirectory;
        Request.KeepAlive = false;
        FtpWebResponse Response = (FtpWebResponse)Request.GetResponse();
        Stream ftpStream = Response.GetResponseStream();
        ftpStream.Close();
        Response.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 上传文件到指定URL路径
/// </summary>
/// <param name="strFilePath">需要上传的文件路径</param>
/// <returns>成功返回true,失败返回false</returns>
public bool UploadFile(string strFilePath)
{
    try
    {
        FileInfo fileInfo = new FileInfo(strFilePath);
        Request = (FtpWebRequest)FtpWebRequest.Create(new Uri(strFTPServerURL + fileInfo.Name));
        Request.Credentials = new NetworkCredential(strFTPServerUserID, strFTPServerPassword);
        Request.Method = WebRequestMethods.Ftp.UploadFile;
        Request.KeepAlive = false;
        Request.UseBinary = true;
        Request.ContentLength = fileInfo.Length;
        int buffLength = 2048;
        byte[] buff = new byte[buffLength];
        int contentLen;
        FileStream fileFtpStream = fileInfo.OpenRead();
        Stream ftpStream = Request.GetRequestStream();
        contentLen = fileFtpStream.Read(buff, 0, buffLength);
        while (contentLen != 0)
        {
            ftpStream.Write(buff, 0, contentLen);
            contentLen = fileFtpStream.Read(buff, 0, buffLength);
        }
        ftpStream.Close();
        fileFtpStream.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 下载指定URL中的文件到指定路径
/// </summary>
/// <param name="strFilePath">文件下载到的路径</param>
/// <param name="strFileName">URL中指定文件名</param>
/// <returns>成功返回true,失败返回false</returns>
public bool DownloadFile(string strFilePath, string strFileName)
{
    try
    {
        if (!Directory.Exists(strFilePath))
        {
            Directory.CreateDirectory(strFilePath);
        }
        FileStream fileStream = new FileStream(strFilePath + "\\" + strFileName, FileMode.Create);
        Request = (FtpWebRequest)FtpWebRequest.Create(new Uri(strFTPServerURL + strFileName));
        Request.Credentials = new NetworkCredential(strFTPServerUserID, strFTPServerPassword);
        Request.Method = WebRequestMethods.Ftp.DownloadFile;
        Request.UseBinary = true;
        FtpWebResponse response = (FtpWebResponse)Request.GetResponse();
        Stream ftpStream = response.GetResponseStream();
        int bufferSize = 2048;
        int readCount;
        byte[] buffer = new byte[bufferSize];
        readCount = ftpStream.Read(buffer, 0, bufferSize);
        while (readCount > 0)
        {
            fileStream.Write(buffer, 0, readCount);
            readCount = ftpStream.Read(buffer, 0, bufferSize);
        }
        ftpStream.Close();
        fileStream.Close();
        response.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 解析FTP-LIST命令
/// </summary>
/// <param name="strListCommand">FTP-LIST命令</param>
private FTPListTypeModel AnalysisLISTCommand(string strListCommand)
{
    FTPListTypeModel ListType = new FTPListTypeModel();
    //根据空格拆分成 9 部分
    string[] strListCommandSplit = strListCommand.Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
    //文件类型 位数(1)
    ListType.FileType = strListCommandSplit[0].Substring(0, 1);
    //文件属主拥有权限 位数(2-4)
    ListType.UserOwnership = strListCommandSplit[0].Substring(1, 3);
    //文件属主同一组用户拥有权限 位数(5-7)
    ListType.GroupUserOwnership = strListCommandSplit[0].Substring(4, 3);
    //其他用户拥有权限 位数(8-10)
    ListType.OtherUserOwnership = strListCommandSplit[0].Substring(7, 3);
    //未知参数1 (1)
    ListType.UnknownParameter1 = strListCommandSplit[1];
    //未知参数2 (user)
    ListType.UnknownParameter2 = strListCommandSplit[2];
    //未知参数3 (group)
    ListType.UnknownParameter3 = strListCommandSplit[3];
    //文件大小(文件夹为0)
    ListType.FileSize = strListCommandSplit[4];
    //文件月份
    ListType.FileMonth = strListCommandSplit[5];
    //文件日期
    ListType.FileDay = strListCommandSplit[6];
    //文件年份或时间
    ListType.FileYearOrTime = strListCommandSplit[7];
    //文件名称
    ListType.FileName = strListCommandSplit[8];
    return ListType;
}
```

#### 应用
一个同学的需求：公司是做三维模型模具的，领导要求每个人每次修改上传至FTP，然后领导每天会定时完整下载（虽然我也不理解为什么他们不用SVN或者GIT，需求是什么样的就做成什么样嘛！）。功能需求是制作一个控制台程序，通过[任务计划](/blog/2017/09/18/csharp-TaskschdHelper/)中每天下班定时执行即可。FTP参数调用本地[INI配置文件](/blog/2017/05/10/csharp-INIHelper/)。
``` CSharp
static void Main(string[] args)
{
    string strPath = System.AppDomain.CurrentDomain.SetupInformation.ApplicationBase + "Config.ini";
    string strFTP = INIHelper.INIGetStringValue(strPath, "System", "FTP", "");
    string strUser = INIHelper.INIGetStringValue(strPath, "System", "User", "");
    string strPassword = INIHelper.INIGetStringValue(strPath, "System", "Password", "");
    string strSavePath = INIHelper.INIGetStringValue(strPath, "System", "SavePath", "");
    FTPSharingHelper ftp = new FTPSharingHelper();
    ftp.FTPConnectionURL(strFTP, strUser, strPassword);
    DownloadFile(ftp, strFTP, strUser, strPassword, strSavePath);
}

/// <summary>
/// 下载目录所有文件
/// </summary>
/// <param name="ftp">FTP链接</param>
/// <param name="strFTP">链接地址</param>
/// <param name="strUser">用户名</param>
/// <param name="strPassword">密码</param>
/// <param name="strSavePath">保存路径</param>
public static void DownloadFile(FTPSharingHelper ftp, string strFTP, string strUser, string strPassword, string strSavePath)
{
    try
    {
        ftp.FTPConnectionURL(strFTP, strUser, strPassword);
        //下载当前目录文件
        List<string> list1 = ftp.GetFilesDetailList();
        if (list1 != null)
        {
            foreach (var item in list1)
            {
                ftp.DownloadFile(strSavePath, item);
            }
        }
        //下载文件夹文件
        List<string> list2 = ftp.GetFoldersDetailList();
        if (list2 != null)
        {
            foreach (var item in list2)
            {
                if (item.Substring(0, 1).Equals("."))
                {
                    continue;
                }
                string strNewFTP = strFTP + item + "/";
                string strNewSavePath = strSavePath + "\\" + item;
                if (!Directory.Exists(strNewSavePath))
                {
                    Directory.CreateDirectory(strNewSavePath);
                }
                DownloadFile(ftp, strNewFTP, strUser, strPassword, strNewSavePath);
            }
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
    }
}
```