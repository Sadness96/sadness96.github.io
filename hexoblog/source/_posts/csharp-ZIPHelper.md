---
title: ZIP压缩文件帮助类
date: 2017-05-25 11:03:18
tags: [c#,helper,zip,7z,gzip]
categories: C#.Net
---
### 操作 ZIP/7z 压缩文件帮助类，以及压缩数据文本的 GZIP 帮助类
<!-- more -->
#### 简介
工作中会有需求把数据集打包成压缩文件，封装一个帮助以方便调用。
[ZIP](https://baike.baidu.com/item/Zip/16684862#viewPageContent) 通用且最为常见的压缩格式。
[7-ZIP](https://www.7-zip.org/) 自由开源的压缩格式，压缩效果要比普通的ZIP效果要好。
[GZIP](https://baike.baidu.com/item/gzip/4487553?fr=aladdin) 用于压缩数据流或文本。

#### 帮助类代码及引用
##### ZIP：
[ZIPHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/FileIO.Helper/ZIP/ZIPHelper.cs) 引用 ICSharpCode.SharpZipLib.Zip 库
``` CSharp
/// <summary>
/// 压缩ZIP文件
/// </summary>
/// <param name="strZipPath">ZIP压缩文件保存位置</param>
/// <param name="listFolderOrFilePath">需要压缩的文件夹或文件</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CompressionZip(string strZipPath, List<string> listFolderOrFilePath)
{
    return CompressionZip(strZipPath, listFolderOrFilePath, string.Empty);
}

/// <summary>
/// 压缩ZIP文件(加密)
/// </summary>
/// <param name="strZipPath">ZIP压缩文件保存位置</param>
/// <param name="listFolderOrFilePath">需要压缩的文件夹或文件</param>
/// <param name="strPassword">压缩文件密码</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CompressionZip(string strZipPath, List<string> listFolderOrFilePath, string strPassword)
{
    try
    {
        ZipOutputStream ComStream = new ZipOutputStream(File.Create(strZipPath));
        //压缩等级(0-9)
        ComStream.SetLevel(9);
        //压缩文件加密
        if (!string.IsNullOrEmpty(strPassword))
        {
            ComStream.Password = strPassword;
        }
        foreach (string strFolderOrFilePath in listFolderOrFilePath)
        {
            if (Directory.Exists(strFolderOrFilePath))
            {
                //如果路径是文件目录
                CompressionZipDirectory(strFolderOrFilePath, ComStream, string.Empty);
            }
            else if (File.Exists(strFolderOrFilePath))
            {
                //如果路径是文件路径
                FileStream fileStream = File.OpenRead(strFolderOrFilePath);
                byte[] btsLength = new byte[fileStream.Length];
                fileStream.Read(btsLength, 0, btsLength.Length);
                ZipEntry zipEntry = new ZipEntry(new FileInfo(strFolderOrFilePath).Name);
                ComStream.PutNextEntry(zipEntry);
                ComStream.Write(btsLength, 0, btsLength.Length);
            }
        }
        ComStream.Finish();
        ComStream.Close();
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 压缩ZIP文件夹
/// </summary>
/// <param name="strRootPath">根目录路径</param>
/// <param name="ComStream">ZIP文件输出流</param>
/// <param name="strSubPath">子目录路径</param>
private static void CompressionZipDirectory(string strRootPath, ZipOutputStream ComStream, string strSubPath)
{
    try
    {
        //创建当前文件夹
        ZipEntry zipEntry = new ZipEntry(Path.Combine(strSubPath, Path.GetFileName(strRootPath) + "/"));
        ComStream.PutNextEntry(zipEntry);
        ComStream.Flush();
        //遍历压缩目录
        foreach (string strFolder in Directory.GetDirectories(strRootPath))
        {
            CompressionZipDirectory(strFolder, ComStream, Path.Combine(strSubPath, Path.GetFileName(strRootPath)));
        }
        //遍历压缩文件
        foreach (string strFileName in Directory.GetFiles(strRootPath))
        {
            FileStream fileStream = File.OpenRead(strFileName);
            byte[] btsLength = new byte[fileStream.Length];
            fileStream.Read(btsLength, 0, btsLength.Length);
            zipEntry = new ZipEntry(Path.Combine(strSubPath, Path.GetFileName(strRootPath) + "/" + Path.GetFileName(strFileName)));
            ComStream.PutNextEntry(zipEntry);
            ComStream.Write(btsLength, 0, btsLength.Length);
            if (fileStream != null)
            {
                fileStream.Close();
                fileStream.Dispose();
            }
        }
        if (zipEntry != null)
        {
            zipEntry = null;
        }
        GC.Collect();
        GC.Collect(1);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
    }
}

/// <summary>
/// 解压缩ZIP文件
/// </summary>
/// <param name="strZipPath">ZIP压缩文件保存位置</param>
/// <param name="strDeCompressionPath">需要解压到的指定位置</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeCompressionZip(string strZipPath, string strDeCompressionPath)
{
    return DeCompressionZip(strZipPath, strDeCompressionPath, string.Empty);
}

/// <summary>
/// 解压缩ZIP文件(加密)
/// </summary>
/// <param name="strZipPath">ZIP压缩文件保存位置</param>
/// <param name="strDeCompressionPath">需要解压到的指定位置</param>
/// <param name="strPassword">压缩文件密码</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeCompressionZip(string strZipPath, string strDeCompressionPath, string strPassword)
{
    try
    {
        if (string.IsNullOrEmpty(strZipPath) || !File.Exists(strZipPath))
        {
            return false;
        }
        ZipInputStream inputStream = new ZipInputStream(File.OpenRead(strZipPath));
        //压缩文件解密
        if (!string.IsNullOrEmpty(strPassword))
        {
            inputStream.Password = strPassword;
        }
        ZipEntry zipEntry = null;
        while ((zipEntry = inputStream.GetNextEntry()) != null)
        {
            if (!string.IsNullOrEmpty(zipEntry.Name))
            {
                string strFileName = Path.Combine(strDeCompressionPath, zipEntry.Name);
                strFileName = strFileName.Replace('/', '\\');
                if (strFileName.EndsWith("\\"))
                {
                    Directory.CreateDirectory(strFileName);
                }
                else
                {
                    FileStream fileStream = null;
                    int intSize = 2048;
                    byte[] btsData = new byte[intSize];
                    while (true)
                    {
                        intSize = inputStream.Read(btsData, 0, btsData.Length);
                        if (fileStream == null)
                        {
                            fileStream = File.Create(strFileName);
                        }
                        if (intSize > 0)
                        {
                            fileStream.Write(btsData, 0, btsData.Length);
                        }
                        else
                        {
                            break;
                        }
                    }
                    if (fileStream != null)
                    {
                        fileStream.Close();
                        fileStream.Dispose();
                    }
                }
            }
        }
        if (zipEntry != null)
        {
            zipEntry = null;
        }
        if (inputStream != null)
        {
            inputStream.Close();
            inputStream.Dispose();
        }
        GC.Collect();
        GC.Collect(1);
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
```

##### 7-ZIP：
[ZIP7Helper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/FileIO.Helper/ZIP/ZIP7Helper.cs) 动态引用 7z.dll 库
``` CSharp
/// <summary>
/// 获得当前系统X86架构7ZIP类库路径
/// </summary>
public static string strX86_DllPath
{
    get
    {
        return string.Format(@"{0}\x86\7z.dll", System.Environment.CurrentDirectory);
    }
}

/// <summary>
/// 获得当前系统X64架构7ZIP类库路径
/// </summary>
public static string strX64_DllPath
{
    get
    {
        return string.Format(@"{0}\x64\7z.dll", System.Environment.CurrentDirectory);
    }
}

/// <summary>
/// 动态链接7ZIP类库
/// </summary>
private static void SetLibraryPath7z()
{
    //动态链接7ZIP类库
    if (IntPtr.Size == 8)
    {
        SevenZipExtractor.SetLibraryPath(strX64_DllPath);
    }
    else
    {
        SevenZipExtractor.SetLibraryPath(strX86_DllPath);
    }
}

/// <summary>
/// 压缩7-ZIP文件
/// </summary>
/// <param name="strZipPath">ZIP压缩文件保存位置</param>
/// <param name="fileFullNames">需要压缩的文件</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool Compression7Zip(string strZipPath, params string[] fileFullNames)
{
    try
    {
        //动态链接7ZIP类库
        SetLibraryPath7z();
        //默认格式为(*.7z)
        strZipPath = Path.ChangeExtension(strZipPath, "7z");
        //压缩7-ZIP文件
        SevenZipCompressor sevenZipCompressor = new SevenZipCompressor();
        //压缩等级(默认正常值)
        sevenZipCompressor.CompressionLevel = CompressionLevel.Normal;
        //压缩格式(默认7z压缩)
        sevenZipCompressor.ArchiveFormat = OutArchiveFormat.SevenZip;
        //是否保持目录结构(默认为true)
        sevenZipCompressor.DirectoryStructure = true;
        //是否包含空目录(默认true)  
        sevenZipCompressor.IncludeEmptyDirectories = true;
        //压缩目录时是否使用顶层目录(默认false)  
        sevenZipCompressor.PreserveDirectoryRoot = false;
        //加密7z头(默认false)  
        sevenZipCompressor.EncryptHeaders = false;
        //文件加密算法
        sevenZipCompressor.ZipEncryptionMethod = ZipEncryptionMethod.ZipCrypto;
        //尽快压缩(不会触发*Started事件,仅触发*Finished事件)  
        sevenZipCompressor.FastCompression = false;
        //压缩文件
        sevenZipCompressor.CompressFiles(strZipPath, fileFullNames);
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 压缩7-ZIP文件(加密)
/// </summary>
/// <param name="strZipPath">ZIP压缩文件保存位置</param>
/// <param name="strPassword">压缩文件密码</param>
/// <param name="fileFullNames">需要压缩的文件</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool Compression7Zip(string strZipPath, string strPassword, params string[] fileFullNames)
{
    try
    {
        //动态链接7ZIP类库
        SetLibraryPath7z();
        //默认格式为(*.7z)
        strZipPath = Path.ChangeExtension(strZipPath, "7z");
        //压缩7-ZIP文件
        SevenZipCompressor sevenZipCompressor = new SevenZipCompressor();
        //压缩等级(默认正常值)
        sevenZipCompressor.CompressionLevel = CompressionLevel.Normal;
        //压缩格式(默认7z压缩)
        sevenZipCompressor.ArchiveFormat = OutArchiveFormat.SevenZip;
        //是否保持目录结构(默认为true)
        sevenZipCompressor.DirectoryStructure = true;
        //是否包含空目录(默认true)  
        sevenZipCompressor.IncludeEmptyDirectories = true;
        //压缩目录时是否使用顶层目录(默认false)  
        sevenZipCompressor.PreserveDirectoryRoot = false;
        //加密7z头(默认false)  
        sevenZipCompressor.EncryptHeaders = true;
        //文件加密算法
        sevenZipCompressor.ZipEncryptionMethod = ZipEncryptionMethod.ZipCrypto;
        //尽快压缩(不会触发*Started事件,仅触发*Finished事件)  
        sevenZipCompressor.FastCompression = false;
        //压缩文件
        sevenZipCompressor.CompressFilesEncrypted(strZipPath, strPassword, fileFullNames);
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 压缩7-ZIP文件夹
/// </summary>
/// <param name="strZipPath">ZIP压缩文件夹</param>
/// <param name="strDirectory">需要压缩的文件夹</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool Compression7ZipDirectory(string strZipPath, string strDirectory)
{
    try
    {
        //动态链接7ZIP类库
        SetLibraryPath7z();
        //默认格式为(*.7z)
        strZipPath = Path.ChangeExtension(strZipPath, "7z");
        //压缩7-ZIP文件
        SevenZipCompressor sevenZipCompressor = new SevenZipCompressor();
        //压缩等级(默认正常值)
        sevenZipCompressor.CompressionLevel = CompressionLevel.Normal;
        //压缩格式(默认7z压缩)
        sevenZipCompressor.ArchiveFormat = OutArchiveFormat.SevenZip;
        //是否保持目录结构(默认为true)
        sevenZipCompressor.DirectoryStructure = true;
        //是否包含空目录(默认true)  
        sevenZipCompressor.IncludeEmptyDirectories = true;
        //压缩目录时是否使用顶层目录(默认false)  
        sevenZipCompressor.PreserveDirectoryRoot = false;
        //加密7z头(默认false)  
        sevenZipCompressor.EncryptHeaders = false;
        //文件加密算法
        sevenZipCompressor.ZipEncryptionMethod = ZipEncryptionMethod.ZipCrypto;
        //尽快压缩(不会触发*Started事件,仅触发*Finished事件)  
        sevenZipCompressor.FastCompression = false;
        //压缩文件
        sevenZipCompressor.CompressDirectory(strDirectory, strZipPath);
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 压缩7-ZIP文件夹(加密)
/// </summary>
/// <param name="strZipPath">ZIP压缩文件夹</param>
/// <param name="strPassword">压缩文件密码</param>
/// <param name="strDirectory">需要压缩的文件夹</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool Compression7ZipDirectory(string strZipPath, string strPassword, string strDirectory)
{
    try
    {
        //动态链接7ZIP类库
        SetLibraryPath7z();
        //默认格式为(*.7z)
        strZipPath = Path.ChangeExtension(strZipPath, "7z");
        //压缩7-ZIP文件
        SevenZipCompressor sevenZipCompressor = new SevenZipCompressor();
        //压缩等级(默认正常值)
        sevenZipCompressor.CompressionLevel = CompressionLevel.Normal;
        //压缩格式(默认7z压缩)
        sevenZipCompressor.ArchiveFormat = OutArchiveFormat.SevenZip;
        //是否保持目录结构(默认为true)
        sevenZipCompressor.DirectoryStructure = true;
        //是否包含空目录(默认true)  
        sevenZipCompressor.IncludeEmptyDirectories = true;
        //压缩目录时是否使用顶层目录(默认false)  
        sevenZipCompressor.PreserveDirectoryRoot = false;
        //加密7z头(默认false)  
        sevenZipCompressor.EncryptHeaders = true;
        //文件加密算法
        sevenZipCompressor.ZipEncryptionMethod = ZipEncryptionMethod.ZipCrypto;
        //尽快压缩(不会触发*Started事件,仅触发*Finished事件)  
        sevenZipCompressor.FastCompression = false;
        //压缩文件
        sevenZipCompressor.CompressDirectory(strDirectory, strZipPath, strPassword);
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 解压缩7-ZIP文件
/// </summary>
/// <param name="strZipPath">ZIP压缩文件保存位置</param>
/// <param name="strDeCompressionPath">需要解压到的指定位置</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeCompression7Zip(string strZipPath, string strDeCompressionPath)
{
    try
    {
        if (string.IsNullOrEmpty(strZipPath) || !File.Exists(strZipPath))
        {
            return false;
        }
        //动态链接7ZIP类库
        SetLibraryPath7z();
        //创建目录
        if (!Directory.Exists(strDeCompressionPath))
        {
            Directory.CreateDirectory(strDeCompressionPath);
        }
        //解压数据
        SevenZipExtractor sevenZipExtractor = new SevenZipExtractor(strZipPath);
        foreach (ArchiveFileInfo itemArchiveFileInfo in sevenZipExtractor.ArchiveFileData)
        {
            sevenZipExtractor.ExtractFiles(strDeCompressionPath, itemArchiveFileInfo.Index);
        }
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 解压缩7-ZIP文件(加密)
/// </summary>
/// <param name="strZipPath">ZIP压缩文件保存位置</param>
/// <param name="strDeCompressionPath">需要解压到的指定位置</param>
/// <param name="strPassword">压缩文件密码</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DeCompression7Zip(string strZipPath, string strDeCompressionPath, string strPassword)
{
    try
    {
        if (string.IsNullOrEmpty(strZipPath) || !File.Exists(strZipPath))
        {
            return false;
        }
        //动态链接7ZIP类库
        SetLibraryPath7z();
        //创建目录
        if (!Directory.Exists(strDeCompressionPath))
        {
            Directory.CreateDirectory(strDeCompressionPath);
        }
        //解压数据
        SevenZipExtractor sevenZipExtractor = new SevenZipExtractor(strZipPath, strPassword);
        foreach (ArchiveFileInfo itemArchiveFileInfo in sevenZipExtractor.ArchiveFileData)
        {
            sevenZipExtractor.ExtractFiles(strDeCompressionPath, itemArchiveFileInfo.Index);
        }
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
```
##### GZIP：
[GZIPHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/FileIO.Helper/ZIP/GZIPHelper.cs) 引用 [System.IO.Compression.GZipStream](https://docs.microsoft.com/zh-cn/dotnet/api/system.io.compression.gzipstream?redirectedfrom=MSDN&view=netframework-4.8) 库
``` CSharp
/// <summary>
/// 压缩GZIP数据
/// </summary>
/// <param name="bytesSourceData">源数据</param>
/// <returns>压缩数据</returns>
public static byte[] CompressionGZIP(byte[] bytesSourceData)
{
    try
    {
        MemoryStream memoryStream = new MemoryStream();
        GZipStream compressedzipStream = new GZipStream(memoryStream, CompressionMode.Compress, true);
        compressedzipStream.Write(bytesSourceData, 0, bytesSourceData.Length);
        compressedzipStream.Close();
        return memoryStream.ToArray();
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 解压缩GZIP数据
/// </summary>
/// <param name="bytesSourceData">源数据</param>
/// <returns>解压缩数据</returns>
public static byte[] DeCompressionGZIP(byte[] bytesSourceData)
{
    try
    {
        MemoryStream memoryStream = new MemoryStream(bytesSourceData);
        GZipStream compressedzipStream = new GZipStream(memoryStream, CompressionMode.Decompress);
        MemoryStream outBuffer = new MemoryStream();
        byte[] block = new byte[1024];
        while (true)
        {
            int bytesRead = compressedzipStream.Read(block, 0, block.Length);
            if (bytesRead <= 0)
                break;
            else
                outBuffer.Write(block, 0, bytesRead);
        }
        compressedzipStream.Close();
        return outBuffer.ToArray();
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 压缩String类型GZIP数据
/// </summary>
/// <param name="strSourceData">源数据</param>
/// <returns>压缩数据(Base64)</returns>
public static string CompressionStringGZIP(string strSourceData)
{
    try
    {
        if (!string.IsNullOrEmpty(strSourceData))
        {
            byte[] rawData = Encoding.UTF8.GetBytes(strSourceData);
            byte[] zippedData = CompressionGZIP(rawData);
            return Convert.ToBase64String(zippedData);
        }
        else
        {
            return string.Empty;
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 解压缩String类型GZIP数据
/// </summary>
/// <param name="strSourceData">源数据(Base64)</param>
/// <returns>解压缩数据</returns>
public static string DeCompressionStringGZIP(string strSourceData)
{
    try
    {
        if (!string.IsNullOrEmpty(strSourceData))
        {
            byte[] zippedData = Convert.FromBase64String(strSourceData.ToString());
            return Encoding.UTF8.GetString(DeCompressionGZIP(zippedData));
        }
        else
        {
            return string.Empty;
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}
```