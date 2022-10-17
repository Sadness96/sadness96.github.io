---
title: 构建 OpenCV
date: 2021-09-16 21:10:30
tags: [software,cmake,opencv]
categories: Software
---
### 从源代码构建支持 CUDA 的 OpenCV
<!-- more -->
### 简介
[OpenCV](https://opencv.org/) 是一个开源的计算机视觉和机器学习软件库，图像算法必备！
编译 OpenCV 一般对相关软件版本都有要求，请谨慎选择版本，本文编译的 OpenCV 版本是 3.4.2。

### 构建环境
* Windows 10
* [Visual Studio 2019](https://visualstudio.microsoft.com/zh-hans/)
* [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
* [CMake 3.17.2](https://github.com/Kitware/CMake/releases/tag/v3.17.2)
* [OpenCV 3.4.2](https://github.com/opencv/opencv/releases/tag/3.4.2)
* [opencv_contrib 3.4.2](https://github.com/opencv/opencv_contrib/releases/tag/3.4.2)

### 构建步骤
#### 预安装软件
1. Visual Studio、CUDA Toolkit、CMake 提前安装好。
1. 下载解压 OpenCV 与 opencv_contrib 源代码并解压。
<img src="https://sadness96.github.io/images/blog/software-BuildOpenCV/1.解压OpenCV源码.jpg"/>

#### 运行 CMake
运行 CMake 选择 OpenCV 源代码路径与编译生成路径
<img src="https://sadness96.github.io/images/blog/software-BuildOpenCV/2.运行CMake.jpg"/>

#### 选择编译环境
点击 Configure 配置选择编译环境
<img src="https://sadness96.github.io/images/blog/software-BuildOpenCV/3.配置选择编译环境.jpg"/>

#### 配置编译内容
点完第一次 Configure 后界面一片红，不用在意，先根据需要配置编译内容
* 勾选 "with_cuda"：支持 CUDA 环境的 OpenCV。
* 勾选 "build_opencv_world"：会把所有的库生成为一个 dll 与 lib，很方便使用，但不建议勾选，如果编译时报错又未完成的编译库，依旧可以生成，但是使用时会报：“无法解析的外部符号”，却又很难找出原因。
* 勾选 "opencv_enable_nonfree"：可以使用具有专利保护的算法。
* 配置 "opencv_extra_modules_path" 为扩展模块的源码路径 ".../opencv_contrib-3.4.2/modules"：可以使用 OpenCV 一些受专利保护算法的扩展模块。

<img src="https://sadness96.github.io/images/blog/software-BuildOpenCV/4.配置编译内容.jpg"/>

#### 生成
点击 Configure 直至没有红色部分，点击 Generate 生成项目
<img src="https://sadness96.github.io/images/blog/software-BuildOpenCV/5.生成.jpg"/>

#### 下载缺失的文件
CMake 构建期间会联网下载一些库，如果下载失败了，此时直接编译代码，有些库会编译失败。找到生成目录下 "CMakeDownloadLog.txt" 文件，里边记录了下载失败的文件名称，以及下载地址与下载位置。
<img src="https://sadness96.github.io/images/blog/software-BuildOpenCV/6.CMakeDownloadLog.jpg"/>

可以手动下载并拷贝到指定位置，或者使用一段简单的代码解析文件内容，批量下载，使用 C# 编写较为容易，想使用其他语言编写自行修改。
``` CSharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace CMakeDownLoadErrorFile
{
    class Program
    {
        static void Main(string[] args)
        {
            // 解决下载异常：未能创建 SSL/TLS 安全通道
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;

            string fileCMakeDownloadLog = @"D:\Software\opencv\CMakeDownloadLog.txt";
            var vFileList = GetFileList(fileCMakeDownloadLog).Where(o => o.StartsWith("do_copy") || o.StartsWith("do_unpack")).ToList();
            for (int i = 0; i < vFileList.Count; i++)
            {
                var vItemSplit = vFileList[i].Split(' ');
                var vItemType = vItemSplit[0];
                var vItemFileName = vItemSplit[1].Replace("\"", "");
                var vItemMD5 = vItemSplit[2].Replace("\"", "");
                var vItemUrl = vItemSplit[3].Replace("\"", "");
                var vItemPath = vItemSplit[4].Replace("\"", "");

                var vSavePath = $"{vItemPath}/{vItemFileName}";
                //如果路径下的文件不存在，自动创建
                string strFolderPath = Path.GetDirectoryName(vSavePath);
                if (!Directory.Exists(strFolderPath))
                {
                    Directory.CreateDirectory(strFolderPath);
                }

                Console.WriteLine($"{DateTime.Now} 下载文件：{vItemFileName}\t({i + 1}/{vFileList.Count})");
                DownLoadFile(vItemUrl, $"{vSavePath}");
            }
            Console.WriteLine("下载完成！");
            Console.ReadKey();
        }

        /// <summary>
        /// 读取TXT文件中的文本(按照每行存到listString中)
        /// </summary>
        /// <param name="strPath">TXT文件路径</param>
        /// <returns>TXT文件中的文本(listString)</returns>
        public static List<string> GetFileList(string strPath)
        {
            string[] strText = null;
            List<string> listText = new List<string>();
            try
            {
                strText = File.ReadAllLines(strPath);
                foreach (string strLine in strText)
                {
                    listText.Add(strLine);
                }
            }
            catch (Exception)
            { }
            return listText;
        }

        /// <summary>
        /// 下载文件
        /// </summary>
        /// <param name="fileUrl">文件 URL</param>
        /// <param name="savePath">保存路径</param>
        public static void DownLoadFile(string fileUrl, string savePath)
        {
            using (var web = new WebClient())
            {
                web.DownloadFile(fileUrl, savePath);
            }
        }
    }
}
```
<img src="https://sadness96.github.io/images/blog/software-BuildOpenCV/7.CMakeDownLoadErrorFile.jpg"/>

下载完成后测试有一处需要手动拷贝文件，否则编译时找不到文件，拷贝：".../{生成目录}/downloads/xfeatures2d" 目录下文件至 ".../opencv_contrib-3.4.2/modules/xfeatures2d/src"。

#### 编译项目
运行 Visual Studio 打开生成目录下 "OpenCV.sln"，点击菜单栏：生成 -> 批生成，勾选 Debug 模式与 Release 模式的 ALL_BUILD 与 INSTALL 项目，点击生成即可生成最完整的项目包，也可根据实际需要勾选，或者直接在项目中右键生成。等待生成完成即可。
<img src="https://sadness96.github.io/images/blog/software-BuildOpenCV/8.编译项目.jpg"/>

#### 结束
如果出现编译失败，可以尝试保证网络完好后再次生成，有些文件似乎还是会通过联网下载。
编译完成后：".../{生成目录}/install" 即为最后生成文件，我这里生成的 install 中的 bin 目录配置环境变量后就可以正常使用，include 与 lib 目录则引用到开发项目中。

##### 项目引用
C++ 项目在引用 OpenCV 附加依赖项时需要区分 Debug 与 Release 库：
其中以 *342d.lib 结尾的文件为 Debug 生成；
其中以 *342.lib 结尾的文件为 Release 生成；
可以使用以下命令生成文件名，方便项目引用时拷贝。
``` cmd
dir *342d.lib > lib_debug_file.txt
dir *342.lib > lib_release_file.txt
```

##### 不同的目录
可能由于软件版本问题，有人生成的目录会是：".../{生成目录}/install/x64/vc15/bin"，环境变量引用即可，似乎没有什么区别。