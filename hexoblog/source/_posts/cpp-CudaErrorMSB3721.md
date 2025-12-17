---
title: C++ Cuda 报错 MSB3721
date: 2025-12-17 14:28:40
tags: [c++,cuda]
categories: C++
---
### 更新 Visual Studio 后编译 CUDA 项目报错
<!-- more -->
### 简介
当前安装 CUDA 版本 11.8，给 Visual Studio 更新到版本 17.14.22 后无法编译，报错 MSB3721，是由于 CUDA 对 Microsoft C/C++ 编译器的版本号限制。
与去年遇到 [C++ Cuda 报错 C1189 MSB372](/blog/2024/05/29/cpp-CudaErrorMSB372/) 的问题相似，但却是由于 CUDA 11.8 不支持你当前使用的 MSVC 14.44（VS2022 最新工具集）导致。

#### 错误信息
``` cmd
错误 static assertion failed with "error STL1002: Unexpected compiler version, expected CUDA 12.4 or newer." C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\include\yvals_core.h 902
错误 MSB3721 命令“"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -gencode=arch=compute_52,code=\"sm_52,compute_52\" --use-local-env -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64" -x cu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include" -G --keep-dir x64\Debug -maxrregcount=0 --machine 64 --compile -cudart static -g -D_DEBUG -DXXX_EXPORTS -D_WINDOWS -D_USRDLL -D_WINDLL -D_UNICODE -DUNICODE -Xcompiler "/EHsc /W3 /nologo /Od /Fdx64\Debug\vc143.pdb /FS /Zi /RTC1 /MDd " -o  "XXX"”已退出，返回代码为 1。 XXX C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 11.8.targets 785
```

### 解决方法
在 Visual Studio Installer 中搜索单个组件 MSVC，安装 CUDA 11.8 最后支持的 MSVC 版本：
* MSVC v143 - VS2022 C++ x64/x86 生成工具(v14.36-17.6)(不受支持)

我这里安装后就可用了，如还是不好使，需要确定编译器是通过这个版本生成的。