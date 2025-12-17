---
title: C++ Cuda 报错 C1189 MSB372
date: 2024-05-29 19:46:20
tags: [c++,cuda]
categories: C++
---
### 更新 Visual Studio 后编译 CUDA 项目报错
<!-- more -->
### 简介
当前安装 CUDA 版本 11.8，给 Visual Studio 更新到版本 17.10.0 后无法编译，报错 C1189、MSB372，是由于 CUDA 对 Microsoft C/C++ 编译器的版本号限制。
CUDA 11.8 版本限制的是 VS 2017 - 2022 版本，但是没想到更新到 17.10.0 版本后 C++ 编译器版本号超过了英伟达对版本升级预计的限制。

#### 错误信息
``` cmd
C1189 #error:  -- unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.	
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\crt\host_config.h 153

MSB3721 命令“"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -gencode=arch=compute_52,code=\"sm_52,compute_52\" --use-local-env -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.40.33807\bin\HostX64\x64" -x cu   -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"  -G   --keep-dir x64\Debug  -maxrregcount=0  --machine 64 --compile -cudart static  -g  -D_DEBUG -DVIDEODECODER_EXPORTS -D_WINDOWS -D_USRDLL -D_WINDLL -D_UNICODE -DUNICODE -Xcompiler "/EHsc /W3 /nologo /Od /Fdx64\Debug\vc143.pdb /FS /Zi /RTC1 /MDd " -o .\test.cu.obj ".\test.cu"”已退出，返回代码为 2。	
C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 11.8.targets 785
```

#### 获取当前编译器版本号
获取当前 Microsoft C/C++ 编译器版本号
``` cmd
#include <iostream>

void main()
{
	// Visual Studio 17.10.0 版本为 1940
	std::cout << "MSC_VER: " << _MSC_VER << std::endl;
}
```

#### 对应编译器版本号
通常 Visual Studio 版本对应编译器版本号
``` cmd
Visual Studio 2017 对应的 MSC_VER 值大约是 1910-1916。
Visual Studio 2019 对应的 MSC_VER 值是 1920-1929。
Visual Studio 2022 对应的 MSC_VER 值是 1930 及更高。
```

### 解决方法
[VS](https://visualstudio.microsoft.com/) 当前最新版本为 Visual Studio 2022 17.10.0
[CUDA](https://developer.nvidia.com/cuda-toolkit-archive) 当前最新版本为 CUDA Toolkit 12.5.0
理论上来说直接更新 CUDA 版本即可解决问题，但是暂时还是按照不更新的情况下解决。
根据错误提示，使用 -allow-unsupported-compiler 标志忽略版本检查，但是在开发环境不知道应该如何配置，所以使用修改判断文件解决。

#### 修改文件
根据报错信息找到报错文件地址与行号 host_config.h 153，使用管理员权限修改文件配置。
Visual Studio 17.10.0 版本为 1940，可修改为 _MSC_VER > 1940，或修改为更大值。
``` cpp
#if _MSC_VER < 1910 || _MSC_VER >= 1940

#error -- unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
```