---
title: 调用神思二代身份证读卡器
date: 2019-04-06 21:15:31
tags: [c++,c#,sdses]
categories: C++
---
<img src="https://sadness96.github.io/images/blog/cpp-Sdses/s100-1.jpg"/>

<!-- more -->
### 基于神思二代身份证读卡器做二次开发
#### 简介
公司中项目[人证合一核查系统](https://baike.baidu.com/item/%E4%BA%BA%E8%AF%81%E5%90%88%E4%B8%80/19776127?fr=aladdin)需要，使用[神思二代身份证读卡器](http://www.sdses.com/)二次开发集成。
由于神思二代证[SKD](https://baike.baidu.com/item/sdk/7815680?fr=aladdin)只提供了C++/Java接口，项目还是采用C#作为主要开发语言，使用WPF开发界面，所以采用 [C#/C++ 混合编程](/blog/2018/08/01/cpp-HybridCSharp/) 的方式开发。
#### 封装代码
由官方提供的C++SDK二次封装为非托管动态链接库暴露接口给C#端调用
读卡器读取身份证照片存为 RdCard.dll 库目录下 zp.bmp 文件
``` C++
#include "stdafx.h"
#include "2ndCardReader.h"
// 主符号
#include "resource.h"
#include "Ucmd.h"
#include <fstream>
#include <atlimage.h>
#pragma warning(disable:4996)
using namespace std;

typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned long  DWORD;
typedef long LONG;

//位图文件头文件定义
typedef struct
{
	DWORD bfSize;		//文件大小
	WORD bfReserved1;	//保留字
	WORD bfReserved2;	//保留字
	DWORD bfOffBits;	//实际位图数据偏移字节数=前三个部分长度之和
}ClBITMAPHEADER;

//信息头BITMAPINFOHEADER
typedef struct
{
	DWORD	biSize;				//指定此结构体长度40
	LONG	biWidth;
	LONG	biHeight;
	WORD	biPlanes;			//平面数 为1
	WORD	biBitCount;			//采用颜色位数
	DWORD	biCompression;		//压缩方式
	DWORD	biSizeImage;		//实际位图占用字节数
	LONG	biXPelsPerMeter;	//x方向分辨率
	LONG	biYPelsPerMeter;	//y方向分辨率
	DWORD	biClrUsed;			//使用的颜色数
	DWORD	biClrImportant;		//重要的颜色数
}ClBITMAPINFOHEADER;


typedef struct
{
	unsigned char rgbBlue;
	unsigned char rgbGreen;
	unsigned char rgbRed;
	unsigned char rgbReserved;
}ClRGBQUAD;

typedef int(__stdcall *_UCommand1)(unsigned char* pCmd, int* parg0, int* parg1, int* parg2);
typedef int(__stdcall *_GetAddr)(char* pbuf);
typedef int(__stdcall *_GetBegin)(char* pbuf);
typedef int(__stdcall *_GetName)(char* pbuf);
typedef int(__stdcall *_GetSex)(char* pbuf);
typedef int(__stdcall *_GetSexGB)(char* pbuf);
typedef int(__stdcall *_GetFolk)(char* pbuf);
typedef int(__stdcall *_GetFolkGB)(char* pbuf);
typedef int(__stdcall *_GetIDNum)(char* pbuf);
typedef int(__stdcall *_GetDep)(char* pbuf);
typedef int(__stdcall *_GetBirth)(char* pbuf);
typedef int(__stdcall *_GetEnd)(char* pbuf);
typedef int(__stdcall *_GetNewAddr)(char* pbuf);
typedef int(__stdcall *_FID_GetEnName)(char* pbuf);
typedef int(__stdcall *_FID_GetSex)(char* pbuf);
typedef int(__stdcall *_FID_GetSexGB)(char* pbuf);
typedef int(__stdcall *_FID_GetIDNum)(char* pbuf);
typedef int(__stdcall *_FID_GetNationality)(char* pbuf);
typedef int(__stdcall *_FID_GetChNationality)(char* pbuf);
typedef int(__stdcall *_FID_GetChName)(char* pbuf);
typedef int(__stdcall *_FID_GetBegin)(char* pbuf);
typedef int(__stdcall *_FID_GetEnd)(char* pbuf);
typedef int(__stdcall *_FID_GetBirth)(char* pbuf);
typedef int(__stdcall *_FID_GetVersion)(char* pbuf);
typedef int(__stdcall *_FID_GetDep)(char* pbuf);
typedef int(__stdcall *_GetSAMIDToStr)(char* id);

_UCommand1				UCommand1;
_GetAddr				GetAddr;
_GetBegin				GetBegin;
_GetName				GetName;
_GetSex					GetSex;
_GetSexGB				GetSexGB;
_GetFolk				GetFolk;
_GetFolkGB				GetFolkGB;
_GetIDNum				GetIDNum;
_GetDep					GetDep;
_GetBirth				GetBirth;
_GetEnd					GetEnd;
_GetNewAddr				GetNewAddr;
_FID_GetEnName			FID_GetEnName;
_FID_GetSex				FID_GetSex;
_FID_GetSexGB			FID_GetSexGB;
_FID_GetIDNum			FID_GetIDNum;
_FID_GetNationality		FID_GetNationality;
_FID_GetChNationality	FID_GetChNationality;
_FID_GetChName			FID_GetChName;
_FID_GetBegin			FID_GetBegin;
_FID_GetEnd				FID_GetEnd;
_FID_GetBirth			FID_GetBirth;
_FID_GetVersion			FID_GetVersion;
_FID_GetDep				FID_GetDep;
_GetSAMIDToStr			GetSAMIDToStr;

extern "C" char* TrimStr(char *str)
{
	char *head = str;
	while (*head == ' ')
	{
		++head;
	}

	char *end = head + strlen(head) - 1;
	while (*end == ' ')
	{
		--end;
	}
	*(end + 1) = 0;
	strcpy(str, head);
	return str;
}

extern "C" MY2NDCARDREADER_API int fn2ndCardReaderInfo(IDInfo* pIDInfo)
{
	int i = LoadDll();
	if (i != 1)
	{
		return -10;
	}
	HINSTANCE hDll = LoadLibraryEx(L"RdCard.dll", 0, LOAD_WITH_ALTERED_SEARCH_PATH);
	if (hDll == NULL)
	{
		return -10;
	}

	UCommand1 = (_UCommand1)GetProcAddress(hDll, "UCommand1");
	GetAddr = (_GetAddr)GetProcAddress(hDll, "GetAddr");
	GetBegin = (_GetBegin)GetProcAddress(hDll, "GetBegin");
	GetName = (_GetName)GetProcAddress(hDll, "GetName");
	GetSex = (_GetSex)GetProcAddress(hDll, "GetSex");
	GetSexGB = (_GetSexGB)GetProcAddress(hDll, "GetSexGB");
	GetFolk = (_GetFolk)GetProcAddress(hDll, "GetFolk");
	GetFolkGB = (_GetFolkGB)GetProcAddress(hDll, "GetFolkGB");
	GetIDNum = (_GetIDNum)GetProcAddress(hDll, "GetIDNum");
	GetDep = (_GetDep)GetProcAddress(hDll, "GetDep");
	GetBirth = (_GetBirth)GetProcAddress(hDll, "GetBirth");
	GetEnd = (_GetEnd)GetProcAddress(hDll, "GetEnd");
	GetNewAddr = (_GetNewAddr)GetProcAddress(hDll, "GetNewAddr");
	FID_GetEnName = (_FID_GetEnName)GetProcAddress(hDll, "FID_GetEnName");
	FID_GetSex = (_FID_GetSex)GetProcAddress(hDll, "FID_GetSex");
	FID_GetSexGB = (_FID_GetSexGB)GetProcAddress(hDll, "FID_GetSexGB");
	FID_GetIDNum = (_FID_GetIDNum)GetProcAddress(hDll, "FID_GetIDNum");
	FID_GetNationality = (_FID_GetNationality)GetProcAddress(hDll, "FID_GetNationality");
	FID_GetChNationality = (_FID_GetChNationality)GetProcAddress(hDll, "FID_GetChNationality");
	FID_GetChName = (_FID_GetChName)GetProcAddress(hDll, "FID_GetChName");
	FID_GetBegin = (_FID_GetBegin)GetProcAddress(hDll, "FID_GetBegin");
	FID_GetEnd = (_FID_GetEnd)GetProcAddress(hDll, "FID_GetEnd");
	FID_GetBirth = (_FID_GetBirth)GetProcAddress(hDll, "FID_GetBirth");
	FID_GetVersion = (_FID_GetVersion)GetProcAddress(hDll, "FID_GetVersion");
	FID_GetDep = (_FID_GetDep)GetProcAddress(hDll, "FID_GetDep");
	GetSAMIDToStr = (_GetSAMIDToStr)GetProcAddress(hDll, "GetSAMIDToStr");

	if (UCommand1 == NULL || GetAddr == NULL || GetBegin == NULL || GetName == NULL || GetSex == NULL || GetSexGB == NULL || GetFolk == NULL || GetFolkGB == NULL || GetIDNum == NULL || GetDep == NULL || GetBirth == NULL || GetEnd == NULL || GetNewAddr == NULL || FID_GetEnName == NULL || FID_GetSex == NULL || FID_GetSexGB == NULL || FID_GetIDNum == NULL || FID_GetNationality == NULL || FID_GetChNationality == NULL || FID_GetChName == NULL || FID_GetBegin == NULL || FID_GetEnd == NULL || FID_GetBirth == NULL || FID_GetVersion == NULL || FID_GetDep == NULL || GetSAMIDToStr == NULL)
	{
		return FALSE;
	}
	//连接设备
	unsigned char cmd = 0x41;
	int para0 = 0, para1 = 8811, para2 = 9986;
	int ret = UCommand1(&cmd, &para0, &para1, &para2);
	if (ret != 62171)
	{
		//MessageBox("设备连接失败，请检查设备是否插好！");
		return -11;
	}

	//验证卡（寻卡）
	cmd = 0x43;
	ret = UCommand1(&cmd, &para0, &para1, &para2);

	//读卡
	cmd = 0x49;
	ret = UCommand1(&cmd, &para0, &para1, &para2);
	if (ret != 62171)
	{
		//MessageBox("读卡失败！");
		return -12;
	}

	GetAddr = (_GetAddr)GetProcAddress(hDll, "GetAddr");
	GetName = (_GetName)GetProcAddress(hDll, "GetName");
	GetSex = (_GetSex)GetProcAddress(hDll, "GetSex");
	GetFolk = (_GetFolkGB)GetProcAddress(hDll, "GetFolkGB");
	GetIDNum = (_GetIDNum)GetProcAddress(hDll, "GetIDNum");
	GetBegin = (_GetBegin)GetProcAddress(hDll, "GetBegin");
	GetEnd = (_GetEnd)GetProcAddress(hDll, "GetEnd");
	GetDep = (_GetDep)GetProcAddress(hDll, "GetDep");
	if (GetAddr == NULL || GetName == NULL || GetSex == NULL || GetFolk == NULL || GetIDNum == NULL || GetBegin == NULL || GetDep == NULL || GetEnd == NULL)
	{
		return -22;
	}

	GetName(pIDInfo->Name);
	TrimStr(pIDInfo->Name);
	GetSexGB(pIDInfo->Gender);
	GetBirth(pIDInfo->BirthDate);
	GetAddr(pIDInfo->Address);
	TrimStr(pIDInfo->Address);
	GetIDNum(pIDInfo->IDNumber);
	GetBegin(pIDInfo->Begin);
	GetEnd(pIDInfo->End);
	GetFolk(pIDInfo->Folk);
	GetDep(pIDInfo->IssuanceAuthority);
	pIDInfo->Nation = "中国";
	//关闭读卡器
	cmd = 0x42;
	UCommand1(&cmd, &para0, &para1, &para2);
	FreeDll();
	return 1;
}
```
#### 调用代码
``` CSharp
[DllImport(@"2ndCardReader.dll", EntryPoint = "fn2ndCardReaderInfo", CallingConvention = CallingConvention.Cdecl)]
public static extern int fn2ndCardReaderInfo(ref IDInfo pIDInfo);
```
``` CSharp
 [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi, Pack = 1)]
public struct IDInfo
{
    //二代证信息
    public IntPtr Name;
    public IntPtr Gender;
    public IntPtr Folk;
    public IntPtr Nation;
    public IntPtr BirthDate;
    public IntPtr Address;
    public IntPtr IDNumber;
    public IntPtr IssuanceAuthority;
    public IntPtr Begin;
    public IntPtr End;
    public IntPtr Image;
}
```
``` CSharp
IDInfo pIDInfo = new IDInfo();

pIDInfo.Name = Marshal.StringToHGlobalAnsi(" ");
pIDInfo.Gender = Marshal.StringToHGlobalAnsi("");
pIDInfo.Nation = Marshal.StringToHGlobalAnsi(" ");
pIDInfo.BirthDate = Marshal.StringToHGlobalAnsi(" ");
pIDInfo.Address = Marshal.StringToHGlobalAnsi(" ");
pIDInfo.IDNumber = Marshal.StringToHGlobalAnsi(" ");
pIDInfo.Image = Marshal.StringToHGlobalAnsi(" ");
pIDInfo.Begin= Marshal.StringToHGlobalAnsi(" ");
pIDInfo.End = Marshal.StringToHGlobalAnsi(" ");
pIDInfo.Folk = Marshal.StringToHGlobalAnsi(" ");
pIDInfo.IssuanceAuthority = Marshal.StringToHGlobalAnsi(" ");
int i = fn2ndCardReaderInfo(ref pIDInfo);
string Name = Marshal.PtrToStringAnsi(pIDInfo.Name);
string Gender = Marshal.PtrToStringAnsi(pIDInfo.Gender);
string Nation = Marshal.PtrToStringAnsi(pIDInfo.Nation);
string BirthDate = Marshal.PtrToStringAnsi(pIDInfo.BirthDate);
string Address = Marshal.PtrToStringAnsi(pIDInfo.Address);
string IDNumber = Marshal.PtrToStringAnsi(pIDInfo.IDNumber);
string IssuanceAuthority = Marshal.PtrToStringAnsi(pIDInfo.IssuanceAuthority);
string Begin = Marshal.PtrToStringAnsi(pIDInfo.Begin);
string End = Marshal.PtrToStringAnsi(pIDInfo.End);
string Folk = Marshal.PtrToStringAnsi(pIDInfo.Folk);
}
```