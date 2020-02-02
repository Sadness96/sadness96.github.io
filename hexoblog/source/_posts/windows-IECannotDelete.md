---
title: Win10 无法删除 IE 桌面图标
date: 2017-07-22 11:54:00
tags: [windows,registry]
categories: Windows
---
### 通过编辑注册表以删除默认的顽固图标
<!-- more -->
#### 简介
有一些旧版本 Win10 安装后 IE 图标顽固无法删除，现微软应该已经修复。
#### 解决方法
编辑以下文本保存为 .reg 后缀
双击运行即可
``` reg
Windows Registry Editor Version 5.00

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace]
"MonitorRegistry"=dword:00000001

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\DelegateFolders]

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\DelegateFolders\{F5FB2C77-0E2F-4A16-A381-3E560C68BC83}]
@="Removable Drives"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{031E4825-7B94-4dc3-B131-E946B44C8DD5}]
@="UsersLibraries"
"Removal Message"="@shell32.dll,-9047"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{04731B67-D933-450a-90E6-4ACD2E9408FE}]
@="CLSID_SearchFolder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{11016101-E366-4D22-BC06-4ADA335C892B}]
@="IE History and Feeds Shell Data Source for Windows Search"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{138508bc-1e03-49ea-9c8f-ea9e1d05d65d}]
@="MAPI Shell Folder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{26EE0668-A00A-44D7-9371-BEB064C98683}]
@="ControlPanelHome"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{2F6CE85C-F9EE-43CA-90C7-8A9BD53A2467}]
@="File History Data Source"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{4336a54d-038b-4685-ab02-99bb52d3fb8b}]
@="Public folder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{450D8FBA-AD25-11D0-98A8-0800361B1103}]
@="Documents"
"Removal Message"="@mydocs.dll,-900"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{5399E694-6CE5-4D6C-8FCE-1D8870FDCBA0}]
@="ControlPanelStartupPage"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{59031a47-3f72-44a7-89c5-5595fe6b30ee}]
@="UsersFiles"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{5b934b42-522b-4c34-bbfe-37a3ef7b9c90}]
@="This Device Folder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{645FF040-5081-101B-9F08-00AA002F954E}]
@="Recycle Bin"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{64693913-1c21-4f30-a98f-4e52906d3b56}]
@="AppInstanceFolder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{89D83576-6BD1-4c86-9454-BEB04E94C819}]
@="MAPI Folder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{8FD8B88D-30E1-4F25-AC2B-553D3D65F0EA}]
@="DXP"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{9343812e-1c37-4a49-a12e-4b2d810d956b}]
@="Search Home"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{98F275B4-4FFF-11E0-89E2-7B86DFD72085}]
@="CLSID_StartMenuLauncherProviderFolder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{a00ee528-ebd9-48b8-944a-8942113d46ac}]
@="CLSID_StartMenuCommandingProviderFolder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{B4FB3F98-C1EA-428d-A78A-D1F5659CBA93}]
@="Other Users"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{B5EAD1DB-7C18-4954-8820-01733BC08C82}]
@="Internet Explorer"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{BD7A2E7B-21CB-41b2-A086-B309680C6B7E}]
@="CSC Folder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{daf95313-e44d-46af-be1b-cbacea2c3065}]
@="CLSID_StartMenuProviderFolder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{DB19096C-5365-4164-A246-59FEFF9D8062}]
@="Nameext"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{e345f35f-9397-435c-8f95-4e922c26259e}]
@="CLSID_StartMenuPathCompleteProviderFolder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{ED228FDF-9EA8-4870-83b1-96b02CFE0D52}]
"Removal Message"="@gameux.dll,-10038"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{EDC978D6-4D53-4b2f-A265-5805674BE568}]
@="StreamBackedFolder"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{F02C1A0D-BE21-4350-88B0-7367FC96EF3C}]
@="Computers and Devices"

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Desktop\NameSpace\{f8278c54-a712-415b-b593-b77a2be0dda9}]
@="This Device Folder"
```