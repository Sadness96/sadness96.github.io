---
title: RDP Wrapper
date: 2021-04-22 22:10:00
tags: [software,rdpwrap]
categories: Software
---
### Windows 家庭版使用 RDP 远程
<!-- more -->
### 简介
Windows 10 家庭版不支持远程桌面([Remote Desktop Connection(RDP)](https://support.microsoft.com/en-us/windows/how-to-use-remote-desktop-5fe128d5-8fb1-7a23-3b8a-41e636865e8c))功能，微软对其进行了限制，但是可以通过开源项目 [RDP Wrapper](https://github.com/asmtron/rdpwrap) 在功能简化的系统上启用远程桌面主机支持和并发RDP会话。

### 使用说明
#### 下载最新发布版本
下载安装或解压缩软件 [RDP Wrapper Releases](https://github.com/stascorp/rdpwrap/releases/)

#### 安装服务
以管理员权限运行 Install.bat 文件。
安装成功后 "C:\Program Files\RDP Wrapper" 包含配置文件。
<img src="https://sadness96.github.io/images/blog/software-RDPWrapper/RDPWrapperInstall.png"/>

#### 验证安装
##### 运行 RDPConf.exe
显示 Service state: Running
显示 Listener state: Listening [fully supported]
则为安装成功
<img src="https://sadness96.github.io/images/blog/software-RDPWrapper/RDPWrapperListenin.png"/>

##### 或运行 RDPCheck.exe
测试远程到本地，显示为远程自身则为安装成功
<img src="https://sadness96.github.io/images/blog/software-RDPWrapper/RDPWrapperChecker.png"/>

#### 异常错误
##### Listener state 提示：Not listening
由于配置文件中缺少当前版本的配置，版本由 "C:\Windows\System32\termsrv.dll" 文件而来，同 Windows 系统一起更新。
<img src="https://sadness96.github.io/images/blog/software-RDPWrapper/RDPWrapperNotListenin.png"/>

可从 RDP Wrapper Configurgation 中查看文件版本，例如当前版本为：10.0.19041.789，则 "C:\Program Files\RDP Wrapper\rdpwrap.ini" 文件中需包含以下内容，如不存在内容则下载最新版 rdpwrap.ini 文件，关闭 TermService 服务并替换，可参考以下任意最新文件：
https://raw.githubusercontent.com/saurav-biswas/rdpwrap-1/master/res/rdpwrap.ini
https://raw.githubusercontent.com/asmtron/rdpwrap/master/res/rdpwrap.ini
https://raw.githubusercontent.com/sebaxakerhtc/rdpwrap.ini/master/rdpwrap.ini
https://raw.githubusercontent.com/affinityv/INI-RDPWRAP/master/rdpwrap.ini
https://raw.githubusercontent.com/DrDrrae/rdpwrap/master/res/rdpwrap.ini
``` ini
[10.0.19041.789]
LocalOnlyPatch.x86=1
LocalOnlyOffset.x86=B59D9
LocalOnlyCode.x86=jmpshort
LocalOnlyPatch.x64=1
LocalOnlyOffset.x64=88F41
LocalOnlyCode.x64=jmpshort
SingleUserPatch.x86=1
SingleUserOffset.x86=3BC45
SingleUserCode.x86=nop
SingleUserPatch.x64=1
SingleUserOffset.x64=0CA4C
SingleUserCode.x64=Zero
DefPolicyPatch.x86=1
DefPolicyOffset.x86=3E7C9
DefPolicyCode.x86=CDefPolicy_Query_eax_ecx
DefPolicyPatch.x64=1
DefPolicyOffset.x64=18A15
DefPolicyCode.x64=CDefPolicy_Query_eax_rcx
SLInitHook.x86=1
SLInitOffset.x86=67BF8
SLInitFunc.x86=New_CSLQuery_Initialize
SLInitHook.x64=1
SLInitOffset.x64=1D5BC
SLInitFunc.x64=New_CSLQuery_Initialize

[10.0.19041.789-SLInit]
bInitialized.x86      =D0954
bServerSku.x86        =D0958
lMaxUserSessions.x86  =D095C
bAppServerAllowed.x86 =D0964
bRemoteConnAllowed.x86=D096C
bMultimonAllowed.x86  =D0970
ulMaxDebugSessions.x86=D0974
bFUSEnabled.x86       =D0978
bInitialized.x64      =106028
bServerSku.x64        =10602C
lMaxUserSessions.x64  =106030
bAppServerAllowed.x64 =106038
bRemoteConnAllowed.x64=106040
bMultimonAllowed.x64  =106044
ulMaxDebugSessions.x64=106048
bFUSEnabled.x64       =10604C
```