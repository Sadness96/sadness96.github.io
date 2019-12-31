---
title: Nsis 使用介绍
date: 2018-11-24 15:59:51
tags: [software,nsis]
categories: Software
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Nsis/HM-VNISEdit.jpg"/>

### 基于 NSIS 的 Windows 桌面端打包程序
<!-- more -->
#### 简介
现工作中作为全栈开发工程师，不光写服务端B/S也要写桌面端C/S程序，在部署B/S的时候一般是拷贝文件或自动部署到服务器，但是桌面端程序普遍是打包为[安装包(Install pack)](https://baike.baidu.com/item/%E5%AE%89%E8%A3%85%E5%8C%85/7693150?fr=aladdin)在官网提供下载或是直接发送给用户安装升级。一般由质检部门打包测试，最后没有BUG的版本发布。
[NSIS（Nullsoft Scriptable Install System）](https://nsis.sourceforge.io/Main_Page)是一个专业的开源系统，用于创建Windows安装程序。它的设计尽可能小巧灵活，因此非常适合互联网分发。
只做了一个最基础安装包程序，如需定制样式可查阅官方网站。
#### 安装包截图
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Nsis/install1.png"/>
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Nsis/install2.png"/>
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Nsis/install3.png"/>
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/software-Nsis/install4.png"/>

#### 脚本代码
替换脚本中 "#" 开头的部分即可
``` NSIS
; 该脚本使用 HM VNISEdit 脚本编辑器向导产生

; 安装程序初始定义常量
!define PRODUCT_NAME "#系统名称"
!define PRODUCT_VERSION "#系统版本"
!define PRODUCT_PUBLISHER "#公司名称"
!define PRODUCT_WEB_SITE "#公司官网"
!define PRODUCT_DIR_REGKEY "Software\Microsoft\Windows\CurrentVersion\App Paths\#EXE名称"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define PRODUCT_UNINST_ROOT_KEY "HKLM"
!define PRODUCT_STARTMENU_REGVAL "NSIS:StartMenuDir"

SetCompressor lzma

; ------ MUI 现代界面定义 (1.67 版本以上兼容) ------
!include "MUI.nsh"

; MUI 预定义常量
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

; 欢迎页面
!insertmacro MUI_PAGE_WELCOME
; 许可协议页面
!define MUI_LICENSEPAGE_CHECKBOX
!insertmacro MUI_PAGE_LICENSE "许可协议.txt"
; 安装目录选择页面
!insertmacro MUI_PAGE_DIRECTORY
; 开始菜单设置页面
var ICONS_GROUP
!define MUI_STARTMENUPAGE_NODISABLE
!define MUI_STARTMENUPAGE_DEFAULTFOLDER "#系统名称"
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "${PRODUCT_UNINST_ROOT_KEY}"
!define MUI_STARTMENUPAGE_REGISTRY_KEY "${PRODUCT_UNINST_KEY}"
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "${PRODUCT_STARTMENU_REGVAL}"
!insertmacro MUI_PAGE_STARTMENU Application $ICONS_GROUP
; 安装过程页面
!insertmacro MUI_PAGE_INSTFILES
; 安装完成页面
!define MUI_FINISHPAGE_RUN "$INSTDIR\#EXE名称"
!insertmacro MUI_PAGE_FINISH

; 安装卸载过程页面
!insertmacro MUI_UNPAGE_INSTFILES

; 安装界面包含的语言设置
!insertmacro MUI_LANGUAGE "SimpChinese"

; 安装预释放文件
!insertmacro MUI_RESERVEFILE_INSTALLOPTIONS
; ------ MUI 现代界面定义结束 ------

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "#打包文件名称"
InstallDir "$PROGRAMFILES\#系统名称"
InstallDirRegKey HKLM "${PRODUCT_UNINST_KEY}" "UninstallString"
ShowInstDetails show
ShowUnInstDetails show

Section "MainSection" SEC01
  SetOutPath "$INSTDIR"
  SetOverwrite ifnewer
  File "#打包文件"
  File /r "#打包特定后缀 *.*"
  SetOverwrite off
  File "#打包排除特定后缀 *.*"

; 创建开始菜单快捷方式
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
  CreateDirectory "$SMPROGRAMS\$ICONS_GROUP"
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\#系统名称.lnk" "$INSTDIR\#EXE名称"
  CreateShortCut "$DESKTOP\#系统名称.lnk" "$INSTDIR\KeDun.Shell.exe"
  !insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

Section -AdditionalIcons
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
  CreateShortCut "$SMPROGRAMS\$ICONS_GROUP\Uninstall.lnk" "$INSTDIR\uninst.exe"
  !insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

Section -Post
  WriteUninstaller "$INSTDIR\uninst.exe"
  WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR\#EXE名称"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayName" "$(^Name)"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninst.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\KeDun.Shell.exe"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
  WriteRegStr ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
SectionEnd

; 安装net45
Function CheckAndDownloadDotNet45
	# Let's see if the user has the .NET Framework 4.5 installed on their system or not
	# Remember: you need Vista SP2 or 7 SP1.  It is built in to Windows 8, and not needed
	# In case you're wondering, running this code on Windows 8 will correctly return is_equal
	# or is_greater (maybe Microsoft releases .NET 4.5 SP1 for example)

	# Set up our Variables
	Var /GLOBAL dotNET45IsThere
	Var /GLOBAL dotNET_CMD_LINE
	Var /GLOBAL EXIT_CODE

        # We are reading a version release DWORD that Microsoft says is the documented
        # way to determine if .NET Framework 4.5 is installed
	ReadRegDWORD $dotNET45IsThere HKLM "SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full" "Release"
	IntCmp $dotNET45IsThere 378389 is_equal is_less is_greater

	is_equal:
		Goto done_compare_not_needed
	is_greater:
		# Useful if, for example, Microsoft releases .NET 4.5 SP1
		# We want to be able to simply skip install since it's not
		# needed on this system
		Goto done_compare_not_needed
	is_less:
		Goto done_compare_needed

	done_compare_needed:
		#.NET Framework 4.5 install is *NEEDED*

		# Microsoft Download Center EXE:
		# Web Bootstrapper: http://go.microsoft.com/fwlink/?LinkId=225704
		# Full Download: http://go.microsoft.com/fwlink/?LinkId=225702

		# Setup looks for components\dotNET45Full.exe relative to the install EXE location
		# This allows the installer to be placed on a USB stick (for computers without internet connections)
		# If the .NET Framework 4.5 installer is *NOT* found, Setup will connect to Microsoft's website
		# and download it for you

		# Reboot Required with these Exit Codes:
		# 1641 or 3010

		# Command Line Switches:
		# /showrmui /passive /norestart

		# Silent Command Line Switches:
		# /q /norestart


		# Let's see if the user is doing a Silent install or not
		IfSilent is_quiet is_not_quiet

		is_quiet:
			StrCpy $dotNET_CMD_LINE "/q /norestart"
			Goto LookForLocalFile
		is_not_quiet:
			StrCpy $dotNET_CMD_LINE "/showrmui /passive /norestart"
			Goto LookForLocalFile

		LookForLocalFile:
			# Let's see if the user stored the Full Installer
			IfFileExists "$EXEPATH\components\dotNET45Full.exe" do_local_install do_network_install

			do_local_install:
				# .NET Framework found on the local disk.  Use this copy

				ExecWait '"$EXEPATH\components\dotNET45Full.exe" $dotNET_CMD_LINE' $EXIT_CODE
				Goto is_reboot_requested

			# Now, let's Download the .NET
			do_network_install:

				Var /GLOBAL dotNetDidDownload
				NSISdl::download "http://go.microsoft.com/fwlink/?LinkId=225704" "$TEMP\dotNET45Web.exe" $dotNetDidDownload

				StrCmp $dotNetDidDownload success fail
				success:
					ExecWait '"$TEMP\dotNET45Web.exe" $dotNET_CMD_LINE' $EXIT_CODE
					Goto is_reboot_requested

				fail:
					MessageBox MB_OK|MB_ICONEXCLAMATION "Unable to download .NET Framework.  ${PRODUCT_NAME} will be installed, but will not function without the Framework!"
					Goto done_dotNET_function

				# $EXIT_CODE contains the return codes.  1641 and 3010 means a Reboot has been requested
				is_reboot_requested:
					${If} $EXIT_CODE = 1641
					${OrIf} $EXIT_CODE = 3010
						SetRebootFlag true
					${EndIf}

	done_compare_not_needed:
		# Done dotNET Install
		Goto done_dotNET_function

	#exit the function
	done_dotNET_function:
FunctionEnd

/******************************
 *  以下是安装程序的卸载部分  *
 ******************************/

Section Uninstall
  !insertmacro MUI_STARTMENU_GETFOLDER "Application" $ICONS_GROUP
  Delete "$INSTDIR\uninst.exe"
  Delete "$INSTDIR\*.*"
  Delete "$INSTDIR\#EXE名称"

  Delete "$SMPROGRAMS\$ICONS_GROUP\Uninstall.lnk"
  Delete "$SMPROGRAMS\$ICONS_GROUP\Website.lnk"
  Delete "$DESKTOP\#系统名称.lnk"
  Delete "$SMPROGRAMS\$ICONS_GROUP\#系统名称.lnk"

  RMDir "$SMPROGRAMS\$ICONS_GROUP"

  RMDir /r "$INSTDIR\x86"
	RMDir /r "$INSTDIR\x64"
	RMDir /r "$INSTDIR\Template"
	RMDir /r "$INSTDIR\Logs"
	RMDir /r "$INSTDIR\Images"

  RMDir "$INSTDIR"

  DeleteRegKey ${PRODUCT_UNINST_ROOT_KEY} "${PRODUCT_UNINST_KEY}"
  DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"
  SetAutoClose true
SectionEnd

#-- 根据 NSIS 脚本编辑规则，所有 Function 区段必须放置在 Section 区段之后编写，以避免安装程序出现未可预知的问题。--#

Function un.onInit
  MessageBox MB_ICONQUESTION|MB_YESNO|MB_DEFBUTTON2 "您确实要完全移除 $(^Name) ，及其所有的组件？" IDYES +2
  Abort
FunctionEnd

Function un.onUninstSuccess
  HideWindow
  MessageBox MB_ICONINFORMATION|MB_OK "$(^Name) 已成功地从您的计算机移除。"
FunctionEnd
```
#### 使用命令调用构建打包程序
``` cmd
:: 调用 makensis 命令构建 NSI
makensis Setup.nsi
```
``` cmd
:: 调用 makensis 命令带参数构建 NSI
:: 从文本中读取版本号传入 NSI 中作为常量
set /p var= < ClientVersion
makensis /DClientVersion=%var% Setup.nsi
```