---
title: 打印Word文件
date: 2018-10-25 16:40:42
tags: [asp,printer]
categories: Asp.Net
---
### 服务端调用打印机打印Word文件
<!-- more -->
#### 简介
工作时需要直接操作 [OA系统](https://baike.baidu.com/item/%E5%8A%9E%E5%85%AC%E8%87%AA%E5%8A%A8%E5%8C%96/1428?fromtitle=OA&fromid=25368&fr=aladdin) 调用打印机打印 [Word](https://baike.baidu.com/item/Microsoft%20Office%20Word/1448679?fromtitle=word&fromid=2970534) 文件。
#### 引用库介绍
需要电脑安装 [Microsoft Office](https://office.microsoft.com/) 并引用COM组件 [Microsoft.Office.Interop.Word](https://docs.microsoft.com/en-us/dotnet/api/microsoft.office.interop.word?redirectedfrom=MSDN&view=word-pia) 才可以调用打印机。
#### 代码及调用
打印Word
``` CSharp
using Word = Microsoft.Office.Interop.Word;

/// <summary>
/// 打印Word
/// </summary>
/// <param name="filePath">需要打印的文件</param>
/// <param name="PrintName">打印机名称</param>
private static void PrintWord(string filePath, string PrintName)
{
    try
    {
        //要打印的文件路径
        Object wordFile = filePath;
        object oMissing = Missing.Value;
        //自定义object类型的布尔值
        object oTrue = true;
        object oFalse = false;
        object doNotSaveChanges = Word.WdSaveOptions.wdDoNotSaveChanges;

        //Word.Application appWord = null;
        //定义word Application相关
        Word.Application appWord = new Word.Application();

        //word程序不可见
        appWord.Visible = false;

        //不弹出警告框
        appWord.DisplayAlerts = Word.WdAlertLevel.wdAlertsNone;

        //先保存默认的打印机
        string defaultPrinter = appWord.ActivePrinter;

        //打开要打印的文件
        Word.Document doc = appWord.Documents.Open(
            ref wordFile,
            ref oMissing,
            ref oTrue,
            ref oFalse,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing
            );

        //设置指定的打印机名字
        appWord.ActivePrinter = PrintName;

        //打印
        doc.PrintOut(
            ref oTrue,//此处为true表示后台打印
            ref oFalse,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing,
            ref oMissing
            );

        //打印完关闭word文件
        doc.Close(ref doNotSaveChanges, ref oMissing, ref oMissing);

        //还原原来的默认打印机
        appWord.ActivePrinter = defaultPrinter;

        //退出word程序
        appWord.Quit(ref oMissing, ref oMissing, ref oMissing);
        doc = null;
        appWord = null;
    }
    catch (Exception ex)
    {
        //代码行数
        string line = ex.StackTrace.ToString();
        //返回错误发生的方法定义
        string errorfunction = ex.TargetSite.ToString();
        int code = ex.HResult;
    }
}
```
调用方法
``` CSharp
//打印Word
PrintWord(dialog.FileName, "审批打印机");
```