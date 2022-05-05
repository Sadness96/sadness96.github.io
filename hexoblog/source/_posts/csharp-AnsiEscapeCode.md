---
title: ANSI 转义码
date: 2022-04-27 13:40:20
tags: [c#,helper,wpf]
categories: C#.Net
---
### WPF 文本框像控制台做一个五彩斑斓的黑
<!-- more -->
#### 简介
参考 [维基百科 ANSI escape code](https://en.wikipedia.org/wiki/ANSI_escape_code) 控制台程序使用 ANSI 转义来体现文字颜色，在控制台中显示出一个五彩斑斓的黑，也有很多控制台程序用这些带文字的颜色显示自家 LOGO。
WPF 想要在一行中加载不同颜色的文字，使用富文本控件 [RichTextBox](https://docs.microsoft.com/zh-cn/dotnet/api/system.windows.controls.richtextbox?f1url=%3FappId%3DDev16IDEF1%26l%3DZH-CN%26k%3Dk(System.Windows.Controls.RichTextBox);k(DevLang-csharp)%26rd%3Dtrue&view=windowsdesktop-6.0) 即可，但是需要先解析文本中的 Ansi 字符。
完整参考代码在开源项目 [EasyDeploy](https://github.com/iceelves/EasyDeploy) 中有重写富文本控件。

#### 打印颜色板
<img src="https://sadness96.github.io/images/blog/csharp-AnsiEscapeCode/ControlColor.jpg"/>

##### 代码
``` CSharp
static void Main(string[] args)
{
    Dictionary<string, int> dicForeground = new Dictionary<string, int>()
    {
        { "Black", 30 },
        { "Red", 31 },
        { "Green", 32 },
        { "Yellow", 33 },
        { "Blue", 34 },
        { "Magenta", 35 },
        { "Cyan", 36 },
        { "White", 37 },
        { "Gray", 90 },
        { "Bright Red", 91 },
        { "Bright Green", 92 },
        { "Bright Yellow", 93 },
        { "Bright Blue", 94 },
        { "Bright Magenta", 95 },
        { "Bright Cyan", 96 },
        { "Bright White", 97 }
    };
    Dictionary<string, int> dicBackground = new Dictionary<string, int>()
    {
        { "Black", 40 },
        { "Red", 41 },
        { "Green", 42 },
        { "Yellow", 43 },
        { "Blue", 44 },
        { "Magenta", 45 },
        { "Cyan", 46 },
        { "White", 47 },
        { "Gray", 100 },
        { "Bright Red", 101 },
        { "Bright Green", 102 },
        { "Bright Yellow", 103 },
        { "Bright Blue", 104 },
        { "Bright Magenta", 105 },
        { "Bright Cyan", 106 },
        { "Bright White", 107 }
    };

    foreach (var itemForeground in dicForeground)
    {
        Console.Write($"{itemForeground.Key}{(itemForeground.Key.Length > 8 ? "\t" : "\t\t")}");
        Console.Write($"\u001b[{itemForeground.Value}m{itemForeground.Value}");
        Console.Write($"\u001b[0m ");
        foreach (var itemBackground in dicBackground)
        {
            Console.Write($"\u001b[{itemForeground.Value};{itemBackground.Value}m{itemForeground.Value};{itemBackground.Value}");
            Console.Write($"\u001b[0m ");
        }
        Console.Write($"\r\n");
    }
    Console.ReadKey();
}
```

#### 解析字符
Ansi 编码由 \u001b[ 开头，加数值编码，由 m 结尾。
数值 0 为默认值。
数值 30-37;90-97 为文字颜色。
数值 40-47;100-107 为背景颜色。
通过正则表达式 @"\u001b\[(.*?)m" 匹配文本，后续的文本即为设定颜色。

##### 代码
``` CSharp
/// <summary>
/// ANSI 转义序列相关帮助类
/// </summary>
public static class AnsiHelper
{
    /// <summary>
    /// 匹配 Ansi 头
    /// </summary>
    public static char AnsiStart = '\u001b';

    /// <summary>
    /// 匹配 Ansi 正则
    /// </summary>
    public static string AnsiRegex = @"\u001b\[(.*?)m";

    /// <summary>
    /// 去除字符串中的 ANSI 序列
    /// </summary>
    /// <param name="text">包含 Ansi 的文本</param>
    /// <returns></returns>
    public static string RemoveAnsi(string text)
    {
        return Regex.Replace(text, AnsiRegex, string.Empty);
    }

    /// <summary>
    /// 获取通过正则 Ansi 拆分的数据集
    /// </summary>
    /// <param name="text">包含 Ansi 的文本</param>
    /// <returns></returns>
    public static List<string> GetAnsiSplit(string text)
    {
        List<string> listAnsiSplit = new List<string>();
        // 获取匹配 Ansi 数据
        var vMatches = Regex.Matches(text, AnsiRegex);
        // 获取 Ansi 下标
        List<int> indexs = new List<int>();
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i].Equals(AnsiStart))
            {
                indexs.Add(i);
            }
        }
        // 遍历拆分数据
        if (indexs.Count == 0)
        {
            // 无颜色数据
            listAnsiSplit.Add(text);
        }
        else
        {
            // 解析颜色数据
            int iSubscript = 0;
            for (int i = 0; i < indexs.Count; i++)
            {
                if (i == 0 && indexs[i] > 0)
                {
                    // 如果大于起始位置，先把起始数据赋值
                    listAnsiSplit.Add(text.Substring(0, indexs[i]));
                    iSubscript += indexs[i];
                }
                // 添加 Ansi 数据
                listAnsiSplit.Add(vMatches[i].Value);
                iSubscript += vMatches[i].Value.Length;

                // 添加其他数据
                int iSubCount = (indexs.Count > i + 1 ? indexs[i + 1] : text.Length) - iSubscript;
                if (iSubCount > 0)
                {
                    listAnsiSplit.Add(text.Substring(iSubscript, iSubCount));
                    iSubscript += iSubCount;
                }
            }
        }
        return listAnsiSplit;
    }
}
```

#### 把 Ansi 字符串解析为富文本所需的控件
通过正则表达式拆分，后创建 [Run](https://docs.microsoft.com/zh-cn/dotnet/api/system.windows.documents.run.-ctor?f1url=%3FappId%3DDev16IDEF1%26l%3DZH-CN%26k%3Dk(System.Windows.Documents.Run.%2523ctor);k(DevLang-csharp)%26rd%3Dtrue&view=windowsdesktop-6.0) 控件来创建前景色和背景色。

##### 代码
``` CSharp
/// <summary>
/// 添加文本
/// </summary>
/// <param name="Text"></param>
public void SetText(string Text)
{
    // 根据最大显示行数删除
    if (this.Document.Blocks.Count >= MaxRows)
    {
        int iRempveNumber = this.Document.Blocks.Count - MaxRows;
        List<Paragraph> listRemoveTemp = new List<Paragraph>();
        foreach (var item in this.Document.Blocks)
        {
            if (iRempveNumber > 0)
            {
                iRempveNumber--;
                listRemoveTemp.Add(item as Paragraph);
            }
        }
        if (listRemoveTemp != null && listRemoveTemp.Count >= 1)
        {
            foreach (var item in listRemoveTemp)
            {
                this.Document.Blocks.Remove(item);
            }
        }
    }

    // 添加文本
    string ansiColor = null;
    Paragraph paragraph = new Paragraph();
    foreach (var item in AnsiHelper.GetAnsiSplit(Text))
    {
        if (item.Contains(AnsiHelper.AnsiStart))
        {
            // 设置颜色
            ansiColor = item;
        }
        else
        {
            paragraph.Inlines.Add(SetColorFromAnsi(new Run() { Text = item }, ansiColor));
        }
    }
    this.Document.Blocks.Add(paragraph);

    // 滚动条超过 80% 或滚动条小于一倍控件高度 滚动到底部
    if (this.VerticalOffset / (this.ExtentHeight - this.ActualHeight) >= 0.8 || (this.ExtentHeight - this.ActualHeight) <= this.ActualHeight)
    {
        this.ScrollToEnd();
    }
}

/// <summary>
/// 根据 Ansi 设置文本颜色
/// </summary>
/// <param name="run">文本</param>
/// <param name="ansiColor">ansi 颜色</param>
/// <returns></returns>
private Run SetColorFromAnsi(Run run, string ansiColor)
{
    if (string.IsNullOrEmpty(ansiColor))
    {
        return run;
    }
    var vMatches = Regex.Matches(ansiColor, AnsiHelper.AnsiRegex);
    if (vMatches != null && vMatches.Count >= 1 && vMatches[0].Groups != null && vMatches[0].Groups.Count >= 2)
    {
        var vSplit = vMatches[0].Groups[1].Value.Split(';');
        foreach (var item in vSplit)
        {
            switch (item)
            {
                // Black
                case "30": run.Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 0)); break;
                case "40": run.Background = new SolidColorBrush(Color.FromRgb(0, 0, 0)); break;

                // Red
                case "31": run.Foreground = new SolidColorBrush(Color.FromRgb(128, 0, 0)); break;
                case "41": run.Background = new SolidColorBrush(Color.FromRgb(128, 0, 0)); break;

                // Green
                case "32": run.Foreground = new SolidColorBrush(Color.FromRgb(0, 128, 0)); break;
                case "42": run.Background = new SolidColorBrush(Color.FromRgb(0, 128, 0)); break;

                // Yellow
                case "33": run.Foreground = new SolidColorBrush(Color.FromRgb(128, 128, 0)); break;
                case "43": run.Background = new SolidColorBrush(Color.FromRgb(128, 128, 0)); break;

                // Blue
                case "34": run.Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 128)); break;
                case "44": run.Background = new SolidColorBrush(Color.FromRgb(0, 0, 128)); break;

                // Magenta
                case "35": run.Foreground = new SolidColorBrush(Color.FromRgb(128, 0, 128)); break;
                case "45": run.Background = new SolidColorBrush(Color.FromRgb(128, 0, 128)); break;

                // Cyan
                case "36": run.Foreground = new SolidColorBrush(Color.FromRgb(0, 128, 128)); break;
                case "46": run.Background = new SolidColorBrush(Color.FromRgb(0, 128, 128)); break;

                // White
                case "37": run.Foreground = new SolidColorBrush(Color.FromRgb(192, 192, 192)); break;
                case "47": run.Background = new SolidColorBrush(Color.FromRgb(192, 192, 192)); break;

                // Bright Black (Gray)
                case "90": run.Foreground = new SolidColorBrush(Color.FromRgb(128, 128, 128)); break;
                case "100": run.Background = new SolidColorBrush(Color.FromRgb(128, 128, 128)); break;

                // Bright Red
                case "91": run.Foreground = new SolidColorBrush(Color.FromRgb(255, 0, 0)); break;
                case "101": run.Background = new SolidColorBrush(Color.FromRgb(255, 0, 0)); break;

                // Bright Green
                case "92": run.Foreground = new SolidColorBrush(Color.FromRgb(0, 255, 0)); break;
                case "102": run.Background = new SolidColorBrush(Color.FromRgb(0, 255, 0)); break;

                // Bright Yellow
                case "93": run.Foreground = new SolidColorBrush(Color.FromRgb(255, 255, 0)); break;
                case "103": run.Background = new SolidColorBrush(Color.FromRgb(255, 255, 0)); break;

                // Bright Blue
                case "94": run.Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 255)); break;
                case "104": run.Background = new SolidColorBrush(Color.FromRgb(0, 0, 255)); break;

                // Bright Magenta
                case "95": run.Foreground = new SolidColorBrush(Color.FromRgb(255, 0, 255)); break;
                case "105": run.Background = new SolidColorBrush(Color.FromRgb(255, 0, 255)); break;

                // Bright Cyan
                case "96": run.Foreground = new SolidColorBrush(Color.FromRgb(0, 255, 255)); break;
                case "106": run.Background = new SolidColorBrush(Color.FromRgb(0, 255, 255)); break;

                // Bright White
                case "97": run.Foreground = new SolidColorBrush(Color.FromRgb(255, 255, 255)); break;
                case "107": run.Background = new SolidColorBrush(Color.FromRgb(255, 255, 255)); break;
                default:
                    break;
            }
        }
    }
    return run;
}
```