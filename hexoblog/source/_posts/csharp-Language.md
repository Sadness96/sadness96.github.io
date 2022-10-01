---
title: WPF 多国语言开发
date: 2018-06-18 15:08:00
tags: [c#]
categories: C#.Net
---
### WPF 多国语言开发与配置
<!-- more -->
#### 简介
配置 WPF 客户端多国语言，并在设置中允许切换语言。
WPF 的标准做法为创建专门用于语言切换的资源字典，使用 <System:String /> 标签创建。

#### 代码
创建 Language 文件夹做为语言存放目录，资源字典命名以语言缩写命名。
英文：/Language/en-US.xaml
中文：/Language/zh-CN.xaml

##### 资源字典
``` XML
<!-- en-US.xaml -->
<ResourceDictionary xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
                    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
                    xmlns:System ="clr-namespace:System;assembly=mscorlib">

    <System:String x:Key="Save">Save</System:String>
    <System:String x:Key="Close">Close</System:String>
    <System:String x:Key="Exit">Exit</System:String>
    <System:String x:Key="OK">OK</System:String>
    <System:String x:Key="Yes">Yes</System:String>
    <System:String x:Key="No">No</System:String>
    <System:String x:Key="Cancel">Cancel</System:String>
</ResourceDictionary>
```

``` XML
<!-- zh-CN.xaml -->
<ResourceDictionary xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
                    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
                    xmlns:System ="clr-namespace:System;assembly=mscorlib">

    <System:String x:Key="Save">保存</System:String>
    <System:String x:Key="Close">取消</System:String>
    <System:String x:Key="Exit">退出</System:String>
    <System:String x:Key="OK">确定</System:String>
    <System:String x:Key="Yes">是</System:String>
    <System:String x:Key="No">否</System:String>
    <System:String x:Key="Cancel">取消</System:String>
</ResourceDictionary>
```

##### 引用资源
在全局资源(App.xaml)中引用语言配置文件
资源样式按顺序加载，标签重复的后加载覆盖先加载的，所以可以移除样式后新增，达到运行中修改语言的目的。
``` XML
<Application>
    <Application.Resources>
        <!--设置语言-->
        <ResourceDictionary Source="/Window;component/Language/en-US.xaml"/>
        <ResourceDictionary Source="/Window;component/Language/zh-CN.xaml"/>
    </Application.Resources>
</Application>
```

##### 切换语言
``` CSharp
private static List<LanguageModel> _listLanguage;
/// <summary>
/// 语言资源集合
/// </summary>
public static List<LanguageModel> ListLanguage
{
    get
    {
        if (_listLanguage == null)
        {
            _listLanguage = new List<LanguageModel>();
            _listLanguage.Add(new LanguageModel() { FileName = "en-US", Language = "English", Resource = new ResourceDictionary() { Source = new Uri("/Window;component/Language/en-US.xaml", UriKind.RelativeOrAbsolute) } });
            _listLanguage.Add(new LanguageModel() { FileName = "zh-CN", Language = "简体中文", Resource = new ResourceDictionary() { Source = new Uri("/Window;component/Language/zh-CN.xaml", UriKind.RelativeOrAbsolute) } });
        }
        return _listLanguage;
    }
    set
    {
        _listLanguage = value;
    }
}

/// <summary>
/// 设置语言
/// </summary>
/// <param name="language"></param>
public static void SetLanguage(string language = null)
{
    // 把要修改的语言放置资源最后
    List<ResourceDictionary> dictionaryList = new List<ResourceDictionary>();
    foreach (ResourceDictionary dictionary in Application.Current.Resources.MergedDictionaries)
    {
        dictionaryList.Add(dictionary);
    }
    if (string.IsNullOrEmpty(language))
    {
        var vSystemConfigInfo_Language = GetSystemConfigInfo(SECTION_SYSTEM, SYSTEM_LANGUAGE);
        foreach (var item in ListLanguage)
        {
            if (item.FileName.Equals(vSystemConfigInfo_Language))
            {
                language = item.Resource.Source.OriginalString;
                break;
            }
        }
    }
    if (!string.IsNullOrEmpty(language))
    {
        var resourceDictionary = dictionaryList.FirstOrDefault(o => o.Source.OriginalString.Equals(language));
        if (resourceDictionary != null)
        {
            Application.Current.Resources.BeginInit();
            Application.Current.Resources.MergedDictionaries.Remove(resourceDictionary);
            Application.Current.Resources.MergedDictionaries.Add(resourceDictionary);
            Application.Current.Resources.EndInit();
        }
    }
}
```

##### 调用
``` CSharp
// 初始化
SetLanguage();

// 切换英文
SetLanguage("/Window;component/Language/en-US.xaml");

// 切换简体中文
SetLanguage("/Window;component/Language/zh-CN.xaml");
```
