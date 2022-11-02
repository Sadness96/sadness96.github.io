---
title: WPF SetBinding
date: 2022-02-21 17:13:00
tags: [c#,wpf,setbinding]
categories: C#.Net
---
### WPF 通过后台绑定数据
<!-- more -->
#### 简介
WPF 中的 [Data binding](https://docs.microsoft.com/en-us/dotnet/desktop/wpf/data/?view=netdesktop-6.0) 为应用程序呈现数据和与数据交互提供了一种简单且一致的方式。元素可以以 .NET 对象和 XML 的形式绑定到来自不同类型数据源的数据。
但是有些特殊情况下只能通过后台 [FrameworkElement.SetBinding](https://docs.microsoft.com/en-us/dotnet/api/system.windows.frameworkelement.setbinding?view=windowsdesktop-6.0) 绑定数据。

#### 示例
示例为项目需要迁移到 .Net Framework 4.5.2 的 [FFME](https://github.com/unosquare/ffmediaelement) 库。
<img src="https://sadness96.github.io/images/blog/csharp-SetBinding/HttpVideo.jpg"/>

#### SetBinding 介绍
第一个参数为用户控件的 [依赖属性](https://docs.microsoft.com/zh-cn/dotnet/desktop/wpf/properties/dependency-properties-overview?view=netdesktop-6.0)。
第二个参数为 [Binding](https://docs.microsoft.com/zh-cn/dotnet/api/system.windows.data.binding?view=netframework-4.5.2) 类，同时可以设置要使用的转换器。

``` CSharp
namespace System.Windows
{
    //
    // 摘要:
    //     为 Windows Presentation Foundation (WPF) 元素提供一套 WPF 框架级别的属性、事件和方法。 此类表示所提供的 WPF
    //     框架级别实现基于 System.Windows.UIElement 定义的 WPF 核心级别 APIs。
    [RuntimeNameProperty("Name")]
    [StyleTypedProperty(Property = "FocusVisualStyle", StyleTargetType = typeof(Control))]
    [UsableDuringInitialization(true)]
    [XmlLangProperty("Language")]
    public class FrameworkElement : UIElement, IFrameworkInputElement, IInputElement, ISupportInitialize, IHaveResources, IQueryAmbient
    {
        //
        // 摘要:
        //     基于已提供的绑定对象将一个绑定附加到此元素上。
        //
        // 参数:
        //   dp:
        //     标识应在其中建立绑定的属性。
        //
        //   binding:
        //     表示数据绑定的详细信息。
        //
        // 返回结果:
        //     记录绑定的条件。 此返回值可用于错误检查。
        public BindingExpressionBase SetBinding(DependencyProperty dp, BindingBase binding);
    }
}
```

#### 公共 IValueConverter 方法
用于提供将自定义逻辑应用于绑定的方法。
``` CSharp
internal class TimeSpanToSecondsConverter : IValueConverter
{
    /// <inheritdoc />
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        switch (value)
        {
            case TimeSpan span:
                return span.TotalSeconds;
            case Duration duration:
                return duration.HasTimeSpan ? duration.TimeSpan.TotalSeconds : 0d;
            default:
                return 0d;
        }
    }

    /// <inheritdoc />
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is double == false) return 0d;
        var result = TimeSpan.FromTicks(System.Convert.ToInt64(TimeSpan.TicksPerSecond * (double)value));

        // Do the conversion from visibility to bool
        if (targetType == typeof(TimeSpan)) return result;
        return targetType == typeof(Duration) ?
            new Duration(result) : Activator.CreateInstance(targetType);
    }
}
```

#### 前台绑定 Binding
App.xaml
``` xml
<ResourceDictionary>
    <local:TimeSpanToSecondsConverter x:Key="TimeSpanToSecondsConverter" />
</ResourceDictionary>
```

UserControl.xaml
``` xml
 <Slider Name="PositionSlider" Grid.Row="0" Margin="10,0" Cursor="Hand"
                IsSnapToTickEnabled="False"
                IsEnabled="{Binding MediaElement.IsOpen}"
                SmallChange="{Binding MediaElement.PositionStep, Converter={StaticResource TimeSpanToSecondsConverter}}"
                LargeChange="{Binding MediaElement.PositionStep, Converter={StaticResource TimeSpanToSecondsConverter}}"
                Minimum="{Binding MediaElement.PlaybackStartTime, Converter={StaticResource TimeSpanToSecondsConverter}}"
                Maximum="{Binding MediaElement.PlaybackEndTime, Converter={StaticResource TimeSpanToSecondsConverter}}" 
                Value="{Binding MediaElement.Position, Converter={StaticResource TimeSpanToSecondsConverter}}" />
```

UserControlViewModel.cs
``` CSharp
private MediaElement m_MediaElement;
/// <summary>
/// Gets the media element hosted by the main window.
/// </summary>
public MediaElement MediaElement
{
    get
    {
        if (m_MediaElement == null)
            m_MediaElement = (Application.Current.MainWindow as MainWindow)?.Media;

        return m_MediaElement;
    }
}
```

#### 后台绑定 SetBinding
UserControl.xaml
``` xml
 <Slider Name="PositionSlider" Grid.Row="0" Margin="10,0" Cursor="Hand" IsSnapToTickEnabled="False"/>
```

UserControl.xaml.cs
``` CSharp
PositionSlider.SetBinding(Slider.IsEnabledProperty, new Binding("IsOpen") { Source = this._innerPlayer });

IValueConverter valueConverter = new TimeSpanToSecondsConverter();
PositionSlider.SetBinding(Slider.SmallChangeProperty, new Binding("PositionStep") { Source = this._innerPlayer, Converter = valueConverter });
PositionSlider.SetBinding(Slider.LargeChangeProperty, new Binding("PositionStep") { Source = this._innerPlayer, Converter = valueConverter });
PositionSlider.SetBinding(Slider.MinimumProperty, new Binding("PlaybackStartTime") { Source = this._innerPlayer, Converter = valueConverter });
PositionSlider.SetBinding(Slider.MaximumProperty, new Binding("PlaybackEndTime") { Source = this._innerPlayer, Converter = valueConverter });
PositionSlider.SetBinding(Slider.ValueProperty, new Binding("Position") { Source = this._innerPlayer, Converter = valueConverter });
```