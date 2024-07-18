---
title: WPF 绘制扇形
date: 2024-07-18 10:00:00
tags: [c#,wpf]
categories: C#.Net
---
### 使用 ArcSegment 绘制扇形
<!-- more -->
### 简介
我想要绘制一个扇形，在实际项目中用于显示视角范围，并且需要方便控制扇形的大小范围角度与旋转方向。
以下代码都以渐变填充，如需绘制饼状图等其他样式可以使用纯色填充。

### 代码
#### 使用 Path 路径绘制
使用 Path 路径可以很容易绘制出一个扇形，由 SVG 图像转换而来，扇形大小与旋转角度可以通过直接调整控件大小和方向解决，但是无法动态修改扇形展开角度。
``` XML
<Path x:Name="path" Stretch="Fill" IsHitTestVisible="False">
    <Path.Data>
        m492.68737,0.00081a781.00618,1008.28771 0 0 0 -492.99185,225.20405l499.80025,773.31873l499.56974,-759.00719a781.00618,1008.28771 0 0 0 -506.37814,-239.51559z
    </Path.Data>
    <Path.Fill>
        <RadialGradientBrush GradientOrigin="0.5,1">
            <GradientStop Color="#B300A7FF" Offset="0"/>
            <GradientStop Color="#005968B9" Offset="1"/>
        </RadialGradientBrush>
    </Path.Fill>
</Path>
```

#### 使用 ArcSegment 绘制
1. 由于使用渐变填充，所以需要绘制一个圆形填充颜色，然后使用 Clip 裁剪，仅显示裁剪的部分。
1. 需要设置 Path 的长宽固定值用于计算圆形与扇形，存在大量固定值，例如圆中心点半径，扇形每个点坐标。
1. 绘制扇形不仅需要 ArcSegment，还需要使用 LineSegment 把图形完整的连接起来。
1. PathGeometry 中 PathFigure 的逻辑是设置一个 StartPoint 起始点，然后内部的每一个控件设置 Point 作为下一点坐标。
1. ArcSegment 中额外需要设置 Size 圆弧半径，设置 SweepDirection 指定圆弧绘制方向，设置 IsLargeArc 决定可能存在的两种圆弧绘制较大的还是较小的。
``` XML
<!--270°扇形-->
<Path x:Name="path" Width="200" Height="200" Stroke="Transparent">
    <Path.Fill>
        <RadialGradientBrush GradientOrigin="0.5,0.5">
            <GradientStop Color="#B300A7FF" Offset="0"/>
            <GradientStop Color="#005968B9" Offset="1"/>
        </RadialGradientBrush>
    </Path.Fill>
    <Path.Data>
        <EllipseGeometry Center="100,100" RadiusX="100" RadiusY="100"/>
    </Path.Data>
    <Path.Clip>
        <PathGeometry>
            <PathFigure StartPoint="100,100">
                <LineSegment Point="0,100"/>
                <ArcSegment Point="100,200" Size="100,100" SweepDirection="Clockwise" IsLargeArc="True"/>
                <LineSegment Point="100,100"/>
            </PathFigure>
        </PathGeometry>
    </Path.Clip>
</Path>
```

#### 使用 ArcSegment 封装控件绘制
1. 由于我的需求是自由调整扇形大小以及角度和旋转，所以封装成控件自由修改。
1. 扇形只有把边的四个点明确坐标，其余角度的圆周需要动态计算。
1. 用户控件内很多参数不适用于 Binding，所以由属性通知值变更后直接后台修改。
1. 使用圆弧控件 ArcSegment 无法绘制完整的圆，所以如果扇形角度 >= 360 则代码临时移除 Path.Clip。

##### 演示效果
<img src="https://sadness96.github.io/images/blog/csharp-Sector/Sector.jpg"/>

##### 用户控件前端代码 Sector.xaml
``` XML
<UserControl x:Class="Sector_Demo.Controls.Sector"
             xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
             xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" 
             xmlns:d="http://schemas.microsoft.com/expression/blend/2008" 
             xmlns:local="clr-namespace:Sector_Demo.Controls"
             mc:Ignorable="d"
             d:DesignHeight="200" d:DesignWidth="200">
    <Path x:Name="path">
        <Path.Fill>
            <RadialGradientBrush GradientOrigin="0.5,0.5">
                <GradientStop Color="#B300A7FF" Offset="0"/>
                <GradientStop Color="#005968B9" Offset="1"/>
            </RadialGradientBrush>
        </Path.Fill>
        <Path.Data>
            <EllipseGeometry x:Name="ellipse"/>
        </Path.Data>
        <Path.Clip>
            <PathGeometry>
                <PathFigure x:Name="figure">
                    <LineSegment x:Name="line1"/>
                    <ArcSegment x:Name="arc" SweepDirection="Clockwise" IsLargeArc="True"/>
                    <LineSegment x:Name="line2"/>
                </PathFigure>
            </PathGeometry>
        </Path.Clip>
    </Path>
</UserControl>
```

##### 用户控件后台代码 Sector.xaml.cs
``` CSharp
/// <summary>
/// Sector.xaml 的交互逻辑
/// </summary>
public partial class Sector : UserControl
{
    public Sector()
    {
        InitializeComponent();

        this.SizeChanged += Sector_SizeChanged;
    }

    /// <summary>
    /// 临时存储 Clip
    /// </summary>
    private Geometry? ClipTemp { get; set; }

    /// <summary>
    /// 初始角度
    /// </summary>
    public double StartAngle
    {
        get { return (double)GetValue(StartAngleProperty); }
        set { SetValue(StartAngleProperty, value); }
    }
    public static readonly DependencyProperty StartAngleProperty =
        DependencyProperty.Register("StartAngle", typeof(double), typeof(Sector), new PropertyMetadata(90d, OnStartAngleCallback));

    /// <summary>
    /// 初始角度变更回调
    /// </summary>
    /// <param name="d"></param>
    /// <param name="e"></param>
    private static void OnStartAngleCallback(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is Sector sector)
        {
            // 设置扇形
            SetSector(sector);
        }
    }

    /// <summary>
    /// 扇形角度
    /// </summary>
    public double SectorAngle
    {
        get { return (double)GetValue(SectorAngleProperty); }
        set { SetValue(SectorAngleProperty, value); }
    }
    public static readonly DependencyProperty SectorAngleProperty =
        DependencyProperty.Register("SectorAngle", typeof(double), typeof(Sector), new PropertyMetadata(90d, OnSectorAngleCallback));

    /// <summary>
    /// 扇形角度变更回调
    /// </summary>
    /// <param name="d"></param>
    /// <param name="e"></param>
    private static void OnSectorAngleCallback(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is Sector sector)
        {
            // 使用圆弧控件 ArcSegment 无法绘制完整的圆
            // 所以如果扇形角度 >= 360 则代码临时移除 Path.Clip
            if (sector.SectorAngle >= 360)
            {
                sector.ClipTemp = sector.path.Clip;
                sector.path.Clip = null;
            }
            else
            {
                if (sector.path.Clip == null)
                {
                    sector.path.Clip = sector.ClipTemp;
                }
            }

            // 设置扇形
            SetSector(sector);
        }
    }

    /// <summary>
    /// 旋转角度
    /// </summary>
    public double RotationAngle
    {
        get { return (double)GetValue(RotationAngleProperty); }
        set { SetValue(RotationAngleProperty, value); }
    }
    public static readonly DependencyProperty RotationAngleProperty =
        DependencyProperty.Register("RotationAngle", typeof(double), typeof(Sector), new PropertyMetadata(0d, OnRotationAngleCallback));

    /// <summary>
    /// 旋转角度变更回调
    /// </summary>
    /// <param name="d"></param>
    /// <param name="e"></param>
    private static void OnRotationAngleCallback(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is Sector sector)
        {
            // 设置扇形
            SetSector(sector);
        }
    }

    /// <summary>
    /// 用户控件大小变化事件
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void Sector_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        var vSize = e.NewSize;

        ellipse.Center = new Point(vSize.Width / 2, vSize.Height / 2);
        ellipse.RadiusX = vSize.Width / 2;
        ellipse.RadiusY = vSize.Height / 2;

        figure.StartPoint = new Point(vSize.Width / 2, vSize.Height / 2);

        arc.Size = new Size(vSize.Width / 2, vSize.Height / 2);
        line2.Point = new Point(vSize.Width / 2, vSize.Height / 2);

        // 设置扇形
        SetSector(this);
    }

    /// <summary>
    /// 设置扇形
    /// </summary>
    private static void SetSector(Sector sector)
    {
        // 扇形左侧
        double leftAngle = sector.RotationAngle - sector.SectorAngle / 2;

        // 扇形右侧
        double rightAngle = sector.RotationAngle + sector.SectorAngle / 2;

        double centerX = sector.Width / 2;
        double centerY = sector.Height / 2;
        double radius = sector.Width / 2;
        double startAngle = sector.StartAngle;
        sector.line1.Point = CalcCirclePoint(centerX, centerY, radius, startAngle, leftAngle);
        sector.arc.Point = CalcCirclePoint(centerX, centerY, radius, startAngle, rightAngle);
    }

    /// <summary>
    /// 计算圆周点
    /// </summary>
    /// <param name="centerX">圆中心坐标 X</param>
    /// <param name="centerY">圆中心坐标 Y</param>
    /// <param name="radius">圆的半径</param>
    /// <param name="startAngle">初始角度</param>
    /// <param name="angle">角度</param>
    /// <returns>圆周点坐标</returns>
    private static Point CalcCirclePoint(double centerX, double centerY, double radius, double startAngle, double angle)
    {
        // 将角度转换为弧度
        double angleRadians = (angle - startAngle) * Math.PI / 180;
        // 计算坐标
        double x = centerX + radius * Math.Cos(angleRadians);
        double y = centerY + radius * Math.Sin(angleRadians);
        return new Point(x, y);
    }
}
```
##### 主界面调用 MainWindow.xaml
``` XML
<Window x:Class="Sector_Demo.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        xmlns:d="http://schemas.microsoft.com/expression/blend/2008"
        xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
        xmlns:control="clr-namespace:Sector_Demo.Controls"
        xmlns:local="clr-namespace:Sector_Demo"
        mc:Ignorable="d"
        Title="Sector" Height="450" Width="800" Background="#1E1E1E">
    <Grid>
        <Grid.ColumnDefinitions>
            <ColumnDefinition Width="8*"/>
            <ColumnDefinition Width="2*"/>
        </Grid.ColumnDefinitions>
        <Grid Grid.Column="0">
            <control:Sector HorizontalAlignment="Center" VerticalAlignment="Center"
                            Width="{Binding ElementName=SectorDiameter,Path=Value}" 
                            Height="{Binding ElementName=SectorDiameter,Path=Value}"
                            SectorAngle="{Binding ElementName=SectorAngle,Path=Value}"
                            RotationAngle="{Binding ElementName=RotationAngle,Path=Value}"/>
        </Grid>
        <Border Grid.Column="1" BorderThickness="1,0,0,0" BorderBrush="Gray">
            <StackPanel Orientation="Vertical">
                <TextBlock Foreground="White">
                    <Run Text="扇形直径："/>
                    <Run Text="{Binding ElementName=SectorDiameter,Path=Value}"/>
                </TextBlock>
                <Slider x:Name="SectorDiameter" Minimum="200" Maximum="620" Value="200"/>

                <TextBlock Foreground="White">
                    <Run Text="扇形角度："/>
                    <Run Text="{Binding ElementName=SectorAngle,Path=Value}"/>
                </TextBlock>
                <Slider x:Name="SectorAngle" Minimum="0" Maximum="360" Value="90"/>

                <TextBlock Foreground="White">
                    <Run Text="旋转角度："/>
                    <Run Text="{Binding ElementName=RotationAngle,Path=Value}"/>
                </TextBlock>
                <Slider x:Name="RotationAngle" Minimum="0" Maximum="360" Value="0"/>
            </StackPanel>
        </Border>
    </Grid>
</Window>
```