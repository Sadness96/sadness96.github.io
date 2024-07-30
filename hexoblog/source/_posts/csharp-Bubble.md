---
title: WPF 绘制气泡弹窗
date: 2024-07-30 20:42:00
tags: [c#,wpf]
categories: C#.Net
---
### 使用 CombinedGeometry 并集绘制气泡弹窗
<!-- more -->
### 简介
使用 Popup + Border 可以很容易绘制一个简单的气泡弹窗，但是通常为了界面好看，都会在气泡弹窗加一个三角指向弹出方向。
经过测试后以绘制 CombinedGeometry 并集图形效果最好。

### 代码
<img src="https://sadness96.github.io/images/blog/csharp-Bubble/BubbleTest.jpg"/>

#### 绘制 Border 与三角形和边线（图 1）
``` XML
<Grid Width="100" Height="100" Opacity="0.6" HorizontalAlignment="Left" Margin="15,0,5,0">
    <Border CornerRadius="8" BorderThickness="1" BorderBrush="Gray" Background="Black">
        <TextBlock Text="1" VerticalAlignment="Center" HorizontalAlignment="Center" Foreground="White"/>
    </Border>
    <Path Width="20" Height="14" HorizontalAlignment="Center" VerticalAlignment="Bottom" Margin="0,-12" Fill="Black">
        <Path.Data>
            <PathGeometry>
                <PathFigure StartPoint="0,0">
                    <LineSegment Point="10,16"/>
                    <LineSegment Point="20,0"/>
                </PathFigure>
            </PathGeometry>
        </Path.Data>
    </Path>
    <Canvas Width="20" Height="14" HorizontalAlignment="Center" VerticalAlignment="Bottom" Margin="0,-13">
        <Line X1="0" Y1="0" X2="10" Y2="14" Stroke="Gray" StrokeThickness="1"/>
        <Line X1="10" Y1="14" X2="20" Y2="0" Stroke="Gray" StrokeThickness="1"/>
    </Canvas>
</Grid>
```

#### 绘制 Border 与三角路径（图 2）
``` XML
<Grid Width="100" Height="100" Opacity="0.6" HorizontalAlignment="Left" Margin="5">
    <Border CornerRadius="8" BorderThickness="1" BorderBrush="Gray" Background="Black">
        <TextBlock Text="2" VerticalAlignment="Center" HorizontalAlignment="Center" Foreground="White"/>
    </Border>
    <Path Width="20" Height="14" HorizontalAlignment="Center" VerticalAlignment="Bottom" Margin="0,-13" Fill="Black" Stroke="Gray">
        <Path.Data>
            <PathGeometry>
                <PathFigure StartPoint="0,0">
                    <LineSegment Point="10,14"/>
                    <LineSegment Point="20,0"/>
                </PathFigure>
            </PathGeometry>
        </Path.Data>
    </Path>
</Grid>
```

#### 绘制 Border 与三角图形裁剪（图 3）
``` XML
<Grid Width="100" Height="100" Opacity="0.6" HorizontalAlignment="Left" Margin="5">
    <Border CornerRadius="8" BorderThickness="1" BorderBrush="Gray" Background="Black">
        <TextBlock Text="3" VerticalAlignment="Center" HorizontalAlignment="Center" Foreground="White"/>
    </Border>
    <Path HorizontalAlignment="Center" VerticalAlignment="Bottom" Margin="0,-13" Fill="Black" Stroke="Gray"
        Data="m12.6888,23.2835c1.6298,2.2887 4.9926,2.2887 6.6224,0l11.9093,-16.72352c1.949,-2.73689 0.0193,-6.55998 -3.3112,-6.55998l-23.81862,0c-3.33049,0 -5.26021,3.82309 -3.31119,6.55998l11.90931,16.72352z">
        <Path.Clip>
            <RectangleGeometry Rect="0,11,40,40" />
        </Path.Clip>
    </Path>
</Grid>
```

#### 绘制并集组合图形（图 4）
``` XML
<Grid Width="100" Height="121" HorizontalAlignment="Left" Margin="5">
    <Path Stroke="Gray" StrokeThickness="1" Fill="Black" Opacity="0.6" Margin="0,10,0,0">
        <Path.Data>
            <CombinedGeometry GeometryCombineMode="Union">
                <CombinedGeometry.Geometry1>
                    <RectangleGeometry Rect="1,0,98,100" RadiusX="8" RadiusY="8"/>
                </CombinedGeometry.Geometry1>
                <CombinedGeometry.Geometry2>
                    <PathGeometry>
                        <PathFigure StartPoint="12.6888,23.2835">
                            <BezierSegment Point1="14.3186,25.5722" Point2="17.6814,25.5722" Point3="19.3112,23.2835" />
                            <LineSegment Point="31.2205,6.55998" />
                            <BezierSegment Point1="33.1695,3.82309" Point2="31.2398,0" Point3="27.9286,0" />
                            <LineSegment Point="4.11098,0" />
                            <BezierSegment Point1="0.780498,0" Point2="-1.16953,3.82309" Point3="0.779489,6.55998" />
                            <LineSegment Point="12.6888,23.2835" />
                        </PathFigure>
                        <PathGeometry.Transform>
                            <TranslateTransform X="34" Y="85"/>
                        </PathGeometry.Transform>
                    </PathGeometry>
                </CombinedGeometry.Geometry2>
            </CombinedGeometry>
        </Path.Data>
    </Path>
    <Border Width="100" Height="110" VerticalAlignment="Top" CornerRadius="8" Margin="0,10,0,0">
        <TextBlock Text="4" VerticalAlignment="Center" HorizontalAlignment="Center" Foreground="White"/>
    </Border>
</Grid>
```

#### 使用 CombinedGeometry 封装控件绘制
经过测试图 4 效果最好，所以对图 4 进行用户控件封装

##### 演示效果
<img src="https://sadness96.github.io/images/blog/csharp-Bubble/Bubble.jpg"/>

##### 用户控件前端代码 Bubble.xaml
``` XML
<UserControl x:Class="Bubble_Demo.Controls.Bubble"
             xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
             xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" 
             xmlns:d="http://schemas.microsoft.com/expression/blend/2008" 
             xmlns:local="clr-namespace:Bubble_Demo.Controls"
             mc:Ignorable="d" 
             d:DesignHeight="200" d:DesignWidth="200">
    <Grid>
        <Path Stroke="{Binding BubbleBrush,RelativeSource={RelativeSource AncestorType=UserControl}}" 
              StrokeThickness="{Binding BubbleThickness,RelativeSource={RelativeSource AncestorType=UserControl}}" 
              Fill="{Binding BubbleBackground,RelativeSource={RelativeSource AncestorType=UserControl}}" 
              Opacity="{Binding BubbleOpacity,RelativeSource={RelativeSource AncestorType=UserControl}}">
            <Path.Data>
                <CombinedGeometry GeometryCombineMode="Union">
                    <CombinedGeometry.Geometry1>
                        <RectangleGeometry x:Name="Rectangle"
                                           RadiusX="{Binding CornerRadius,RelativeSource={RelativeSource AncestorType=UserControl}}" 
                                           RadiusY="{Binding CornerRadius,RelativeSource={RelativeSource AncestorType=UserControl}}"/>
                    </CombinedGeometry.Geometry1>
                    <CombinedGeometry.Geometry2>
                        <PathGeometry x:Name="Triangle">
                            <PathFigure StartPoint="12.6888,23.2835">
                                <BezierSegment Point1="14.3186,25.5722" Point2="17.6814,25.5722" Point3="19.3112,23.2835" />
                                <LineSegment Point="31.2205,6.55998" />
                                <BezierSegment Point1="33.1695,3.82309" Point2="31.2398,0" Point3="27.9286,0" />
                                <LineSegment Point="4.11098,0" />
                                <BezierSegment Point1="0.780498,0" Point2="-1.16953,3.82309" Point3="0.779489,6.55998" />
                                <LineSegment Point="12.6888,23.2835" />
                            </PathFigure>
                            <PathGeometry.Transform>
                                <TransformGroup>
                                    <RotateTransform x:Name="TriangleAngle" Angle="0"/>
                                    <TranslateTransform x:Name="TriangleCoordinate" X="84" Y="175"/>
                                </TransformGroup>
                            </PathGeometry.Transform>
                        </PathGeometry>
                    </CombinedGeometry.Geometry2>
                </CombinedGeometry>
            </Path.Data>
        </Path>
        <ContentPresenter x:Name="BubbleContent" Content="{Binding Content,RelativeSource={RelativeSource AncestorType=UserControl}}"/>
    </Grid>
</UserControl>
```

##### 用户控件后台代码 Bubble.xaml.cs
``` CSharp
/// <summary>
/// Bubble.xaml 的交互逻辑
/// </summary>
public partial class Bubble : UserControl
{
    public Bubble()
    {
        InitializeComponent();

        this.SizeChanged += Bubble_SizeChanged;
    }

    /// <summary>
    /// 用户控件大小变化事件
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void Bubble_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        // 设置气泡
        SetBubble(this);
    }

    /// <summary>
    /// 气泡边框粗细
    /// </summary>
    public double BubbleThickness
    {
        get { return (double)GetValue(BubbleThicknessProperty); }
        set { SetValue(BubbleThicknessProperty, value); }
    }
    public static readonly DependencyProperty BubbleThicknessProperty =
        DependencyProperty.Register("BubbleThickness", typeof(double), typeof(Bubble), new PropertyMetadata(default(double), OnBubbleThicknessCallback));

    /// <summary>
    /// 气泡边框粗细变更回调
    /// </summary>
    /// <param name="d"></param>
    /// <param name="e"></param>
    private static void OnBubbleThicknessCallback(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is Bubble bubble)
        {
            // 设置气泡
            SetBubble(bubble);
        }
    }

    /// <summary>
    /// 气泡边框颜色
    /// </summary>
    public Brush BubbleBrush
    {
        get { return (Brush)GetValue(BubbleBrushProperty); }
        set { SetValue(BubbleBrushProperty, value); }
    }
    public static readonly DependencyProperty BubbleBrushProperty =
        DependencyProperty.Register("BubbleBrush", typeof(Brush), typeof(Bubble), new PropertyMetadata(default(Brush)));

    /// <summary>
    /// 气泡背景颜色
    /// </summary>
    public Brush BubbleBackground
    {
        get { return (Brush)GetValue(BubbleBackgroundProperty); }
        set { SetValue(BubbleBackgroundProperty, value); }
    }
    public static readonly DependencyProperty BubbleBackgroundProperty =
        DependencyProperty.Register("BubbleBackground", typeof(Brush), typeof(Bubble), new PropertyMetadata(default(Brush)));

    /// <summary>
    /// 气泡透明度
    /// </summary>
    public double BubbleOpacity
    {
        get { return (double)GetValue(BubbleOpacityProperty); }
        set { SetValue(BubbleOpacityProperty, value); }
    }
    public static readonly DependencyProperty BubbleOpacityProperty =
        DependencyProperty.Register("BubbleOpacity", typeof(double), typeof(Bubble), new PropertyMetadata(default(double)));

    /// <summary>
    /// 矩形角半径
    /// </summary>
    public double CornerRadius
    {
        get { return (double)GetValue(CornerRadiusProperty); }
        set { SetValue(CornerRadiusProperty, value); }
    }
    public static readonly DependencyProperty CornerRadiusProperty =
        DependencyProperty.Register("CornerRadius", typeof(double), typeof(Bubble), new PropertyMetadata(default(double)));

    /// <summary>
    /// 气泡三角指向方向
    /// </summary>
    public BubbleAlignment PointingDirection
    {
        get { return (BubbleAlignment)GetValue(PointingDirectionProperty); }
        set { SetValue(PointingDirectionProperty, value); }
    }
    public static readonly DependencyProperty PointingDirectionProperty =
        DependencyProperty.Register("PointingDirection", typeof(BubbleAlignment), typeof(Bubble), new PropertyMetadata(BubbleAlignment.Bottom, OnPointingDirectionCallback));

    /// <summary>
    /// 气泡三角指向方向变更回调
    /// </summary>
    /// <param name="d"></param>
    /// <param name="e"></param>
    private static void OnPointingDirectionCallback(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is Bubble bubble)
        {
            // 设置气泡
            SetBubble(bubble);
        }
    }

    /// <summary>
    /// 气泡三角漏出大小(0-20)
    /// 默认 10
    /// </summary>
    public double TriangleSize
    {
        get { return (double)GetValue(TriangleSizeProperty); }
        set { SetValue(TriangleSizeProperty, value); }
    }
    public static readonly DependencyProperty TriangleSizeProperty =
        DependencyProperty.Register("TriangleSize", typeof(double), typeof(Bubble), new PropertyMetadata(10d, OnTriangleSizeCallback));

    /// <summary>
    /// 气泡漏出大小变更回调
    /// </summary>
    /// <param name="d"></param>
    /// <param name="e"></param>
    private static void OnTriangleSizeCallback(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is Bubble bubble)
        {
            // 设置气泡
            SetBubble(bubble);
        }
    }

    /// <summary>
    /// 组件
    /// </summary>
    public new object Content
    {
        get { return (object)GetValue(ContentProperty); }
        set { SetValue(ContentProperty, value); }
    }
    public static readonly new DependencyProperty ContentProperty =
        DependencyProperty.Register("Content", typeof(object), typeof(Bubble), new PropertyMetadata(null));

    /// <summary>
    /// 设置气泡
    /// </summary>
    /// <param name="bubble"></param>
    private static void SetBubble(Bubble bubble)
    {
        var vTriangleBounds = bubble.Triangle.Bounds;
        var vTriangleWidth = Math.Round(vTriangleBounds.Width, 0);
        var vTriangleHeight = Math.Round(vTriangleBounds.Height, 0);

        switch (bubble.PointingDirection)
        {
            case BubbleAlignment.Left:
                // 矩形
                bubble.Rectangle.Rect = new Rect(
                    bubble.BubbleThickness + bubble.TriangleSize,
                    bubble.BubbleThickness,
                    bubble.Width - (bubble.BubbleThickness * 2) - bubble.TriangleSize,
                    bubble.Height - (bubble.BubbleThickness * 2));
                // Content
                bubble.BubbleContent.HorizontalAlignment = HorizontalAlignment.Right;
                bubble.BubbleContent.VerticalAlignment = VerticalAlignment.Stretch;
                bubble.BubbleContent.Width = bubble.Width - bubble.TriangleSize;
                bubble.BubbleContent.Height = bubble.Height;
                // 三角
                bubble.TriangleAngle.Angle = 90;
                bubble.TriangleCoordinate.X = vTriangleWidth + bubble.BubbleThickness;
                bubble.TriangleCoordinate.Y = bubble.Height / 2 - vTriangleWidth / 2;
                break;
            case BubbleAlignment.Right:
                // 矩形
                bubble.Rectangle.Rect = new Rect(
                    bubble.BubbleThickness,
                    bubble.BubbleThickness,
                    bubble.Width - (bubble.BubbleThickness * 2) - bubble.TriangleSize,
                    bubble.Height - (bubble.BubbleThickness * 2));
                // Content
                bubble.BubbleContent.HorizontalAlignment = HorizontalAlignment.Left;
                bubble.BubbleContent.VerticalAlignment = VerticalAlignment.Stretch;
                bubble.BubbleContent.Width = bubble.Width - bubble.TriangleSize;
                bubble.BubbleContent.Height = bubble.Height;
                // 三角
                bubble.TriangleAngle.Angle = -90;
                bubble.TriangleCoordinate.X = bubble.Width - vTriangleWidth - bubble.BubbleThickness;
                bubble.TriangleCoordinate.Y = bubble.Height / 2 + vTriangleWidth / 2;
                break;
            case BubbleAlignment.Top:
                // 矩形
                bubble.Rectangle.Rect = new Rect(
                    bubble.BubbleThickness,
                    bubble.TriangleSize + bubble.BubbleThickness,
                    bubble.Width - (bubble.BubbleThickness * 2),
                    bubble.Height - bubble.TriangleSize - (bubble.BubbleThickness * 2));
                // Content
                bubble.BubbleContent.HorizontalAlignment = HorizontalAlignment.Stretch;
                bubble.BubbleContent.VerticalAlignment = VerticalAlignment.Bottom;
                bubble.BubbleContent.Width = bubble.Width;
                bubble.BubbleContent.Height = bubble.Height - bubble.TriangleSize;
                // 三角
                bubble.TriangleAngle.Angle = 180;
                bubble.TriangleCoordinate.X = (bubble.Width - vTriangleWidth) / 2 + vTriangleWidth;
                bubble.TriangleCoordinate.Y = vTriangleHeight + bubble.BubbleThickness;
                break;
            case BubbleAlignment.Bottom:
                // 矩形
                bubble.Rectangle.Rect = new Rect(
                    bubble.BubbleThickness,
                    bubble.BubbleThickness,
                    bubble.Width - (bubble.BubbleThickness * 2),
                    bubble.Height - bubble.TriangleSize - (bubble.BubbleThickness * 2));
                // Content
                bubble.BubbleContent.HorizontalAlignment = HorizontalAlignment.Stretch;
                bubble.BubbleContent.VerticalAlignment = VerticalAlignment.Top;
                bubble.BubbleContent.Width = bubble.Width;
                bubble.BubbleContent.Height = bubble.Height - bubble.TriangleSize;
                // 三角
                bubble.TriangleAngle.Angle = 0;
                bubble.TriangleCoordinate.X = (bubble.Width - vTriangleWidth) / 2;
                bubble.TriangleCoordinate.Y = bubble.Height - vTriangleHeight - bubble.BubbleThickness;
                break;
        }
    }
}

/// <summary>
/// 气泡指向方向
/// </summary>
public enum BubbleAlignment
{
    Left,
    Right,
    Top,
    Bottom
}
```

##### 主界面调用 MainWindow.xaml
``` XML
<Window x:Class="Bubble_Demo.MainWindow"
        xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        xmlns:d="http://schemas.microsoft.com/expression/blend/2008"
        xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
        xmlns:control="clr-namespace:Bubble_Demo.Controls"
        xmlns:local="clr-namespace:Bubble_Demo"
        mc:Ignorable="d"
        Title="Bubble" Height="450" Width="800" Background="#1E1E1E">
    <Grid>
        <Grid.RowDefinitions>
            <RowDefinition Height="*"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="*"/>
        </Grid.RowDefinitions>
        <Grid.ColumnDefinitions>
            <ColumnDefinition Width="*"/>
            <ColumnDefinition Width="*"/>
            <ColumnDefinition Width="*"/>
        </Grid.ColumnDefinitions>
        <!--左-->
        <control:Bubble Width="200" Height="120" Grid.Row="0" Grid.RowSpan="3" Grid.Column="0" PointingDirection="Right"
                        BubbleBrush="Gray" BubbleBackground="Black" 
                        BubbleThickness="{Binding ElementName=BubbleThickness,Path=Value}" 
                        BubbleOpacity="{Binding ElementName=BubbleOpacity,Path=Value}" 
                        CornerRadius="{Binding ElementName=CornerRadius,Path=Value}" 
                        TriangleSize="{Binding ElementName=TriangleSize,Path=Value}">
            <control:Bubble.Content>
                <Grid>
                    <TextBlock Text="Left" VerticalAlignment="Center" HorizontalAlignment="Center" Foreground="White"/>
                </Grid>
            </control:Bubble.Content>
        </control:Bubble>
        <!--右-->
        <control:Bubble Width="200" Height="120" Grid.Row="0" Grid.RowSpan="3" Grid.Column="2" PointingDirection="Left"
                        BubbleBrush="Gray" BubbleBackground="Black" 
                        BubbleThickness="{Binding ElementName=BubbleThickness,Path=Value}" 
                        BubbleOpacity="{Binding ElementName=BubbleOpacity,Path=Value}" 
                        CornerRadius="{Binding ElementName=CornerRadius,Path=Value}" 
                        TriangleSize="{Binding ElementName=TriangleSize,Path=Value}">
            <control:Bubble.Content>
                <Grid>
                    <TextBlock Text="Right" VerticalAlignment="Center" HorizontalAlignment="Center" Foreground="White"/>
                </Grid>
            </control:Bubble.Content>
        </control:Bubble>
        <!--上-->
        <control:Bubble Width="200" Height="120" Grid.Row="0" Grid.RowSpan="2" VerticalAlignment="Top" Grid.Column="1" Margin="10" PointingDirection="Bottom"
                        BubbleBrush="Gray" BubbleBackground="Black" 
                        BubbleThickness="{Binding ElementName=BubbleThickness,Path=Value}" 
                        BubbleOpacity="{Binding ElementName=BubbleOpacity,Path=Value}" 
                        CornerRadius="{Binding ElementName=CornerRadius,Path=Value}" 
                        TriangleSize="{Binding ElementName=TriangleSize,Path=Value}">
            <control:Bubble.Content>
                <Grid>
                    <TextBlock Text="Top" VerticalAlignment="Center" HorizontalAlignment="Center" Foreground="White"/>
                </Grid>
            </control:Bubble.Content>
        </control:Bubble>
        <!--下-->
        <control:Bubble Width="200" Height="120"
                        Grid.Row="1" Grid.RowSpan="2" VerticalAlignment="Bottom" Grid.Column="1" Margin="10" PointingDirection="Top"
                        BubbleBrush="Gray" BubbleBackground="Black" 
                        BubbleThickness="{Binding ElementName=BubbleThickness,Path=Value}" 
                        BubbleOpacity="{Binding ElementName=BubbleOpacity,Path=Value}" 
                        CornerRadius="{Binding ElementName=CornerRadius,Path=Value}" 
                        TriangleSize="{Binding ElementName=TriangleSize,Path=Value}">
            <control:Bubble.Content>
                <Grid>
                    <TextBlock Text="Bottom" VerticalAlignment="Center" HorizontalAlignment="Center" Foreground="White"/>
                </Grid>
            </control:Bubble.Content>
        </control:Bubble>
        <!--调整-->
        <Grid Grid.Row="1" Grid.Column="1" Width="220" HorizontalAlignment="Center" VerticalAlignment="Center">
            <StackPanel Orientation="Vertical">
                <TextBlock Foreground="White">
                    <Run Text="气泡边框粗细："/>
                    <Run Text="{Binding ElementName=BubbleThickness,Path=Value}"/>
                </TextBlock>
                <Slider x:Name="BubbleThickness" Minimum="0" Maximum="10" Value="1"/>

                <TextBlock Foreground="White">
                    <Run Text="气泡三角大小："/>
                    <Run Text="{Binding ElementName=TriangleSize,Path=Value}"/>
                </TextBlock>
                <Slider x:Name="TriangleSize" Minimum="0" Maximum="20" Value="10"/>

                <TextBlock Foreground="White">
                    <Run Text="气泡圆角大小："/>
                    <Run Text="{Binding ElementName=CornerRadius,Path=Value}"/>
                </TextBlock>
                <Slider x:Name="CornerRadius" Minimum="0" Maximum="40" Value="8"/>

                <TextBlock Foreground="White">
                    <Run Text="气泡透明度："/>
                    <Run Text="{Binding ElementName=BubbleOpacity,Path=Value}"/>
                </TextBlock>
                <Slider x:Name="BubbleOpacity" Minimum="0" Maximum="1" Value="0.6"/>
            </StackPanel>
        </Grid>
    </Grid>
</Window>
```