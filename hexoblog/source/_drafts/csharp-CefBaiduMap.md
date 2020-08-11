---
title: 基于 CEF 控件在桌面应用中加载百度地图
date: 2020-8-11 20:51:12
tags: [c#,wpf,cef,baidu]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-CefBaiduMap/CefBaiduMapDemo.png"/>

### 在 WPF 中调用 CEF 加载百度地图
<!-- more -->
#### 简介
最近有需求需要把带定位的百度地图加载到桌面应用中
#### 前置条件
[使用 CEF 在 PC 客户端中加载网页](https://sadness96.github.io/blog/2020/08/11/csharp-CefSharp/)
#### 开发 BaiduMap 用户控件
##### 创建带参数百度地图 WEB 页面
map.baidu.html?Lon=116.4716&Lat=40.01849
``` html
<html>

<head>
    <script type="text/javascript" src="https://cdn.bootcss.com/jquery/3.4.1/jquery.min.js"></script>
    <script type="text/javascript" src="https://api.map.baidu.com/api?v=3.0&您的ak"></script>
    <style type="text/css">
        body {
            margin: 0;
        }
        
        #allmap {
            width: 100%;
            height: 100%;
            overflow: hidden;
            margin: 0;
        }
    </style>
</head>

<body>
    <div id="allmap"></div>
</body>

<script>
    //获取经纬度参数
    function getUrlParam(name) {
        var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)");
        var r = window.location.search.substr(1).match(reg);
        if (r != null) {
            return unescape(r[2]);
        } else {
            return null;
        }
    }
    var vLon = getUrlParam('Lon');
    var vLat = getUrlParam('Lat');
    //加载百度地图
    //创建Map实例
    var map = new BMap.Map("allmap");
    //创建点坐标
    var point = new BMap.Point(vLon, vLat);
    map.centerAndZoom(point, 17);
    //启用滚轮放大缩小
    map.enableScrollWheelZoom();
    //创建标注
    var marker = new BMap.Marker(point);
    //将标注添加到地图中
    map.addOverlay(marker);
</script>

</html>
```
##### BaiduMap.xaml 用户控件
``` XML
<UserControl x:Class="Ice.BaiduMap.Control.BaiduMap"
             xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
             xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" 
             xmlns:d="http://schemas.microsoft.com/expression/blend/2008" 
             xmlns:chrome="clr-namespace:CefSharp.Wpf;assembly=CefSharp.Wpf"
             xmlns:local="clr-namespace:Ice.BaiduMap.Control"
             mc:Ignorable="d" 
             d:DesignHeight="450" d:DesignWidth="800">
    <Grid>
        <chrome:ChromiumWebBrowser/>
    </Grid>
</UserControl>
```
##### BaiduMap.xaml.cs 用户控件后台
``` CSharp
using CefSharp.Wpf;
using System;
using System.IO;
using System.Windows;
using System.Windows.Controls;

namespace Ice.BaiduMap.Control
{
    /// <summary>
    /// BaiduMap.xaml 的交互逻辑
    /// </summary>
    public partial class BaiduMap : UserControl
    {
        public BaiduMap()
        {
            InitializeComponent();
            this.Loaded += BaiduMap_Loaded;
        }

        private void BaiduMap_Loaded(object sender, RoutedEventArgs e)
        {
            if (File.Exists(_webapp_baidumap_path) && !string.IsNullOrEmpty(Lon) && !string.IsNullOrEmpty(Lat))
            {
                var webView = new ChromiumWebBrowser();
                this.Content = webView;
                webView.Address = $"{_webapp_baidumap_path}?Lon={Lon}&Lat={Lat}";
            }
        }

        /// <summary>
        /// 百度地图加载文件
        /// </summary>
        private string _webapp_baidumap_path = $"{AppDomain.CurrentDomain.BaseDirectory}WebApp\\map.baidu.html";

        /// <summary>
        /// 经度
        /// </summary>
        public string Lon
        {
            get { return (string)GetValue(LonProperty); }
            set { SetValue(LonProperty, value); }
        }
        public static readonly DependencyProperty LonProperty =
            DependencyProperty.Register("Lon", typeof(string), typeof(BaiduMap));

        /// <summary>
        /// 纬度
        /// </summary>
        public string Lat
        {
            get { return (string)GetValue(LatProperty); }
            set { SetValue(LatProperty, value); }
        }
        public static readonly DependencyProperty LatProperty =
            DependencyProperty.Register("Lat", typeof(string), typeof(BaiduMap));
    }
}
```
##### 在使用的地方引用
``` XML
<Grid>
    <control:BaiduMap Lon="116.4716" Lat="40.01849"/>
</Grid>
```