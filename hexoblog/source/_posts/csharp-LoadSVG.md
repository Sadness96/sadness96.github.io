---
title: WPF 加载 SVG
date: 2018-06-10 12:15:42
tags: [c#,wpf]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-LoadSVG/svgwindow.jpg"/>

<!-- more -->
### 简介
[SVG](https://baike.baidu.com/item/SVG格式/3463453) 是一种可缩放的 [矢量图形](https://baike.baidu.com/item/矢量图)，在软件开发中，使用图片加载如果分辨率不足被拉伸后会导致模糊，尽可能使用矢量图形开发，减少项目大小的同时又保证 UI 的质量。
自己绘制 SVG 推荐使用 [Adobe Illustrator](https://www.adobe.com/products/illustrator.html)、[Vectornator](https://www.vectornator.io/)、[SVG 在线编辑器](https://c.runoob.com/more/svgeditor/)。
网络资源推荐使用 [iconfont](https://www.iconfont.cn/)、[undraw](https://undraw.co/illustrations)

### 代码
#### 加载 Path 路径
SVG 又一些基础组件构成，例如 点、线、方形、圆形，而使用最多也是最重要的就是钢笔工具，体现在 SVG 中就是 Path 路径，这个方法就是仅使用钢笔绘制的图形加载，需要把图形左上角对齐后才更方便调整位置。
``` XML
<Viewbox Margin="30">
    <Path Data="m251,34.2l-164.8,0c-47.5,0 -86.2,38.7 -86.2,86.2l0,164.8c0,47.5 38.7,86.2 86.2,86.2l164.8,0c47.5,0 86.2,-38.7 86.2,-86.2l0,-164.8c0.1,-47.6 -38.6,-86.2 -86.2,-86.2zm-9.9,241.1l-145,0l0,-145l145,0l0,145zm9.9,150.8l-164.8,0c-47.5,0 -86.2,38.7 -86.2,86.2l0,164.8c0,47.5 38.7,86.2 86.2,86.2l164.8,0c47.5,0 86.2,-38.7 86.2,-86.2l0,-164.8c0.1,-47.5 -38.6,-86.2 -86.2,-86.2zm-9.9,241.1l-145,0l0,-145l145,0l0,145zm499.3,-525.4l-116.5,-116.5c-16.3,-16.3 -37.9,-25.3 -61,-25.3c-23,0 -44.7,9 -61,25.3l-116.5,116.5c-33.6,33.6 -33.6,88.3 0,121.9l116.6,116.5c16.3,16.3 37.9,25.3 61,25.3c23,0 44.7,-9 61,-25.3l116.6,-116.6c33.4,-33.4 33.4,-88.1 -0.2,-121.8zm-177.5,163.5l-102.5,-102.5l102.5,-102.5l102.5,102.5l-102.5,102.5zm82.4,120.8l-164.8,0c-47.5,0 -86.2,38.7 -86.2,86.2l0,164.8c0,47.5 38.7,86.2 86.2,86.2l164.8,0c47.5,0 86.2,-38.7 86.2,-86.2l0,-164.8c0,-47.5 -38.7,-86.2 -86.2,-86.2zm-9.9,241.1l-145,0l0,-145l145,0l0,145z" Fill="Black"/>
</Viewbox>
```

#### 加载 DrawingImage
使用开源库 [SvgToXaml](https://github.com/BerndK/SvgToXaml) 可以更方便的加载 SVG，转换为 DrawingImage 后放在 Imgae 控件中即可。
<img src="https://raw.githubusercontent.com/BerndK/SvgToXaml/master/Doc/MainView.PNG"/>

<img src="https://raw.githubusercontent.com/BerndK/SvgToXaml/master/Doc/DetailViewXaml.PNG"/>

``` XML
<Image>
    <Image.Source>
        <DrawingImage>
            <DrawingImage.Drawing>
                <DrawingGroup ClipGeometry="M0,0 V1024 H1024 V0 H0 Z">
                    <DrawingGroup Opacity="1">
                        <GeometryDrawing Brush="#FF000000" Geometry="F1 M1024,1024z M0,0z M382.2,165.7L217.4,165.7C169.9,165.7,131.2,204.4,131.2,251.9L131.2,416.7C131.2,464.2,169.9,502.9,217.4,502.9L382.2,502.9C429.7,502.9,468.4,464.2,468.4,416.7L468.4,251.9C468.5,204.3,429.8,165.7,382.2,165.7z M372.3,406.8L227.3,406.8 227.3,261.8 372.3,261.8 372.3,406.8z M382.2,557.6L217.4,557.6C169.9,557.6,131.2,596.3,131.2,643.8L131.2,808.6C131.2,856.1,169.9,894.8,217.4,894.8L382.2,894.8C429.7,894.8,468.4,856.1,468.4,808.6L468.4,643.8C468.5,596.3,429.8,557.6,382.2,557.6z M372.3,798.7L227.3,798.7 227.3,653.7 372.3,653.7 372.3,798.7z M871.6,273.3L755.1,156.8C738.8,140.5 717.2,131.5 694.1,131.5 671.1,131.5 649.4,140.5 633.1,156.8L516.6,273.3C483,306.9,483,361.6,516.6,395.2L633.2,511.7C649.5,528 671.1,537 694.2,537 717.2,537 738.9,528 755.2,511.7L871.8,395.1C905.2,361.7,905.2,307,871.6,273.3z M694.1,436.8L591.6,334.3 694.1,231.8 796.6,334.3 694.1,436.8z M776.5,557.6L611.7,557.6C564.2,557.6,525.5,596.3,525.5,643.8L525.5,808.6C525.5,856.1,564.2,894.8,611.7,894.8L776.5,894.8C824,894.8,862.7,856.1,862.7,808.6L862.7,643.8C862.7,596.3,824,557.6,776.5,557.6z M766.6,798.7L621.6,798.7 621.6,653.7 766.6,653.7 766.6,798.7z" />
                    </DrawingGroup>
                </DrawingGroup>
            </DrawingImage.Drawing>
        </DrawingImage>
    </Image.Source>
</Image>
```