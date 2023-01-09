---
title: UTM WGS84 互相转换
date: 2023-01-09 19:52:28
tags: [c#]
categories: C#.Net
---
### Mercator 投影与 WGS-84坐标系互相转换
<!-- more -->
### 简介
[WGS84 坐标系](https://en.wikipedia.org/wiki/World_Geodetic_System) 是为GPS全球定位系统使用而建立的坐标系统。
[Mercator (墨卡托投影)](https://en.wikipedia.org/wiki/Mercator_projection) 是一种地图投影系统，用于为地球表面的位置分配坐标，可以参考别人编制的 [UTM Grid](https://www.dmap.co.uk/utmworld.htm)，例如中国东部属于 UTM Zone 50N。

<img src="https://www.dmap.co.uk/utmworld.gif"/>

### 代码
引用 [ProjNet](https://github.com/NetTopologySuite/ProjNet4GeoAPI) 库，ProjNet 是 [Proj](https://proj.org/) 的 .Net 版本，是一种通用坐标转换软件，可将地理空间坐标从一个坐标参考系统 (CRS) 转换到另一个坐标参考系统。
``` csharp
/// <summary>
/// Transform UTM to WGS84
/// </summary>
/// <param name="points">UTM lng lat</param>
/// <param name="zone">UTM zone</param>
/// <param name="north">true of Northern hemisphere, false if southern</param>
/// <returns>WGS84</returns>
private static double[] TransformUtm32ToWgs84(double[] points, int zone = 50, bool north = true)
{
    CoordinateTransformationFactory cstFactory = new CoordinateTransformationFactory();
    ICoordinateTransformation utm32ToWgs84 = cstFactory.CreateFromCoordinateSystems(
        ProjectedCoordinateSystem.WGS84_UTM(zone, north),
        GeographicCoordinateSystem.WGS84
    );
    return utm32ToWgs84.MathTransform.Transform(points);
}

/// <summary>
/// Transform WGS84 to UTM
/// </summary>
/// <param name="points">WGS84 lng lat</param>
/// <param name="zone">UTM zone</param>
/// <param name="north">true of Northern hemisphere, false if southern</param>
/// <returns>UTM</returns>
private static double[] TransformWgs84ToUtm32(double[] points, int zone = 50, bool north = true)
{
    CoordinateTransformationFactory cstFactory = new CoordinateTransformationFactory();
    ICoordinateTransformation wgs84ToUtm32 = cstFactory.CreateFromCoordinateSystems(
        GeographicCoordinateSystem.WGS84,
        ProjectedCoordinateSystem.WGS84_UTM(zone, north)
    );
    return wgs84ToUtm32.MathTransform.Transform(points);
}

static void Main(string[] args)
{
    var output1 = TransformUtm32ToWgs84(new[] { 447617.70449733676, 4429247.0759452293 });
    var output2 = TransformWgs84ToUtm32(new[] { 116.386231, 40.011798 });
}
```