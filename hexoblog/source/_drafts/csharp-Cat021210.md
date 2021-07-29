---
title: Cat021 报文解析（兼容 2.1 版本）
date: 2021-07-29 20:55:00
tags: [c#]
categories: C#.Net
---
### ADS-B CAT021 2.1 报文协议解析

<!-- more -->
### 简介/声明
作为 2.1 版本的补充，前置内容 [点此查看](https://sadness96.github.io/blog/2019/08/19/csharp-Cat021026/)

### 代码
<span id="1021071"><span/>

#### 1021/071
``` csharp
/// <summary>
/// 计算日时间项(I021/071)对应的值
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static string I021_071(byte[] byteData)
{
    return I021_030(byteData);
}
```
<span id="1021130"><span/>

#### 1021/130
``` csharp
/// <summary>
/// 计算位置坐标(WGS-84中)项(I021/130)对应的值
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static double[] I021_130(byte[] byteData)
{
    if (byteData.Length == 6)
    {
        return I021_131(byteData);
    }
    else if (byteData.Length == 8)
    {
        double[] res = { 0, 0 };
        int value1;
        //将容器中前4个字节合并为一个字节，用以计算纬度。
        value1 = (byteData[0] << 24) + (byteData[1] << 16) + (byteData[2] << 8) + byteData[3];
        double temp1 = value1 * (5.364418e-6);
        //Console.WriteLine($"坐标值:纬度值{temp1}");
        res[1] = temp1;
        int value0;
        //将容器中后4个字节合并为一个字节，用以计算经度。
        value0 = (byteData[4] << 24) + (byteData[5] << 16) + (byteData[6] << 8) + byteData[7];
        double temp0 = value0 * (5.364418e-6);
        //Console.WriteLine($"经度值{temp0}");
        res[0] = temp0;
        return res;
    }
    return null;
}
```
<span id="1021131"><span/>

#### 1021/131
``` csharp
/// <summary>
/// 计算位置坐标(WGS-84中)项(I021/131)对应的值
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static double[] I021_131(byte[] byteData)
{
    var len = byteData.Length;
    var startIndex = 0;
    var Len6Ruler = 180 / Math.Pow(2, 23);
    var Len8Ruler = 180 / Math.Pow(2, 30);
    //根据长度确定转换标尺
    var ruler = len == 6 ? Len6Ruler : Len8Ruler;
    var res = new double[] { 0, 0 };

    int startValue = 0;
    byte lshBit = 0;
    //将容器中前一半字节合并为一个字节，用以计算纬度。
    for (int i = startIndex + len / 2 - 1; i >= startIndex; i--)
    {
        startValue += (int)byteData[i] << lshBit;
        lshBit += 8;
    }
    res[1] = startValue * ruler;

    int endValue = 0;
    lshBit = 0;
    //将容器中后一半字节合并为一个字节，用以计算经度。
    for (int i = startIndex + len - 1; i >= startIndex + len / 2; i--)
    {
        endValue += (int)byteData[i] << lshBit;
        lshBit += 8;
    }

    res[0] = endValue * ruler;
    return res;
}
```