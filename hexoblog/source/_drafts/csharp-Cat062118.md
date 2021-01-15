---
title: Cat062 报文解析（未解析）
date: 2019-10-15 15:40:00
tags: [c#]
categories: C#.Net
---
### Radar CAT062 1.18 报文协议解析

<!-- more -->
### 简介/声明
[Radar](https://baike.baidu.com/item/%E9%9B%B7%E8%BE%BE/10485?fr=aladdin) 雷达，源于radio detection and ranging的缩写，意思为"无线电探测和测距"，即用无线电的方法发现目标并测定它们的空间位置。因此，雷达也被称为“无线电定位”。雷达是利用电磁波探测目标的电子设备。雷达发射电磁波对目标进行照射并接收其回波，由此获得目标至电磁波发射点的距离、距离变化率（径向速度）、方位、高度等信息。

解析文档均为[欧洲航空交通管理](https://www.eurocontrol.int/)官方提供。

## 参考资料
### 原文
[EuroControl](https://www.eurocontrol.int)：[cat062p9ed118.pdf](https://www.eurocontrol.int/sites/default/files/content/documents/nm/asterix/cat062p9ed118.pdf)

## 测试数据解析
3E0034BB7D25040203000E584F003806E501460641FD2601B70D4A000D33B3C37E2080780CCB000601000550000028002A003E04

### 解析步骤
#### 数据格式
| CAT = 062 | LEN | FSPEC | Items of the first record |
| ---- | ---- | ---- | ---- |

#### 解析报文区域数据
| 16进制代码 | 解析值（二进制或十进制） | 备注 |
| ---- | ---- | ---- |
| 0x3E | 62 | 报文头，转换十进制为62 |
| 0x00 | 00 | 报文长度起始位 |
| 0x24 | 36 | 报文长度 LEN，为报文数据字节数，两个字节表示，该报文长度为0x00*256+0x24=0x24=36字节 |
| 97 | 10010111 | I062/010、I062/070、I062/100、I062/185 |
| 39 | 00111001 | I062/245、I062/380、I062/040 |
| 08 | 00001000 | I062/135 |
| 01 |  |  |
| 01 |  |  |
| 04 | 00000100 |  |
| 27 | 00100111 |  |
| 8A | 10001010 | 0:35:27 |
| 00 |  |  |
| 01 |  |  |
| C8 |  |  |
| 00 |  |  |
| 02 |  |  |
| BE |  |  |
| 00 |  |  |
| 00 |  |  |
| 00 |  |  |
| 00 |  |  |
| 40 |  |  |
| 0C |  |  |
| 80 |  |  |
| B6 |  |  |
| D3 |  |  |
| 5C |  |  |
| 60 |  |  |
| 80 |  |  |
| 78 |  |  |
| 0C |  |  |
| CA |  |  |
| 01 |  |  |
| E5 |  |  |
| 00 |  |  |
| 00 |  |  |
| 01 |  |  |
| D0 |  |  |
| FF |  |  |
| C8 |  |  |
| 01 |  |  |
| 01 |  |  |
| 43 |  |  |
| 47 |  |  |
| 48 |  |  |
| 31 |  |  |
| 30 |  |  |
| 31 |  |  |
| 39 |  |  |
| 00 |  |  |
| 00 |  |  |
| 0C |  |  |
| A3 |  |  |
| 80 |  |  |
| 41 |  |  |
| 33 |  |  |
| 32 |  |  |
| 30 |  |  |
| 4D |  |  |
| 5A |  |  |
| 47 |  |  |
| 4B |  |  |
| 4C |  |  |
| 56 |  |  |
| 54 |  |  |
| 42 |  |  |
| 53 |  |  |
| 31 |  |  |
| 32 |  |  |
| 4C |  |  |
| 03 |  |  |
| 16 |  |  |
| 10 |  |  |
| 0A |  |  |
| 00 |  |  |
| 3E |  |  |
| 10 |  |  |
| 0E |  |  |
| 00 |  |  |
| 40 |  |  |
| 00 |  |  |
| 37 |  |  |
| 00 |  |  |
| 00 |  |  |
| 00 |  |  |
| 20 |  |  |
| 00 |  |  |
| 20 |  |  |
| 20 |  |  |
| 36 |  |  |
| 20 |  |  |
| 20 |  |  |
| 20 |  |  |

### 代码
<span id="1062070"><span/>

#### I062/070
``` csharp
/// <summary>
/// 计算日时间项(I062/070)对应的值
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static string I062_070(byte[] byteData)
{
    //将几个独立字节合并为一个字节
    uint rhs = ((uint)byteData[0] << 16) + ((uint)byteData[1] << 8) + (uint)byteData[2];
    //总秒数
    uint value0 = rhs / 128;
    //小时数
    uint value1 = value0 / 3600;
    //分钟数
    uint value2 = (value0 - value1 * 3600) / 60;
    //秒数
    uint value3 = (value0 - value1 * 3600) % 60;
    //毫秒数
    uint value4 = ((rhs % 128) * 1000) / 128;
    return $"{DateTime.Now.ToShortDateString()} {value1}:{value2}:{value3}.{value4}";
}
```

<span id="1062245"><span/>

#### I062/245
``` csharp
/// <summary>
/// 解析(I062_245)目标识别
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static string I062_245(byte[] byteData)
{
    string str = "";
    for (int i = 1; i < byteData.Length; i++)
    {
        // 把第一位去掉
        str += Convert.ToString(byteData[i], 2).PadLeft(8, '0');
    }

    char[] strCharArray = str.ToCharArray();
    string flno2BinaryStr = "";
    string result = "";

    for (int i = 0; i < strCharArray.Length; i++)
    {
        flno2BinaryStr += strCharArray[i] + "";
        if ((i + 1) % 6 == 0)
        {
            string flightNumberValue = Constants.flightNumberMap[flno2BinaryStr];
            if (!string.IsNullOrEmpty(flightNumberValue))
            {
                result += flightNumberValue;
            }
            flno2BinaryStr = "";
        }
    }
    return result;
}
```

<span id="1062105"><span/>

#### I062/105
``` csharp
/// <summary>
/// 解析(I062_070)经纬度坐标
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static double[] I062_105(byte[] byteData)
{
    double[] relDataArray = new double[2];
    if (byteData.Length == 8)
    {
        // 16进制转成10进制（4位一转）
        string xCoordinate10 = byteData[0].ToString("X2") + byteData[1].ToString("X2") + byteData[2].ToString("X2") + byteData[3].ToString("X2");
        string yCoordinate10 = byteData[4].ToString("X2") + byteData[5].ToString("X2") + byteData[6].ToString("X2") + byteData[7].ToString("X2");
        // 10进制计算规则（xCoordinate10 * 180 /2^25）
        relDataArray[0] = double.Parse(Convert.ToInt32(xCoordinate10, 16).ToString()) * 180 / 33554432;
        relDataArray[1] = double.Parse(Convert.ToInt32(yCoordinate10, 16).ToString()) * 180 / 33554432;
        return relDataArray;
    }
    return null;
}
```

<span id="1062100"><span/>

#### I062/100
``` csharp
/// <summary>
/// 解析(I062_100)卡迪尔坐标
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static double[] I062_100(byte[] byteData)
{
    double[] relDataArray = new double[2];
    if (byteData.Length == 6)
    {
        string xAngle16 = byteData[0].ToString("X2") + byteData[1].ToString("X2") + byteData[2].ToString("X2");
        string yAngle16 = byteData[3].ToString("X2") + byteData[4].ToString("X2") + byteData[5].ToString("X2");
        string xAngle10 = Convert.ToInt32(xAngle16, 16).ToString();
        string yAngle10 = Convert.ToInt32(yAngle16, 16).ToString();
        // 10进制计算规则（xAngle10 * 0.5）
        relDataArray[0] = double.Parse(xAngle10) * 0.5;
        relDataArray[1] = double.Parse(yAngle10) * 0.5;
        return relDataArray;
    }
    return null;
}
```