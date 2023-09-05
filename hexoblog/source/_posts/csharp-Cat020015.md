---
title: Cat020 报文解析
date: 2019-09-09 09:45:00
tags: [c#]
categories: C#.Net
---
### MLAT CAT020 0.15 报文协议解析

<!-- more -->
### 简介/声明
多点定位(MLAT) 全称是 Multilateration，多点定位技术利用多个地面接收机接收到机载应答机信号的时间差，计算得出飞机位置。

解析文档均为[欧洲航空交通管理](https://www.eurocontrol.int/)官方提供。

### 参考资料
[EuroControl](https://www.eurocontrol.int)：[cat020p14ed15.pdf](https://www.eurocontrol.int/sites/default/files/2019-06/cat020-asterix-mlt-messages-part-14.pdf)

### 解析数据
``` txt
14 00 46 FF 0F 01 84 16 07 41 10 A1 A0 BB 00 57 8B 48 01 44 DC F6 00 17 06 00 1F AD 0E F2 02 78 10 45 80 0C 54 F2 DB 3C 60 00 02 20 40 19 98 D0 00 00 00 00 00 01 00 0C 00 0C 00 03 00 06 00 05 00 05 A1 A0 C2 00
```

#### 数据格式
| CAT = 020 | LEN | FSPEC | Items of the first record |
| ---- | ---- | ---- | ---- |

#### 解析报文区域数据
| 16进制代码 | 解析值（二进制或十进制） | 备注 |
| ---- | ---- | ---- |
| 0x14 | 20 | 报文头，转换十进制为20 |
| 0x00 | 00 | 报文长度起始位 |
| 0x46 | 70 | 报文长度 LEN，为报文数据字节数，两个字节表示，该报文长度为0x00*256+0x46=0x46=70字节 |
| FF | 1111 1111 | I020/010、I020/020、I020/140、I020/041、I020/042、I020/161、I020/170 |
| 0F | 0000 1111 | I020/220、I020/245、I020/110 |
| 01 | 0000 0001 |  |
| 84 | 1000 0100 | I020/230、RE |
| 16 | 22 |  |
| 07 | 7 | I020/010 数据源的标识符 |
| 41 | 65 |  |
| 10 | 16 | I020/020 目标报告描述符 |
| A1 | 161 |  |
| A0 | 160 |  |
| BB | 187 | [I020/140](#1020140) 日时间项 |
| 00 | 0 |  |
| 57 | 87 |  |
| 8B | 139 |  |
| 48 | 72 |  |
| 01 | 1 |  |
| 44 | 68 |  |
| DC | 220 |  |
| F6 | 246 | [I020/041](#1020041) 位置坐标（WGS-84）项 |
| 00 | 0 |  |
| 17 | 23 |  |
| 06 | 6 |  |
| 00 | 0 |  |
| 1F | 31 |  |
| AD | 173 | [I020/042](#1020042) 在笛卡尔坐标中的位置 |
| 0E | 14 |  |
| F2 | 242 | [I020/161](#1020161) 跟踪号码 |
| 02 | 2 | I020/170 追踪发送状态 |
| 78 | 120 |  |
| 10 | 16 |  |
| 45 | 69 | I020/220 目标地址 |
| 80 | 128 |  |
| 0C | 12 |  |
| 54 | 84 |  |
| F2 | 242 |  |
| DB | 219 |  |
| 3C | 60 |  |
| 60 | 96 | [I020/245](#1020245) 目标识别 航班号 |
| 00 | 0 |  |
| 02 | 2 | [I020/110](#1020110) 测量高度(笛卡尔坐标) |
| 20 | 32 |  |
| 40 | 64 | I020/230 通讯/自动识别系统的能力和飞行状态 |
| 19 |  |  |
| 98 |  |  |
| D0 |  |  |
| 00 |  |  |
| 00 |  |  |
| 00 |  |  |
| 00 |  |  |
| 00 |  |  |
| 01 |  |  |
| 00 |  |  |
| 0C |  |  |
| 00 |  |  |
| 0C |  |  |
| 00 |  |  |
| 03 |  |  |
| 00 |  |  |
| 06 |  |  |
| 00 |  |  |
| 05 |  |  |
| 00 |  |  |
| 05 |  |  |
| A1 |  |  |
| A0 |  |  |
| C2 |  |  |
| 00 |  |  |

### 代码
<span id="1020140"><span/>

#### I020/140
``` csharp
/// <summary>
/// 解析I020_140日时间
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static string I020_140(byte[] byteData)
{
    // 16进制转成10进制
    string timeDec = (((uint)byteData[0] << 16) + ((uint)byteData[1] << 8) + byteData[2]).ToString();

    // 字符串转数值/128 * 1000 总毫秒数
    long ms = (long)((double.Parse(timeDec) / 128) * 1000);

    int ss = 1000;
    int mi = ss * 60;
    int hh = mi * 60;

    long hour = ms / hh;
    long minute = (ms - hour * hh) / mi;
    long second = (ms - hour * hh - minute * mi) / ss;
    long milliSecond = ms - hour * hh - minute * mi - second * ss;

    // 小时
    string strHour = hour < 10 ? "0" + hour : "" + hour;
    // 分钟
    string strMinute = minute < 10 ? "0" + minute : "" + minute;
    // 秒
    string strSecond = second < 10 ? "0" + second : "" + second;
    // 毫秒
    string strMilliSecond = milliSecond < 10 ? "0" + milliSecond : "" + milliSecond;
    strMilliSecond = milliSecond < 100 ? "0" + strMilliSecond : "" + strMilliSecond;
    //增加UTC时间
    strHour = int.Parse(strHour) + 8 > 24 ? (int.Parse(strHour) + 8 - 24).ToString() : (int.Parse(strHour) + 8).ToString();
    return $"{DateTime.Now.ToShortDateString()} {strHour}:{strMinute}:{strSecond}.{strMilliSecond}";
}
```

<span id="1020245"><span/>

#### I020/245
``` csharp
/// <summary>
/// 解析I020/245目标识别
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static string I020_245(byte[] byteData)
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
            string flightNumberValue = flightNumberMap[flno2BinaryStr];
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
NOTE: See ICAO document Annex 10, Volume IV, section 3.1.2.9
for the coding rules.
每六字节代表一个字母或数字
<img src="https://sadness96.github.io/images/blog/csharp-Cat020015/TheCodingRules.png"/>

<span id="1020041"><span/>

#### I020/041
``` csharp
/// <summary>
/// 解析I020_041在WGS-84中的坐标位置
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static double[] I020_041(byte[] byteData)
{
    double[] relDataArray = new double[2];
    if (byteData.Length == 8)
    {
        // 16进制转成10进制（4位一转）
        string xCoordinate10 = byteData[0].ToString("X2") + byteData[1].ToString("X2") + byteData[2].ToString("X2") + byteData[3].ToString("X2");
        string yCoordinate10 = byteData[4].ToString("X2") + byteData[5].ToString("X2") + byteData[6].ToString("X2") + byteData[7].ToString("X2");
        // 10进制计算规则（xCoordinate10 * 180 /2^25）
        relDataArray[0] = double.Parse(Convert.ToInt32(xCoordinate10, 16).ToString()) * 180 / Math.Pow(2, 25);
        relDataArray[1] = double.Parse(Convert.ToInt32(yCoordinate10, 16).ToString()) * 180 / Math.Pow(2, 25);
        return relDataArray;
    }
    return null;
}
```

<span id="1020042"><span/>

#### I020/042
``` csharp
/// <summary>
/// 解析I020_042轨道位置(直角)
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static double[] I020_042(byte[] byteData)
{
    double[] relDataArray = new double[2];
    if (byteData.Length == 6)
    {
        // 16进制转成10进制
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

<span id="1020161"><span/>

#### I020/161
``` csharp
/// <summary>
/// 解析I020_161
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static int I020_161(byte[] byteData)
{
    return Convert.ToInt32(byteData[0].ToString("X2") + byteData[1].ToString("X2"), 16);
}
```

<span id="1020110"><span/>

#### I020/110
``` csharp
/// <summary>
/// 解析I020_110
/// </summary>
/// <param name="byteData">二进制数据</param>
/// <returns></returns>
public static double I020_110(byte[] byteData)
{
    string strByteData = byteData[0].ToString("X2") + byteData[1].ToString("X2");
    double dByteData = Convert.ToInt32(strByteData, 16);

    if (Convert.ToString(byteData[0], 2).Substring(0, 1).Equals("1"))
    {
        // 如果2进制长度为16，说明第16位一定为1，则为负数
        return -(Math.Pow(2, 16) - dByteData) * 6.25;
    }
    else
    {
        // 如果2进制长度不为16，说明第16位一定为0，则为正数
        return dByteData * 6.25;
    }
}
```