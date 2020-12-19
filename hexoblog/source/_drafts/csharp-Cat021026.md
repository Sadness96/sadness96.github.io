---
title: Cat021 报文解析
date: 2019-08-19 12:49:00
tags: [c#]
categories: C#.Net
---
# ADS-B 报文解析
ADS-B CAT021 0.26 报文协议解析
<!-- more -->
### 简介/声明
[ADS-B](https://baike.baidu.com/item/ADS-B/9750451?fr=aladdin) 全称是Automatic Dependent Surveillance - Broadcast，中文是广播式自动相关监视，顾名思义，即无需人工操作或者询问，可以自动地从相关机载设备获取飞机或地面站广播飞机的位置、高度、速度、航向、识别号等参数信息，以供管制员对飞机状态进行监控。它衍生于ADS（自动相关监视），最初是为越洋飞行的航空器在无法进行雷达监视的情况下，希望利用卫星实施监视所提出的解决方案。

解析文档均为[欧洲航空交通管理](https://www.eurocontrol.int/)官方提供，解析代码基于 CSDN 付费文档修改，仅包含极少部分公司业务格式不包含涉密文件。
## 参考资料
### 原文
[EuroControl](https://www.eurocontrol.int)：[cat021p12ed026.pdf](https://www.eurocontrol.int/sites/default/files/content/documents/nm/asterix/20150615-asterix-adsbtr-cat021-part12-v2.4.pdf)
### 中文文档
https://wenku.baidu.com/view/d94cb4fcd1f34693daef3efd
### 参考实验报告
https://wenku.baidu.com/view/67d8787ccc7931b764ce150b.html
### 参考代码
https://blog.csdn.net/qingfengleerge/article/details/81102854

## 测试数据解析
20190816:130000.052 15002efba1df80000100302327660055a0b60144ae0a7802610006080388000a077e043e0d33b3c72de000800002
### 解析步骤
#### 根据空格分隔报文
```csharp
static byte[] GetBytesFromADSB(string strADSB)
{
    string[] strsADSB = strADSB.Split(' ');
    return StringToByte(strsADSB[1]);
}
```
| 数据 | 备注 |
| ---- | ---- |
| 20190816:130000.052 | 时间戳 |
| 15002efba1df80000100302… | ADS-B 数据报文 |
#### 数据报文区域每两位一组16进制数据
15002e 解析为：0x15、0x00、0x2e
``` csharp
static byte[] StringToByte(string str)
{
    if (str.Length % 2 != 0)
    {
        str = str + "0";
    }
    byte[] temp = new byte[str.Length / 2];
    for (int i = 0; i < temp.Length; i++)
    {
        temp[i] = Convert.ToByte(str.Substring(i * 2, 2), 16);
    }
    return temp;
}
```
#### 数据格式
| CAT = 021 | LEN | FSPEC | Items of the first record |
| ---- | ---- | ---- | ---- |
#### 解析报文区域数据
| 16进制代码 | 解析值（二进制或十进制） | 备注 |
| ---- | ---- | ---- |
| 0x15 | 21 | 报文头，转换十进制为21 |
| 0x00 | 00 | 报文长度起始位 |
| 0x2e | 46 | 报文长度 LEN，为报文数据字节数，两个字节表示，该报文长度为0x00*256+0x2e=0x2e=46字节 |
| 0xfb | 1111 1011 | 1021/010，1021/040，1021/030，1021/130，1021/080，1021/090 |
| 0xa1 | 1010 0001 | 1021/210，1021/145 |
| 0xdf | 1101 1111 | 1021/157，1021/160，1021/170，1021/095，1021/032，1021/200 |
| 0x80 | 1000 0000 | 1021/020（FSPEC 字段，该字段可变，(x<<=7;x>>=7)==0 下一条数据不为 FSPEC） |
| 0x00 |  |  |
| 0x01 |  | 1021/010 |
| 0x00 |  |  |
| 0x30 |  | 1021/040 |
| 0x23 | 35 |  |
| 0x27 | 39 |  |
| 0x66 | 102 | [1021/030](#1021030) 日时间项 |
| 0x00 | 0 |  |
| 0x55 | 85 |  |
| 0xa0 | 160 |  |
| 0xb6 | 182 |  |
| 0x01 | 1 |  |
| 0x44 | 68 |  |
| 0xae | 174 |  |
| 0x0a | 10 | [1021/130](#1021130) 位置坐标（WGS-84）项，前四位为纬度值，后四位为经度值 |
| 0x78 | 120 |  |
| 0x02 | 2 |  |
| 0x61 | 97 | [1021/080](#1021080) 目标地址项 |
| 0x00 |  |  |
| 0x06 |  | 1021/090 |
| 0x08 |  | 1021/210 |
| 0x03 |  |  |
| 0x88 |  | 1021/145 |
| 0x00 | 0 |  |
| 0x0a | 10 | [1021/157](#1021157) |
| 0x07 |  |  |
| 0x7e |  |  |
| 0x04 |  |  |
| 0x3e |  | 1021/160 |
| 0x0d | 13 |  |
| 0x33 | 51 |  |
| 0xb3 | 179 |  |
| 0xc7 | 199 |  |
| 0x2d | 45 |  |
| 0xe0 | 224 | [1021/170](#1021170) |
| 0x00 |  | 1021/095 |
| 0x80 |  | 1021/032 |
| 0x00 |  | 1021/200 |
| 0x02 |  | 1021/020 |
#### 项目需要最终生成 Json 数据
``` json
{
    "FlightTag": {
        "FlightNumer": "CSN3127 ", 
        "Reg": "780261", 
        "Lat": 30.103516, 
        "Lon": 114.145256, 
        "Stand": "", 
        "PlaneType": "", 
        "Runway": "", 
        "Direction": 0, 
        "Height": 0, 
        "StandSide": "", 
        "UpdateTime": "2019/8/16 4:59:58.796", 
        "CenterDistance": 0, 
        "Speed": 0, 
        "Xspeed": 0, 
        "Yspeed": 62.5, 
        "Remark": ""
    }
}
```
| Json 参数 | 备注 | 获得数据编号 | 数据内容 |
| ---- | ---- | ---- | ---- |
| FlightNumer | 航班号 | [1021/170](#1021170) | CSN3127 |
| Reg | 机号 | [1021/080](#1021080) | 780261 |
| Lat | 纬度 | [1021/130](#1021130) | 30.1035156 |
| Lon | 经度 | [1021/130](#1021130) | 114.145256 |
| Stand | 机位 | | |
| PlaneType | 机型 | | |
| Runway | 跑道 | | |
| Direction | 方向 | | |
| Height | 飞机高度 | [1021/140](#1021140) | 0 |
| StandSide | 机位所在的位置 | | |
| UpdateTime | 信息更新的时间 | [1021/030](#1021030) | 2019/8/16 4:59:58.796 |
| CenterDistance | 距离 | | |
| Speed | 速度 | | |
| Xspeed | X方向速度 | | |
| Yspeed | Y方向速度 | [1021/157](#1021157) | 62.5 |
| Remark | 备注 | | |
### 代码
<span id="1021030"><span/>

#### 1021/030
``` csharp
/// <summary>
/// 计算日时间项(1021/030)对应的值
/// </summary>
/// <param name="lhs"></param>
/// <returns></returns>
public static string TimeOfDay(byte[] lhs)
{
    //将几个独立字节合并为一个字节
    uint rhs = ((uint)lhs[0] << 16) + ((uint)lhs[1] << 8) + (uint)lhs[2];
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
<span id="1021130"><span/>

#### 1021/130
``` csharp
/// <summary>
/// 计算位置坐标（WGS-84中）项（1021/130)对应的值
/// </summary>
/// <param name="lhs"></param>
/// <returns></returns>
public static double[] PositionWGS_84(byte[] lhs)
{
    double[] res = { 0, 0 };
    int value1;
    //将容器中前4个字节合并为一个字节，用以计算纬度。
    value1 = ((int)lhs[0] << 24) + ((int)lhs[1] << 16) + ((int)lhs[2] << 8) + (int)lhs[3];
    double temp1 = value1 * (5.364418e-6);
    //Console.WriteLine($"坐标值:纬度值{temp1}");
    res[1] = temp1;
    int value0;
    //将容器中后4个字节合并为一个字节，用以计算经度。
    value0 = ((int)lhs[4] << 24) + ((int)lhs[5] << 16) + ((int)lhs[6] << 8) + (int)lhs[7];
    double temp0 = value0 * (5.364418e-6);
    //Console.WriteLine($"经度值{temp0}");
    res[0] = temp0;
    return res;
}
```
<span id="1021080"><span/>

#### 1021/080
``` csharp
/// <summary>
/// 计算目标地址项（1021/080）对应的值
/// </summary>
/// <param name="lhs"></param>
/// <returns></returns>
public static string TargetAddress(byte[] lhs)
{
    uint rhs;
    rhs = ((uint)lhs[0] << 16) + ((uint)lhs[1] << 8) + (uint)lhs[2];
    string res = String.Format("{0:X}", rhs);

    return res;
}
```
<span id="1021157"><span/>

#### 1021/157
``` csharp
/// <summary>
/// 计算几何垂直速率项（1021/157)对应的值
/// </summary>
/// <param name="lhs"></param>
/// <returns></returns>
public static double GeometricVerticalRate(byte[] lhs)
{
    uint rhs;
    rhs = ((uint)lhs[0]) + (uint)lhs[1];
    double value0 = rhs * 6.25;
    //Console.WriteLine($"VR:{ value0 }");
    return value0;
}
```
<span id="1021170"><span/>

#### 1021/170
``` csharp
/// <summary>
/// 计算目标识别项（1021/170)对应的值
/// </summary>
/// <param name="lhs"></param>
/// <returns></returns>
public static string TargetIdentification(byte[] lhs)
{
    // Console.WriteLine("ACID:");
    string res = "";
    long rhs;
    rhs = ((long)lhs[0] << 40) + ((long)lhs[1] << 32) + ((long)lhs[2] << 24) + ((long)lhs[3] << 16) + ((long)lhs[4] << 8) + (long)lhs[5];//将6个独立字节合并为一个字节
    long value0 = (rhs >> 42) & 63;//取出第42~47位
    long value01 = (value0 >> 5) & 1;//取出新的二进制数的第5位，并判断为0还是1.
    if (value01 == 1)
    {
        char value02 = (char)value0;
        res += value02;
    }
    else
    {
        //value0 = (value0^(1 << 6));
        value0 ^= (1 << 6);//如果第5位为1，则将第6位取反。
        char value03 = (char)value0;
        res += value03;
    }
    long value1 = (rhs >> 36) & 63;//取出第36~41位
    long value11 = (value1 >> 5) & 1;
    if (value11 == 1)
    {
        char value12 = (char)value1;
        res += value12;
    }
    else
    {
        value1 ^= (1 << 6);
        char value13 = (char)value1;
        res += value13;
    }
    long value2 = (rhs >> 30) & 63;//取出第30~35位
    long value21 = (value2 >> 5) & 1;
    if (value21 == 1)
    {
        char value22 = (char)value2;
        res += value22;
    }
    else
    {
        value2 ^= (1 << 6);
        char value23 = (char)value2;
        res += value23;
    }
    long value3 = (rhs >> 24) & 63;//取出第24~29位
    long value31 = (value3 >> 5) & 1;
    if (value31 == 1)
    {
        char value32 = (char)value3;
        res += value32;
    }
    else
    {
        value3 ^= (1 << 6);
        char value33 = (char)value3;
        res += value33;
    }
    long value4 = (rhs >> 18) & 63;//取出第18~23位
    long value41 = (value4 >> 5) & 1;
    if (value41 == 1)
    {
        char value42 = (char)value4;
        res += value42; ;
    }
    else
    {
        value4 ^= (1 << 6);
        char value43 = (char)value4;
        res += value43;
    }
    long value5 = (rhs >> 12) & 63;//取出第12~17位
    long value51 = (value5 >> 5) & 1;
    if (value51 == 1)
    {
        char value52 = (char)value5;
        res += value52;
    }
    else
    {
        value5 ^= (1 << 6);
        char value53 = (char)value5;
        res += value53;
    }
    long value6 = (rhs >> 6) & 63;//取出第6~11位
    long value61 = (value6 >> 5) & 1;
    if (value61 == 1)
    {
        char value62 = (char)value6;
        res += value62;
    }
    else
    {
        value6 ^= (1 << 6);
        char value63 = (char)value6;
        res += value63;
    }
    long value7 = rhs & 63;//取出第0~5位
    long value71 = (value7 >> 5) & 1;
    if (value71 == 1)
    {
        char value72 = (char)value7;
        res += value72;
    }
    else
    {
        value7 ^= (1 << 6);
        char value73 = (char)value7;
        res += value73;
    }
    return res;
}
```
<span id="1021140"><span/>

#### 1021/140
``` csharp
/// <summary>
/// 计算几何高度项（1021/140)对应的值
/// </summary>
/// <param name="lhs"></param>
/// <returns></returns>
public static double GeometricAltitude(byte[] lhs)
{
    uint rhs;
    rhs = ((uint)lhs[0] << 8) + (uint)lhs[1];
    double value0 = rhs * 6.25;
    //以小数点后两位的形式输出
    //Console.WriteLine($"ALTITUD: {value0} " );
    return value0;
}
```