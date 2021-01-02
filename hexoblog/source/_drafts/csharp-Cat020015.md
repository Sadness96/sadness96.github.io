---
title: Cat020 报文解析
date: 2019-09-09 09:45:00
tags: [c#]
categories: C#.Net
---
# MLAT CAT020 0.15 报文协议解析
<!-- more -->
### 简介/声明
多点定位(MLAT) 全称是 Multilateration，多点定位技术利用多个地面接收机接收到机载应答机信号的时间差，计算得出飞机位置。

解析文档均为[欧洲航空交通管理](https://www.eurocontrol.int/)官方提供，解析代码基于 CSDN 付费文档修改，仅包含极少部分公司业务格式不包含涉密文件。
## 参考资料
### 原文
[EuroControl](https://www.eurocontrol.int)：[cat020p14ed15.pdf](https://www.eurocontrol.int/sites/default/files/2019-06/cat020-asterix-mlt-messages-part-14.pdf)
## 测试数据解析
14 00 46 FF 0F 01 84 16 07 41 10 A1 A0 BB 00 57 8B 48 01 44 DC F6 00 17 06 00 1F AD 0E F2 02 78 10 45 80 0C 54 F2 DB 3C 60 00 02 20 40 19 98 D0 00 00 00 00 00 01 00 0C 00 0C 00 03 00 06 00 05 00 05 A1 A0 C2 00
### 解析步骤
#### 解析为 Byte 数组
``` csharp
/// <summary>
/// 16进制文本转为 byte 数组
/// </summary>
public static byte[] GetBytesFromMultilateration(string strMultilateration)
{
    string[] strs = strMultilateration.Split(' ');
    byte[] temp = new byte[strs.Length];
    for (int i = 0; i < strs.Length; i++)
    {
        temp[i] = Convert.ToByte(strs[i], 16);
    }
    return temp;
}
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
#### 项目需要最终生成 Json 数据
``` json
{
	"FlightTag": {
		"FlightNumer": "CES2631",
		"Reg": "",
		"Lat": 30.77721,
		"Lon": 114.209694,
		"Stand": "",
		"PlaneType": "",
		"Runway": "",
		"Direction": 0.0,
		"Height": 0.0,
		"StandSide": "",
		"UpdateTime": "2019/9/23 6:59:13.460",
		"CenterDistance": 0.0,
		"Speed": 0.0,
		"Xspeed": 0.0,
		"Yspeed": 0.0,
		"Remark": ""
	}
}
```
| Json 参数 | 备注 | 获得数据编号 | 数据内容 |
| ---- | ---- | ---- | ---- |
| FlightNumer | 航班号 | [I020/245](#1020245) | CES2631 |
| Reg | 机号 |  |  |
| Lat | 纬度 | [I020/041](#1020041) | 30.77721 |
| Lon | 经度 | [I020/041](#1020041) | 114.209694 |
| Stand | 机位 | | |
| PlaneType | 机型 | | |
| Runway | 跑道 | | |
| Direction | 方向 | | |
| Height | 飞机高度 | | |
| StandSide | 机位所在的位置 | | |
| UpdateTime | 信息更新的时间 | [I020/140](#1020140) | 2019/9/23 6:59:13.460 |
| CenterDistance | 距离 | | |
| Speed | 速度 | | |
| Xspeed | X方向速度 | | |
| Yspeed | Y方向速度 | | |
| Remark | 备注 | | |
### 代码
#### 拆分数据（部分）
``` csharp
/// <summary>
/// 格式 Cat020 数据
/// </summary>
/// <param name="bytesData"></param>
/// <returns></returns>
public static Cat020 Format(byte[] bytesData)
{
    List<byte> cat020List = new List<byte>();
    cat020List.Add(bytesData[3]);
    cat020List.Add(bytesData[4]);
    cat020List.Add(bytesData[5]);
    cat020List.Add(bytesData[6]);

    //11111111000011110000000110000100
    string isData = "";
    foreach (var item in cat020List)
    {
        isData += Convert.ToString(item, 2).PadLeft(8, '0');
    }

    int index = 7;
    char[] isDataStr = isData.ToCharArray();
    bool flag = false;
    Dictionary<string, string> strData = new Dictionary<string, string>();

    for (int i = 0; i < Constants.cat020ItemNameList.Count; i++)
    {
        string data1 = Constants.cat020ItemNameList[i];
        string data2 = Constants.cat020ItemLengthList[i];
        if (data1.Equals("FX") && data2.Equals("-") && isDataStr[i] == '0')
        {
            flag = true;
        }
        if (data2.Equals("1+") && isDataStr[i] == '1')
        {
            string newStr = "";
            while (true)
            {
                if (index >= bytesData.Length)
                {
                    //logger.info("下标长度大于或者等于数据总长度,循环结束");
                    break;
                }
                string isextend = Convert.ToString(bytesData[index], 2);
                if (!isextend.EndsWith("1"))
                {
                    newStr += bytesData[index] + " ";
                    index++;
                    break;
                }
                newStr += bytesData[index] + " ";
                index++;
            }
            strData[(i + 1) + ""] = newStr.Trim();
        }
        else if (data2.Equals("-") || data2.Equals("") || data2.Equals("1+N*8"))
        { }
        else if (isDataStr[i] == '1')
        {
            int b = int.Parse(data2);
            string c = "";
            for (int k = 0; k < b; k++)
            {
                c += bytesData[index] + " ";
                index++;
            }
            strData[(i + 1) + ""] = c;
        }

        if (flag)
            break;
    }

    Cat020 cat020 = new Cat020();
    ---
    return cat020;
}
```

<span id="1020140"><span/>

#### I020/140
``` csharp
/// <summary>
/// 解析I020_140日时间
/// </summary>
/// <param name="timeStr"></param>
/// <returns></returns>
public static string I020_140(string timeStr)
{
    string[] strs = timeStr.Trim().Split(' ');
    byte[] byts16 = new byte[strs.Length];
    for (int i = 0; i < strs.Length; i++)
    {
        byts16[i] = Convert.ToByte(strs[i], 10);
    }
    // 16进制转成10进制
    string timeDec = (((uint)byts16[0] << 16) + ((uint)byts16[1] << 8) + (uint)byts16[2]).ToString();

    // 字符串转数值/128 * 1000 总毫秒数
    long ms = (long)((Double.Parse(timeDec) / 128) * 1000);

    int ss = 1000;
    int mi = ss * 60;
    int hh = mi * 60;

    long hour = ms / hh;
    long minute = (ms - hour * hh) / mi;
    long second = (ms - hour * hh - minute * mi) / ss;
    long milliSecond = ms - hour * hh - minute * mi - second * ss;

    string strHour = hour < 10 ? "0" + hour : "" + hour;// 小时
    string strMinute = minute < 10 ? "0" + minute : "" + minute;// 分钟
    string strSecond = second < 10 ? "0" + second : "" + second;// 秒
    string strMilliSecond = milliSecond < 10 ? "0" + milliSecond : "" + milliSecond;// 毫秒
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
/// <param name="flnoStr"></param>
/// <returns></returns>
public static string I020_245(string flnoStr)
{
    if (null != flnoStr)
    {
        try
        {
            string[] strflno = flnoStr.Trim().Split(' ');

            string str = "";
            for (int i = 1; i < strflno.Length; i++)
            {
                // 把第一位去掉
                string newFLNO = strflno[i];
                str += Convert.ToString(byte.Parse(newFLNO), 2).PadLeft(8, '0');
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
        catch (Exception e)
        { }
    }
    return "";
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
/// <param name="str"></param>
/// <returns></returns>
public static string[] I020_041(string str)
{
    string[] relDataArray = new string[2];
    if (!string.IsNullOrEmpty(str))
    {
        string[] dataArray = str.Trim().Split(' ');
        int[] tempDataArray = new int[dataArray.Length];
        for (int i = 0; i < dataArray.Length; i++)
        {
            tempDataArray[i] = int.Parse(dataArray[i]);
        }
        for (int i = 0; i < tempDataArray.Length; i++)
        {
            dataArray[i] = tempDataArray[i].ToString("X2");
        }

        if (dataArray.Length == 8)
        {
            string Item1 = dataArray[0];
            string Item2 = dataArray[1];
            string Item3 = dataArray[2];
            string Item4 = dataArray[3];
            string Item5 = dataArray[4];
            string Item6 = dataArray[5];
            string Item7 = dataArray[6];
            string Item8 = dataArray[7];
            // 16进制转成10进制（4位一转）
            string xCoordinate10 = Item1 + Item2 + Item3 + Item4;
            string yCoordinate10 = Item5 + Item6 + Item7 + Item8;
            // 10进制计算规则（xCoordinate10 * 180 /2^25）
            double xCoordinate = double.Parse(Convert.ToInt32(xCoordinate10, 16).ToString()) * 180 / 33554432;
            double yCoordinate = double.Parse(Convert.ToInt32(yCoordinate10, 16).ToString()) * 180 / 33554432;
            relDataArray[0] = xCoordinate.ToString();
            relDataArray[1] = yCoordinate.ToString();

            return relDataArray;
        }
        else
        { }
    }
    return null;
}
```
<span id="1020042"><span/>

#### I020/042
``` csharp
/// <summary>
/// 解析I020_042
/// </summary>
/// <param name="str"></param>
/// <returns></returns>
public static string[] I020_042(string str)
{
    string[] relDataArray = new string[2];
    if (!string.IsNullOrEmpty(str))
    {
        string[] dataArray = str.Trim().Split(' ');
        if (dataArray.Length == 6)
        {
            int[] tempDataArray = new int[dataArray.Length];
            for (int i = 0; i < dataArray.Length; i++)
            {
                tempDataArray[i] = int.Parse(dataArray[i]);
            }
            for (int i = 0; i < tempDataArray.Length; i++)
            {
                dataArray[i] = tempDataArray[i].ToString("X2");
            }
            // 16进制转成10进制
            string Item1 = dataArray[0];
            string Item2 = dataArray[1];
            string Item3 = dataArray[2];
            string Item4 = dataArray[3];
            string Item5 = dataArray[4];
            string Item6 = dataArray[5];
            string xAngle16 = Item1 + Item2 + Item3;
            string yAngle16 = Item4 + Item5 + Item6;
            string xAngle10 = Convert.ToInt32(xAngle16, 16).ToString();
            string yAngle10 = Convert.ToInt32(yAngle16, 16).ToString();
            // 10进制计算规则（xAngle10 * 0.5）
            double xAngle = Double.Parse(xAngle10) * 0.5;
            double yAngle = Double.Parse(yAngle10) * 0.5;
            relDataArray[0] = xAngle.ToString();
            relDataArray[1] = yAngle.ToString();

            return relDataArray;
        }
        else
        { }
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
/// <param name="str"></param>
/// <returns></returns>
public static string I020_161(string str)
{
    try
    {
        string[] dataArray = str.Trim().Split(' ');
        int[] tempDataArray = new int[dataArray.Length];
        for (int i = 0; i < dataArray.Length; i++)
        {
            tempDataArray[i] = int.Parse(dataArray[i]);
        }
        for (int i = 0; i < tempDataArray.Length; i++)
        {
            dataArray[i] = tempDataArray[i].ToString("X2");
        }
        str = dataArray[0] + dataArray[1];
        return Convert.ToInt32(str.Replace(" ", ""), 16).ToString();
    }
    catch (Exception e)
    { }
    return str;
}
```
<span id="1020110"><span/>

#### I020/110
``` csharp
/// <summary>
/// 解析I020_110
/// </summary>
/// <param name="str"></param>
/// <returns></returns>
public static string I020_110(string str)
{
    try
    {

        Double measuredFlightLevel = (double)0;

        if (!string.IsNullOrEmpty(str))
        {
            str = str.Replace(" ", "");
            // 将16进制转换为2进制
            string binStr = str;
            // 转为10进制
            string decStr = binStr;
            // 将十进制的字符串转换为double
            Double desDouble = Double.Parse(decStr);

            if (binStr.Length == 16)
            {
                // 如果2进制长度为16，说明第16位一定为1，则为负数
                measuredFlightLevel = -(Math.Pow(2, 16) - desDouble) * 6.25;

            }
            else
            {
                // 如果2进制长度不为16，说明第16位一定为0，则为正数
                measuredFlightLevel = desDouble * 6.25;
            }
            return measuredFlightLevel.ToString();
        }
        else
        { }
    }
    catch (Exception e)
    {
        return "";
    }
    return "";
}
```