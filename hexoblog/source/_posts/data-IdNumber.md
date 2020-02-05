---
title: 数据分析-身份证号码
date: 2017-08-01 15:00:00
tags: [data,idnumber]
categories: Data
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/data-IdNumber/IDLogo.png"/>

<!-- more -->
### 简介

<style>
table {
	width: 490px;
}
</style>

| XX | XX | XX | XXXX | XX | XX | XXX | X |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 省 | 市 | 区(县) | 出生年 | 月 | 日 | 顺序码 | 校验码 |

*现身份证由 [GB 11643-1999](http://www.gb688.cn/bzgk/gb/newGbInfo?hcno=080D6FBF2BB468F9007657F26D60013E) 标准规定的 18 位数字或字母(仅结尾X)组成。
1.前六位表示为常驻户口所在地的行政区划代码，由 [GB/T 2260-2007](http://www.gb688.cn/bzgk/gb/newGbInfo?hcno=C9C488FD717AFDCD52157F41C3302C6D) 标准执行。
2.第七位至第十四位为出生年、月、日。
3.第十五位至第十七位为顺序码(其中包含派出所代码，第十七位也用来表示性别:奇数表示男性，偶数表示女性)
4.第18位数字是校检码：用来检验身份证的正确性。校检码可以是0~10的数字，10用X表示。

### 空白正反面
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/data-IdNumber/Clip1.bmp"/>

<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/data-IdNumber/Clip2.bmp"/>

### 正则表达式校验
``` CSharp
/// <summary>
/// 效验身份证号码
/// </summary>
/// <param name="strIDNumber">身份证号码</param>
/// <returns>效验通过返回true,失败返回false</returns>
public static bool CheckIDNumber(string strIDNumber)
{
    try
    {
        if (strIDNumber.Length == 15 && CheckIDCard15(strIDNumber))
        {
            return true;
        }
        else if (strIDNumber.Length == 18 && CheckIDCard18(strIDNumber))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    catch (Exception ex)
    {
        return false;
    }
}

/// <summary>
/// 15位身份证号验证
/// </summary>
/// <param name="idNumber">身份证号</param>
/// <returns>效验通过返回true,失败返回false</returns>
private static bool CheckIDCard15(string idNumber)
{
    long n = 0;
    if (long.TryParse(idNumber, out n) == false || n < Math.Pow(10, 14))
    {
        return false;//数字验证  
    }
    string address = "11x22x35x44x53x12x23x36x45x54x13x31x37x46x61x14x32x41x50x62x15x33x42x51x63x21x34x43x52x64x65x71x81x82x91";
    if (address.IndexOf(idNumber.Remove(2)) == -1)
    {
        return false;//省份验证  
    }
    string birth = idNumber.Substring(6, 6).Insert(4, "-").Insert(2, "-");
    DateTime time = new DateTime();
    if (DateTime.TryParse(birth, out time) == false)
    {
        return false;//生日验证  
    }
    return true;//符合15位身份证标准  
}

/// <summary>  
/// 18位身份证号码验证  
/// </summary>  
/// <param name="idNumber">身份证号</param>  
/// <returns>效验通过返回true,失败返回false</returns>  
private static bool CheckIDCard18(string idNumber)
{
    long n = 0;
    if (long.TryParse(idNumber.Remove(17), out n) == false || n < Math.Pow(10, 16) || long.TryParse(idNumber.Replace('x', '0').Replace('X', '0'), out n) == false)
    {
        return false;//数字验证    
    }
    string address = "11x22x35x44x53x12x23x36x45x54x13x31x37x46x61x14x32x41x50x62x15x33x42x51x63x21x34x43x52x64x65x71x81x82x91";
    if (address.IndexOf(idNumber.Remove(2)) == -1)
    {
        return false;//省份验证    
    }
    string birth = idNumber.Substring(6, 8).Insert(6, "-").Insert(4, "-");
    DateTime time = new DateTime();
    if (DateTime.TryParse(birth, out time) == false)
    {
        return false;//生日验证    
    }
    string[] arrVarifyCode = ("1,0,x,9,8,7,6,5,4,3,2").Split(',');
    string[] Wi = ("7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2").Split(',');
    char[] Ai = idNumber.Remove(17).ToCharArray();
    int sum = 0;
    for (int i = 0; i < 17; i++)
    {
        sum += int.Parse(Wi[i]) * int.Parse(Ai[i].ToString());
    }
    int y = -1;
    Math.DivRem(sum, 11, out y);
    Console.WriteLine("Y的理论值: " + y);
    if (arrVarifyCode[y] != idNumber.Substring(17, 1).ToLower())
    {
        return false;//校验码验证    
    }
    return true;//符合GB11643-1999标准    
}
```

### 行政区划查询
下载：[areacodebase.xlsb](https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/file/data-IdNumber/areacodebase.xlsb)
<iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/file/data-IdNumber/areacodebase.xlsb" style="width:100%; height:1500px;" frameborder="0"></iframe>