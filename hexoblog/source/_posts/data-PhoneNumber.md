---
title: 数据分析-手机号
date: 2017-08-01 11:00:00
tags: [data,phonenumber]
categories: Data
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/data-PhoneNumber/PhoneLogo.png"/>

<!-- more -->
### 简介

<style>
table {
	width: 300px;
}
</style>

| +86 | XXX - XXXX | XXXX |
| --- | --- | --- |
| 中国 | 运营商及归属地 | 随机号码 |

### 正则表达式校验
``` csharp
/// <summary>
/// 效验中国大陆手机号码
/// </summary>
/// <param name="strPhoneNumber">中国大陆手机号码</param>
/// <returns>效验通过返回true,失败返回false</returns>
public static bool CheckPhoneNumber(string strPhoneNumber)
{
    try
    {
        //+86替换成空(只考虑中国大陆手机号)
        if (strPhoneNumber.Length == 14)
        {
            strPhoneNumber.Replace("+86", string.Empty);
        }
        //中国电信正则表达式匹配
        string strRegexChinaTelecom = @"^1[3578][01379]\d{8}$";
        Regex regexChinaTelecom = new Regex(strRegexChinaTelecom);
        //中国移动正则表达式匹配
        string strRegexChinaMobile = @"^(134[012345678]\d{7}|1[34578][012356789]\d{8})$";
        Regex regexChinaMobile = new Regex(strRegexChinaMobile);
        //中国联通正则表达式匹配
        string strRegexChinaUnicom = @"^1[34578][01256]\d{8}$";
        Regex regexChinaUnicom = new Regex(strRegexChinaUnicom);
        //验证手机号
        if (regexChinaTelecom.IsMatch(strPhoneNumber) || regexChinaMobile.IsMatch(strPhoneNumber) || regexChinaUnicom.IsMatch(strPhoneNumber))
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
```

### 归属地查询
下载：[mobile.xls](https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/file/data-PhoneNumber/mobile.xls)
<iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/file/data-PhoneNumber/mobile.xlsb" style="width:100%; height:1500px;" frameborder="0"></iframe>