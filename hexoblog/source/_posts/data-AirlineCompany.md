---
title: 数据分析-航空公司
date: 2019-08-20 09:50:00
tags: [data,aviation,python]
categories: Data
---
### 航空公司数据分析
<!-- more -->
### 简介
为便于组织运输生产，每个航班都按照一定的规律编有不同的号码以便于区别和管理，这种号码称为航班号。

### 参考资料
航班号编排方式参考 2004 年中国民用航空局发布政府公文：[关于印发《中国民航航班号分配和使用方案》的通知](http://www.caac.gov.cn/XXGK/XXGK/ZFGW/201601/t20160122_27543.html)

### 航班号组成
航空公司代码由民航局规定发布，包含 [IATA](https://www.iata.org/) 发布的[二字码](https://baike.baidu.com/item/%E4%BA%8C%E5%AD%97%E7%A0%81/8016030?fr=aladdin)和 [ICAO](https://www.icao.int/Pages/default.aspx) 发布的[三字码](https://baike.baidu.com/item/%E4%B8%89%E5%AD%97%E4%BB%A3%E7%A0%81/19936463?fr=aladdin)，航班号使用的是二字码加四或三位阿拉伯数字组成(不同的设备可能使用不同的标准)，还有各个航空公司向民航局自己登记的呼号，用于无线电通讯中使用的代号。

### 爬虫爬取携程网航空公司二字码LOGO
#### 爬虫地址
1. 携程旅行 32x32 864个有效图标
    http://pic.c-ctrip.com/AssetCatalog/airline/32/MU.png
1. 同程旅行 SVG 66个有效图标
	https://m.elongstatic.com/flight/configmng/prod/airline/logo/MU.svg
1. 飞友科技（字母小写） 50x50 313个有效图标
	https://static.variflight.com/assets/image/aircorp/mu.png

#### 爬虫代码
``` Python
# _*_coding:utf-8_*_
import requests
import re
import os

class GetImage(object):
    def __init__(self, url):
        self.url = url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36'
        }
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.path = self.dir_path+'/imgs'
        isExists = os.path.exists(self.dir_path+'/imgs')
        # 创建目录
        if not isExists:
            os.makedirs(self.path)

    def download(self, url):
        try:
            res = requests.get(url, headers=self.headers)
            return res
        except Exception as e:
            print(url+'下载失败,原因:'+e.args)

    def save(self, res_img, file_name):
        if res_img:
            with open(file_name, 'wb') as f:
                f.write(res_img.content)
            print(url+'下载成功')

    def run(self):
        # 下载图片
        res_img = self.download(self.url)
        name = self.url.strip().split('/').pop()
        file_name = self.path+'/'+name
        # 保存
        self.save(res_img, file_name)

if __name__ == '__main__':
    url_list = []
    Letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    for i in Letter:
        for j in Letter:
            url_list.append(
                "http://pic.c-ctrip.com/AssetCatalog/airline/32/" + i + j + ".png")

    for url in url_list:
        print(url)
        text = GetImage(url)
        text.run()
```

### 航空公司信息查询
下载：[航空公司信息.xlsx](https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/file/data-AirlineCompany/航空公司信息.xlsx)
<iframe src="https://view.officeapps.live.com/op/view.aspx?src=https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/file/data-AirlineCompany/航空公司信息.xlsx" style="width:100%; height:1500px;" frameborder="0"></iframe>