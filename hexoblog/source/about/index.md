---
title: 关于
date: 2019-03-05 14:00:00
---
<style>
.content-wrap {
    background: #fff;
    margin-bottom: 20px;
    padding: 20px;
}
</style>
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script type="text/javascript">
    $(document).ready(function () {
        var date = new Date;
        var vWorkYear = date.getFullYear() - 2016;
        $("#WorkYear").html(toChinesNum(vWorkYear));
    });

    let toChinesNum = (num) => {
        let changeNum = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']; 
        let unit = ["", "十", "百", "千", "万"];
        num = parseInt(num);
        let getWan = (temp) => {
        　　let strArr = temp.toString().split("").reverse();
        　　let newNum = "";
        　　for (var i = 0; i < strArr.length; i++) {
        　　newNum = (i == 0 && strArr[i] == 0 ? "" : (i > 0 && strArr[i] == 0 && strArr[i - 1] == 0 ? "" : changeNum[strArr[i]] + (strArr[i] == 0 ? unit[0] : unit[i]))) + newNum;
        　　}
        　 return newNum;
        }
        let overWan = Math.floor(num / 10000);
        let noWan = num % 10000;
        if (noWan.toString().length < 4) {
            noWan = "0" + noWan;
        }
        return overWan ? getWan(overWan) + "万" + getWan(noWan) : getWan(num);
    }
</script>

## 欢迎访问我的个人博客！我叫刘佳华
### 个人简介
<!-- <label id="WorkYear">五</label>年全栈开发工作经验，拥有面向林业局、公安、国际机场港口的工作经历，具有项目设计管理经验，也善于独立开发。 -->
水平不高，能力有限，但是依旧希望有改变世界的一天！

<!-- 
#### 学校
[齐齐哈尔职业教育中心学校](http://www.qzjzx.com/) 2011 ~ 2014
[哈尔滨信息工程学院](http://www.hxci.com.cn/) 2014 ~ 2017

#### 工作
齐齐哈尔百脑汇电脑城 2014-01-05 ~ 2014-08-31
[北京地林伟业科技股份有限公司](http://www.forestar.com.cn/) 2016-06-28 ~ 2018-06-28
[科盾科技股份有限公司北京分公司](http://www.kedun.com/) 2018-07-12 ~ 2019-08-09
[北京天睿空间科技股份有限公司](http://www.iseetech.com.cn) 2019-08-14 ~ 至今 

#### 联系方式
QQ：[623155166](http://wpa.qq.com/msgrd?v=3&uin=623155166&site=qq&menu=yes)
E-Mail：qsbs623@163.com
-->

### 结束语
希望以这个博客记录生活，记录学习，记录成长！
一首纯音乐，来自游戏《彩虹岛》中的《Wind from the Far East》，敬请欣赏！

<iframe frameborder="no" border="0" marginwidth="0" marginheight="0" width="330" height="86" src="//music.163.com/outchain/player?type=2&id=28445602&auto=1&height=66"></iframe>