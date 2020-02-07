---
title: 计算机硬件维修检测
date: 2020-02-07 01:44:45
tags: [repair,computer,hardware]
categories: Repair
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/repair-ComputerHardware/mainboard.jpg"/>

<!-- more -->
### 简介
在上大学之前，曾在百脑汇电子城工作过大半年，有维修过电脑、打印机、监控摄像头，时隔多年，这依然使我受益匪浅，在此记录一些硬件故障排除的经验，以及一些检测软件的使用。以组装台式机为主，部分介绍笔记本。

### 电脑构成以及可能存在的问题
| 计算机硬件 | 是否必须 | 可能存在的问题 |
| --- | --- | --- |
| 机箱 | 可选 | [内部静电导致无法开机](#机箱内部静电导致无法开机)、[机箱前面板接主板容易接错](#机箱前面板接主板容易接错) |
| 电源 | 是 | [风扇转速不够导致声音大或过热短路](#风扇转速不够导致声音大或过热短路)、[电源不工作](#电源不工作) |
| 主板 | 是 | [主板南桥北桥芯片过热](#主板南桥北桥芯片过热)、[电容鼓包等导致运行中断电或无法开机](#电容鼓包等导致运行中断电或无法开机)、[按开机键无反应](#按开机键无反应)、[无法从U盘启动](#无法从优盘启动) |
| CPU | 是 | [CPU 完全不工作](#CPU完全不工作)(可能性极低并且不能修复) |
| 内存 | 是| 闪存颗粒损坏、接触不良导致无法开机 |
| 硬盘 | 是 | (机械)硬盘坏道导致数据丢失或电脑死机、(机械)硬盘主控板不供电、(固态)硬盘写入次数达到上限、(固态)硬盘内存颗粒损坏、系统引导错误无法进入系统、SATA 线损坏导致无法读取磁盘、FAT32 磁盘格式单个文件最大只能支持4GB |
| 显卡 | 可选(板载) | 电容鼓包、接触不良等导致无法开机 |
| 散热器 | 是 | [缺少硅脂或风扇转速不够导致声音大或过热短路](#风扇转速不够导致声音大或过热短路) |
| 光驱 | 可选 | 标记为 RW 的光驱才可以刻录光盘 |
| 声卡 | 可选(板载) | 由于接口错误或系统设置导致的麦克风或音响失效 |
| 网卡 | 可选(板载) | 网线损坏、部分系统安装不包含网卡驱动、配置问题导致的网络无法连接 |
| 显示器 | 可选(远程) | 线材损坏(VGA/DVI/HDMI/DP)无法显示、高压板或驱动板损坏、亮点或屏线(修复概率较低) |
| 键盘 | 是 | 维修建议使用 PS/2 接口、(机械)键盘轴脱焊 |
| 鼠标 | 可选(极客) | 鼠标连键(单击变双击) |
| 音响 | 可选(耳麦) | 外置 USB 供电的音响出现杂音 |
| 打印机 | 可选 | 驱动问题导致无法使用、(针式)打印机色带打卷或断裂、(喷墨式)打印机喷头堵塞或需更换墨盒、(激光)打印机卡纸重影 |
| 游戏手柄 | 可选 | 驱动问题导致无法使用、连接数据线虚连 |

### 运行中故障(软件或硬件导致)
| 问题状况 | 可能存在的问题 |
| --- | --- |
| 移动电脑后无法开机 | 搬运导致的 PCI 或内存接口松动，参考：[按开机键无反应](#按开机键无反应) |
| 电脑运行一段时间后突然关机 | 温度过高、电压不稳 |
| 电脑运行一段时间后突然蓝屏 | 磁盘出现坏道、系统软件或应用软件导致驱动异常 |
| 鼠标键盘操作明显延迟 | CPU/内存/磁盘占用过高导致 |
| 屏幕花屏或色彩异常 | 线材松动或损坏(VGA/DVI/HDMI/DP)、显卡接触不良 |

### 系统软件或应用软件相关推荐
| 软件功能 | 推荐软件 | 简介 |
| --- | --- | --- |
| U盘启动(WinPE) | [老毛桃](https://www.laomaotao.net/)、[大白菜](http://www.winbaicai.com/) | 通过U盘启动提供的 WinPE 包含很多有用的工具，比如：[DiskGenius 分区工具](http://www.diskgenius.cn/)、[Ghost](https://baike.baidu.com/item/ghost/847?fr=aladdin) |
| 操作系统 | Windows、Linux | 现大部分装电脑都会选择 Windows(原版建议通过[itellyou](https://msdn.itellyou.cn/)下载，Ghost 版建议使用[Deepin](http://www.deepinghost.com/)或[雨林木风](http://www.ylmf888.com/))、部分企业或个人学习选择 Linux(建议使用[Ubuntu](https://ubuntu.com/download)、[CentOS](https://www.centos.org/)、[Red Hat](https://www.redhat.com/en)、[Debian](https://www.debian.org/))、MAC(黑苹果请移步到社区论坛吧) |
| 系统还原 | [冰点还原](https://www.bingdianhuanyuan.cn/) | 重启电脑还原至初始状态，避免病毒入侵，广泛用于银行学校宾馆，除了硬件 PCI 接口的还原卡之外，用软件还原也是不错的选择 |
| 磁盘分区/坏道检测/数据恢复 | [DiskGenius](http://www.diskgenius.cn/)、[MHDD](https://baike.baidu.com/item/MHDD/2755765?fr=aladdin)、[EasyRecovery](https://www.easyrecoverychina.com/) | 1.DiskGenius 是一个常用的分区、坏道检测以及数据恢复的工具。<br />2.MHDD 是一个古老的坏道检测工具，适用于老式 AMD 型机器坏道检测，但是其工具强大可以抹除坏道，现 U 盘启动 DOS 工具箱包含<br />3. EasyRecovery 更专业的数据恢复厂商|
| 光盘刻录 | [UltraISO](https://cn.ultraiso.net/)、[Nero](https://www.nero.com/chs/) | 个人觉着重要的数据刻录在光盘里还是蛮棒的 |
| 副本分屏 | [OnTopReplica](https://github.com/LorenzCK/OnTopReplica) | 这超过了维修的范围，但还是推荐一下，一个打游戏时可以优雅的把小地图投到另一块屏幕上(放大至全屏) |
| 驱动安装 | [鲁大师](https://www.ludashi.com/)、[驱动精灵](http://www.drivergenius.com/)、[驱动人生](https://www.160.com/)、[万能驱动](https://www.itsk.com/) | 除了官网下载指定型号的驱动外，选择国内厂商制作的一键安装也不错。<br />1.鲁大师(娱乐大师、鲁大姐)是最常见的驱动安装软件，压力检测也可以很好的测试电脑问题以及…没什么太大用的跑分<br />2.驱动精灵、驱动人生 特色在于可以下载带网卡驱动版，有些时候系统装完没有安装网卡驱动，需要联网才能下载网卡驱动，没有网卡驱动又连不上网，陷入死循环<br />3.万能驱动是 IT 天空制作的整合版驱动包，文件较大，一般 Ghost 系统中集成的就是万能驱动，另外 IT 天空提供了完整的工具程序，直接 Ghost 出一个自己的系统也不错哦 |
| 测试软件 | [3DMark](https://www.3dmark.com/)、[FritzChessBenchmark](http://www.jens-hartmann.at/Fritzmarks/)、[FurMark](https://geeks3d.com/furmark/)、[Unigine Heaven](https://benchmark.unigine.com/heaven?lang=en)、[CrystalDiskMark](https://crystalmark.info/en/software/crystaldiskmark/) | 1.3DMark 测试显卡游戏性能的专业软件<br />2.FritzChessBenchmark 国际象棋算法测试 CPU 运算速度<br />3. FurMark 知名的 GPU 拷机压力测试软件<br />4. Unigine Heaven 同样是知名的性能稳定测试软件<br />5.知名的磁盘读写基准测试软件 |

### 维修
#### 维修前建议准备
1.各种型号螺丝刀
2.条件允许的情况下准备测试机(能够正常运行的低配零件)，用于维修时替换部分可用零件检测。
3.主板诊断卡
4.U盘或光盘(包含启动项和系统)
5.电烙铁、焊锡、焊油、匹配型号的电容等零件、电工万能表
6.网线钳、水晶头、网线、网线测线仪

#### 机箱内部静电导致无法开机
<span id="机箱内部静电导致无法开机"><span/>
有遇到过几次这样的情况：电脑无法开机，短接主板开关灯不亮风扇不转，主板诊断卡不显示数字，主板上不存在明显损坏(鼓电容等)，重新插拔内存以及 PCI 接口设备后无明显改善，短接电源却可以使电源正常工作，尝试将所有零部件取出放在桌子上可以正常运行，应该是机箱设计缺陷或太多灰尘导致主板短路无法运行，全部拆出来清理灰尘即可。

#### 机箱前面板接主板容易接错
<span id="机箱前面板接主板容易接错"><span/>
机箱的前面板都会有一些功能型接线，常见的有音频、USB/SD、电源开关/复位需要对应的接在主板的接口上(通常是下方)，每种主板的接口顺序和位置不一样，建议查看主板上标记的缩写，或根据型号查询使用手册。
举例我现在使用的主板是 ASUS PRIME Z390-A
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/repair-ComputerHardware/AsusPrimeZ390-A.png"/>

#### 电源不工作
<span id="电源不工作"><span/>
1.确认其他硬件正常后无法通过主板开关运行电源，可尝试通过短路电源 ATX 24 PIN 接线的 PS-ON 与 COM 尝试启动电源。
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/repair-ComputerHardware/Atx24Pin.png"/>
2.尝试更换 220V 接入电源线(电脑通用电源线，实在没有多余可以找找身边电饭锅)
3.如果是内部高压滤波电容损坏或变压器损坏，不建议维修，想动手尝试的可以玩一玩。

#### 风扇转速不够导致声音大或过热短路
<span id="风扇转速不够导致声音大或过热短路"><span/>
目前大部分设备还避免不了使用风冷散热，使用过久以后会导致积灰严重，噪音较大或没有良好散热后导致计算机突然保护断电。
1.拆出风扇清理灰尘、可适当涂抹润滑油。
2.直接更换相同型号风扇部件。
3.CPU 由于缺少硅脂也会出现过热保护断电情况。

#### 主板南桥北桥芯片过热
<span id="主板南桥北桥芯片过热"><span/>
1.如果运行时突然保护断电，可尝试触摸桥芯片是否过热，如果没有影响到基础硬件使用，可尝试在桥芯片上涂抹硅脂然后贴上一片铝制散热片。
2.如果无法开机或影响其他硬件工作且桥芯片过热，需要更换桥芯片才可以，通常桥芯片为 [BGA 封装](https://baike.baidu.com/item/BGA%E5%B0%81%E8%A3%85/5900329?fr=aladdin)，除了购买到型号匹配的芯片外(通常电脑店都是用废板上拆一个同型号的)，还需要准备热风枪或 [BGA 焊台](https://baike.baidu.com/item/BGA%E7%84%8A%E5%8F%B0/9588841?fr=aladdin)(芯片级维修存在很大风险，救活了赚不少，救不活赔个芯片钱还有时间)。

#### 电容鼓包等导致运行中断电或无法开机
<span id="电容鼓包等导致运行中断电或无法开机"><span/>
电容鼓包是主板上比较常见的问题，大部分情况更换电容可以修复好，但是也难免有出现隐藏的其他问题导致更换完部件效果，下图是一个比较常见的鼓电容情况，更换同型号电容即可。
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/repair-ComputerHardware/Capacitance.png"/>

#### 按开机键无反应
<span id="按开机键无反应"><span/>
有时难免遇到接通电源后按开机键电脑无响应，逐步排除
1.机箱前面板按钮或连接线出现问题，尝试使用短接主板上的开关启动，以主板 ASUS PRIME Z390-A 为例：短接 PWRSW
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/repair-ComputerHardware/AsusPrimeZ390-A.png"/>

2.如依旧无法点亮机器依次更换电源，主板等主要零件测试。

如果按开机键后点亮机器，但是短时间内未进系统再次关机，逐步排除
1.如果有主板诊断卡，可插入诊断卡开机检查故障诊断码检测对应的硬件设备。通常来说两位诊断卡会提示大概错误如下，详细错误请查询购买诊断卡说明书。

| 故障代码 | 故障位置 |
| --- | --- |
| FF、00、C0、D0、CF、F1 | CPU 未通过 |
| C1、C6、C3、D3、D4、D6、D8、B0、A7、E1 | 内存未通过 |
| 24、25、26、01、0A、0B、2A、2B、31 | 显卡未通过 |

2.如手头没有诊断卡，尝试重新插拔 PCI 接口上的设备和内存，减少至最简零件数启动，通过替换配件检测故障。

#### 无法从U盘启动
<span id="无法从优盘启动"><span/>
对于维修人员来讲，排除所有硬件问题后，重装系统就可以做最后的收尾了，但是对于市面上各式各样的 U 盘来说，主板经常存在无法识别的情况，如果 U 盘作为系统盘的话，购买时优先考虑与主板类似时间出厂的产品，匹配 USB 标准(v1.1/v2.0/v3.0)，WinPE 推荐使用与主板出厂时间相近的镜像([老毛桃](https://www.laomaotao.net/)、[大白菜](http://www.winbaicai.com/))，选择与主板设置匹配的启动方式(BIOS/UEFI)，如果依旧无法启动，建议把系统镜像以 ISO 光盘的方式写入到 U 盘中。
以下记录各厂商进入 BIOS 的快捷键：

| 组装机主板 | | 品牌笔记本 | | 品牌台式机 | |
| --- | --- | --- | --- | --- | --- |
| 主板品牌 | 启动按键 | 笔记本品牌 | 启动按键 | 台式机品牌 | 启动按键 |
| 华硕主板 | F8 | 联想笔记本 | F12 | 联想台式机 | F12 |
| 技嘉主板 | F12 | 弘基笔记本 | F12 | 惠普台式机 | F12 |
| 微星主板 | F11 | 华硕笔记本 | ESC | 宏基台式机 | F12 |
| 映泰主板 | F9 | 惠普笔记本 | F9 | 戴尔台式机 | ESC |
| 梅捷主板 | ESC或F12 | 联想Thinkpad | F12 | 神舟台式机 | F12 |
| 七彩虹主板 | ESC或F11 | 戴尔笔记本 | F12 | 华硕台式机 | F8 |
| 华擎主板 | F11 | 神舟笔记本 | F12 | 方正台式机 | F12 |
| 斯巴达主板 | ESC | 东芝笔记本 | F12 | 清华同方台式机 | F12 |
| 昂达主板 | F11 | 三星笔记本 | F12 | 海尔台式机 | F12 |
| 双敏主板 | ESC | IBM笔记本 | F12 | 明基台式机 | F8 |
| 翔升主板 | F10 | 富士通笔记本 | F12 |
| 精英主板 | ESC或F11 | 海尔笔记本 | F12 |
| 冠盟主板 | F11或F12 | 方正笔记本 | F12 |
| 富士康主板 | ESC或F12 | 清华同方笔记本 | F12 |
| 顶星主板 | F11或F12 | 微星笔记本 | F11 |
| 铭瑄主板 | ESC | 明基笔记本 | F9 |
| 盈通主板 | F8 | 技嘉笔记本 | F12 |
| 捷波主板 | ESC | Gateway笔记本 | F12 |
| Intel主板 | F12 | eMachines笔记本 | F12 |
| 杰微主板 | ESC或F8 | 索尼笔记本 | ESC |
| 致铭主板 | F12 |
| 磐英主板 | ESC |
| 磐正主板 | ESC |
| 冠铭主板 | F9 |

#### CPU 完全不工作
<span id="CPU完全不工作"><span/>
由于 CPU 的做工及其技术导致无法修复，排除 CPU 损坏，尝试更换相同针脚的 CPU 测试。
扔了怪可惜的，做成项链还是蛮不错的(上班路上偷拍)
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/repair-ComputerHardware/CPUNecklace.jpg"/>