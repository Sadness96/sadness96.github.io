---
title: 计算机硬件维修检测
date: 2020-02-07 01:44:45
tags: [repair,computer,hardware]
categories: Repair
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/repair-ComputerHardware/mainboard.jpg"/>

<!-- more -->
### 简介
在上大学之前，曾在百脑汇电子城工作过大半年，有维修过电脑、打印机、监控摄像头，时隔多年，这依然使我受益匪浅，在此记录一些硬件故障排除的经验，以及一些检测软件的使用。
### 电脑构成以及可能存在的问题
| 计算机硬件 | 是否必须 | 可能存在的问题 |
| --- | --- | --- |
| 机箱 | 可选 | 内部静电导致无法开机、机箱前面板接主板容易接错 |
| 电源 | 是 | 电源线损坏、风扇转速不够导致声音大或过热短路、完全不工作 |
| 主板 | 是 | 南桥北桥芯片过热、电容鼓包等导致运行中断电或无法开机、BIOS 不识别 U盘 |
| CPU | 是 | 完全不工作(可能性极低并且不能修复) |
| 内存 | 是| 闪存颗粒损坏、接触不良导致无法开机 |
| 硬盘 | 是 | (机械)硬盘坏道导致数据丢失或电脑死机、(机械)硬盘主控板不供电、(固态)硬盘写入次数达到上限、(固态)硬盘内存颗粒损坏、系统引导错误无法进入系统、SATA 线损坏导致无法读取磁盘、FAT32 磁盘格式单个文件最大只能支持4GB |
| 显卡 | 可选(板载) | 电容鼓包、接触不良等导致无法开机 |
| 散热器 | 是 | 缺少硅脂或风扇转速不够导致声音大或过热短路 |
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
| 移动电脑后无法开机 | 搬运导致的 PCI 或内存接口松动 |
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