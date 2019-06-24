---
title: EntityFramework Demo
date: 2018-07-02 20:53:18
tags: [c#,ef,mysql]
categories: C#.Net
---
### EntityFramewor框架使用介绍
<!-- more -->
#### 简介
[Entity Framework](https://baike.baidu.com/item/ADO.NET%20Entity%20Framework/6444727?fr=aladdin) 是微软以 ADO.NET 为基础所发展出来的对象关系对应 ([O/R Mapping](https://baike.baidu.com/item/%E5%AF%B9%E8%B1%A1%E5%85%B3%E7%B3%BB%E6%98%A0%E5%B0%84/311152?fromtitle=O%2FR%20Mapping&fromid=1229659)) 解决方案。
在.NET 3.5之前，开发者通常使用 [ADO.NET](/blog/2016/12/21/csharp-ADOHelper/) 直接连接操作数据库，而Entity Framework的出现可以让开发者更多的从代码层面考虑数据交互，Entity Framework 会把字段映射为实体模型，通过 [Lambda表达式](https://baike.baidu.com/item/Lambda%E8%A1%A8%E8%BE%BE%E5%BC%8F/4585794?fr=aladdin) 来操作数据，不需要考虑各种类型数据库和拼写SQL语句。同时也有效的防止了 [SQL注入](https://baike.baidu.com/item/sql%E6%B3%A8%E5%85%A5)。
#### 搭建-以MySQL为例
##### 类库或应用程序项目下新建项
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-EntityFramework/ef1.png"/>
##### 实体模型向导
从EF 4.X开始支持三种构建方法：
Database First：数据库优先，你的项目已经有一个数据库，并且使用工具（如Visual Studio中的EF设计器)依据数据库生成C#或VB类。你可以通过EF设计器修改这些创建的类以及类和数据库之间的映射关系；这种方式的关键是先要有数据，然后才有代码和模型。
Model First：模型优先，通过在EF设计器中创建一个空的容器，在其中设计实体数据模型。这个模型将用于生成数据库以及C#或者VB类。这种方式的关键是先要有模型，然后才有数据库和代码。
Code First：代码优先，开发者只需要写代码，将会自动创建模型和数据库。
现采用基础又最常用的 Database First 方式创建！

#### 增删改查
``` CSharp
//新增
officeautomationEntities ef = new officeautomationEntities();
user_info user = new user_info();
user.UserName = "00006";
user.Password = "123456";
ef.user_info.Add(user);
ef.SaveChanges();
//修改
var query = ef.user_info.Where(o => o.UserName.Equals("00006")).FirstOrDefault();
query.Password = "mq1i1JC92zal7nnbFZjtPQ==";
ef.SaveChanges();
//删除
ef.user_info.Remove(query);
ef.SaveChanges();
//查询
var v = ef.user_info.Where(o => o.UserName.Equals("00003")).ToList();
```
#### 错误及处理
##### EF创建时崩溃
MySql引用库版本修改为6.9.9
##### 报错：Host “”is not allowed to connect to this MySQL server
``` SQL
grant all privileges on *.* to 'root'@'192.168.0.1' identified by '密码';
grant all privileges on *.* to 'root'@'%' identified by '密码';
flush privileges;
```
##### 报错：Mysql表 "TableDetails" 中列 "IsPrimaryKey" 的值位 DBNull。
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/csharp-EntityFramework/error1.png"/>
``` SQL
use 库名;
SET GLOBAL optimizer_switch='derived_merge=off';
SET optimizer_switch='derived_merge=off';
```