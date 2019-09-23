---
title: Kafka Demo
date: 2019-09-16 18:07:06
tags: [c#,kafka]
categories: C#.Net
---
### Kafka 消息队列使用介绍
<!-- more -->
#### 简介
[ApacheKafka](http://kafka.apache.org/) 是一个分布式流平台。

流平台具有三个关键功能：

* 发布和订阅记录流，类似于消息队列或企业消息传递系统。
* 以容错的持久方式存储记录流。
* 处理记录流。

Kafka通常用于两大类应用程序：

* 建立实时流数据管道，以可靠地在系统或应用程序之间获取数据
* 构建实时流应用程序以转换或响应数据流
#### 安装部署
请参阅[官方文档](http://kafka.apache.org/documentation/)
#### C#代码调用
引用 [Confluent.Kafka](https://github.com/confluentinc/confluent-kafka-dotnet) 库
##### 生产者(未测试)
``` CSharp
using System;
using Confluent.Kafka;

class Program
{
    public static void Main(string[] args)
    {
        var conf = new ProducerConfig { BootstrapServers = "localhost:9092" };

        Action<DeliveryReport<Null, string>> handler = r => 
            Console.WriteLine(!r.Error.IsError
                ? $"Delivered message to {r.TopicPartitionOffset}"
                : $"Delivery Error: {r.Error.Reason}");

        using (var p = new ProducerBuilder<Null, string>(conf).Build())
        {
            for (int i=0; i<100; ++i)
            {
                p.Produce("my-topic", new Message<Null, string> { Value = i.ToString() }, handler);
            }

            // wait for up to 10 seconds for any inflight messages to be delivered.
            p.Flush(TimeSpan.FromSeconds(10));
        }
    }
}
```
##### 消费者
``` CSharp
using System;
using System.Threading;
using Confluent.Kafka;

class Program
{
    public static void Main(string[] args)
    {
        var conf = new ConsumerConfig
        { 
            GroupId = "test-consumer-group",
            BootstrapServers = "localhost:9092",
            // Note: The AutoOffsetReset property determines the start offset in the event
            // there are not yet any committed offsets for the consumer group for the
            // topic/partitions of interest. By default, offsets are committed
            // automatically, so in this example, consumption will only start from the
            // earliest message in the topic 'my-topic' the first time you run the program.
            AutoOffsetReset = AutoOffsetReset.Earliest
        };

        using (var c = new ConsumerBuilder<Ignore, string>(conf).Build())
        {
            c.Subscribe("my-topic");

            CancellationTokenSource cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => {
                e.Cancel = true; // prevent the process from terminating.
                cts.Cancel();
            };

            try
            {
                while (true)
                {
                    try
                    {
                        var cr = c.Consume(cts.Token);
                        Console.WriteLine($"Consumed message '{cr.Value}' at: '{cr.TopicPartitionOffset}'.");
                    }
                    catch (ConsumeException e)
                    {
                        Console.WriteLine($"Error occured: {e.Error.Reason}");
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Ensure the consumer leaves the group cleanly and final offsets are committed.
                c.Close();
            }
        }
    }
}
```
#### 遇到问题
##### 在实际使用中消费数据使用多服务器链接
追加配置多 IP 地址到 Hosts 文件中（例）：
```
172.26.78.135 tdh02
172.26.78.136 tdh03
172.26.78.137 tdh04
172.26.78.140 tdh07
172.26.78.141 tdh08
172.26.78.142 tdh09
```
Host 文件所在目录：
| 系统 | 目录 |
| ---- | ---- |
| Windows | C:\windows\system32\drivers\etc\Hosts |
| Linux / Unix | /etc/Hosts |
| Mac OS | /private/etc/Hosts |
修改部分代码：
``` CSharp
var conf = new ConsumerConfig
{ 
    GroupId = "test-consumer-group",
    BootstrapServers = "tdh02:9092,tdh03:9092,tdh04:9092,tdh07:9092,tdh08:9092,tdh09:9092",
    AutoOffsetReset = AutoOffsetReset.Earliest
};
```