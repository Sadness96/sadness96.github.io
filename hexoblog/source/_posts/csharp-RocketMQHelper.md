---
title: RocketMQ 帮助类
date: 2023-10-24 16:45:00
tags: [c#,rocketmq]
categories: C#.Net
---
### RocketMQ 消息队列帮助类使用介绍
<!-- more -->
### 简介
[RocketMQ Demo](https://sadness96.github.io/blog/2023/08/16/go-RocketMQ/) 写了关于 RocketMQ 的基础介绍与官方提供的最简生产者与消费者的 Demo，为了方便，当时使用的 Golang 对接的数据，现在给 c# 的版本封装一下。

### 核心代码
[RocketMQHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Queue.Helper/RocketMQ/RocketMQHelper.cs)

``` CSharp
/// <summary>
/// RocketMQ 消息队列帮助类
/// 创建日期:2023年10月24日
/// 调用程序也需要引用部分依赖
/// </summary>
public class RocketMQHelper
{
    /// <summary>
    /// 生产者
    /// </summary>
    private Producer producer;

    /// <summary>
    /// 消费者
    /// </summary>
    private Consumer consumer;

    /// <summary>
    /// 消息回调
    /// </summary>
    public event Action<string> MessageCallback;

    /// <summary>
    /// 注册生产者
    /// </summary>
    /// <param name="nameServer">服务地址</param>
    /// <param name="topic">主题 需要提前创建</param>
    public void RegisterProducer(string nameServer, string topic)
    {
        producer = new Producer()
        {
            NameServerAddress = nameServer,
            Topic = topic,
        };

        producer.Start();
    }

    /// <summary>
    /// 注册消费者
    /// </summary>
    /// <param name="nameServer">服务地址</param>
    /// <param name="topic">主题 需要提前创建</param>
    /// <param name="group">消费组</param>
    /// <param name="batchSize">拉取的批大小</param>
    /// <param name="isNotice">通知消息队是否消费了消息</param>
    public void RegisterConsumer(string nameServer, string topic, string group, int batchSize = 1, bool isNotice = true)
    {
        consumer = new Consumer
        {
            NameServerAddress = nameServer,
            Topic = topic,
            Group = group,
            BatchSize = batchSize,
            OnConsume = (q, ms) =>
            {
                //string mInfo = $"BrokerName={q.BrokerName},QueueId={q.QueueId},Length={ms.Length}";

                foreach (var item in ms.ToList())
                {
                    //string msg = string.Format($"接收到消息：msgId={item.MsgId},key={item.Keys}，产生时间【{item.BornTimestamp.ToDateTime()}】，内容：{item.BodyString}");

                    MessageCallback?.Invoke(item.BodyString);
                }
                return isNotice;
            }
        };

        consumer.Start();
    }

    /// <summary>
    /// 发送消息
    /// </summary>
    /// <param name="message">消息</param>
    public void SendMessage(string message)
    {
        try
        {
            var sr = producer.Publish(message);
        }
        catch (Exception)
        {
            //throw;
        }
    }

    /// <summary>
    /// 关闭连接
    /// </summary>
    public void Close()
    {
        producer?.Stop();
        consumer?.Stop();
    }
}
```

### 调用测试
#### 生产者
``` CSharp
static void Main(string[] args)
{
    // 服务地址
    string nameServer = "127.0.0.1:9876";
    // 主题
    string topic;

    RocketMQHelper rocketMQHelper = new RocketMQHelper();
    rocketMQHelper.RegisterProducer(nameServer, topic);

    while (true)
    {
        rocketMQHelper.SendMessage($"{DateTime.Now}");
        Task.Delay(TimeSpan.FromMilliseconds(500)).Wait();
    }
}
```

#### 消费者
``` CSharp
static void Main(string[] args)
{
    // 服务地址
    string nameServer = "127.0.0.1:9876";
    // 主题
    string topic;
    // 消费组
    string group;

    RocketMQHelper rocketMQHelper = new RocketMQHelper();
    rocketMQHelper.MessageCallback += RocketMQHelper_MessageCallback;
    rocketMQHelper.RegisterConsumer(nameServer, topic, group);
    Console.ReadLine();
}

/// <summary>
/// 队列消息回调
/// </summary>
/// <param name="message"></param>
private static void RocketMQHelper_MessageCallback(string obj)
{
    Console.WriteLine(obj);
}
```