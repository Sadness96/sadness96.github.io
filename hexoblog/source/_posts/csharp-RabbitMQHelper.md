---
title: RabbitMQ 帮助类
date: 2023-09-07 15:45:00
tags: [c#,rabbitmq]
categories: C#.Net
---
### RabbitMQ 消息队列帮助类使用介绍
<!-- more -->
### 简介
[RabbitMQ Demo](https://sadness96.github.io/blog/2018/07/07/csharp-RabbitMQ/) 写了关于 RabbitMQ 的基础介绍与官方提供的最简生产者与消费者的 Demo，平时都是直接使用公司同事写好的库，但是越用越觉得臃肿难以配置，所以接下来计划实现一系列队列相关的通讯帮助类。

### 核心代码
[RabbitMQHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Queue.Helper/RabbitMQ/RabbitMQHelper.cs)

``` CSharp
/// <summary>
/// RabbitMQ 消息队列帮助类
/// </summary>
public class RabbitMQHelper
{
    private IConnection _connection;
    private IModel _channel;
    private string _exchangeName;

    /// <summary>
    /// 消息回调
    /// </summary>
    public event Action<string> MessageCallback;

    /// <summary>
    /// 构造函数
    /// </summary>
    /// <param name="hostName">IP</param>
    /// <param name="port">端口号</param>
    /// <param name="userName">用户名</param>
    /// <param name="password">密码</param>
    public RabbitMQHelper(string hostName, int port = 5672, string userName = "", string password = "")
    {
        var factory = new ConnectionFactory()
        {
            HostName = hostName,
            Port = port,
            UserName = userName,
            Password = password,
            // 自动重连
            AutomaticRecoveryEnabled = true,
            // 恢复拓扑结构
            TopologyRecoveryEnabled = true,
            // 后台处理消息
            UseBackgroundThreadsForIO = true,
            // 心跳超时时间
            RequestedHeartbeat = 60
        };

        _connection = factory.CreateConnection();
        _channel = _connection.CreateModel();
    }

    /// <summary>
    /// 注册生产者
    /// </summary>
    /// <param name="exchangeName">交换机</param>
    /// <param name="queueName">队列</param>
    /// <param name="durable">持久化</param>
    /// <param name="ttl">生存时间</param>
    public void RegisterProducer(string exchangeName, string queueName, bool durable = true, TimeSpan? ttl = null)
    {
        _exchangeName = exchangeName;
        _channel.ExchangeDeclare(exchangeName, ExchangeType.Topic, durable);

        var arguments = new Dictionary<string, object>();
        if (ttl != null)
        {
            arguments.Add("x-message-ttl", (int)ttl.Value.TotalMilliseconds);
        }
        _channel.QueueDeclare(queueName, durable, false, false, arguments);
        _channel.QueueBind(queueName, exchangeName, routingKey: queueName);
    }

    /// <summary>
    /// 注册消费者
    /// </summary>
    /// <param name="queueName">队列名称</param>
    /// <param name="durable">持久化</param>
    /// <param name="ttl">生存时间</param>
    public void RegisterConsumer(string queueName, bool durable = true, TimeSpan? ttl = null)
    {
        var arguments = new Dictionary<string, object>();

        if (ttl != null)
        {
            arguments.Add("x-message-ttl", (int)ttl.Value.TotalMilliseconds);
        }

        _channel.QueueDeclare(queueName, durable, false, false, arguments);

        var consumer = new EventingBasicConsumer(_channel);
        consumer.Received += (model, ea) =>
        {
            var body = ea.Body.ToArray();
            var message = Encoding.UTF8.GetString(body);

            MessageCallback?.Invoke(message);
        };

        _channel.BasicConsume(queue: queueName, autoAck: true, consumer: consumer);
    }

    /// <summary>
    /// 发送消息
    /// </summary>
    /// <param name="routingKey">路由键</param>
    /// <param name="message">消息</param>
    /// <param name="persistent">消息持久化</param>
    public void SendMessage(string routingKey, string message, bool persistent = true)
    {
        var properties = _channel.CreateBasicProperties();
        properties.Persistent = persistent;

        var body = Encoding.UTF8.GetBytes(message);

        _channel.BasicPublish(exchange: _exchangeName, routingKey: routingKey, basicProperties: properties, body: body);
    }

    /// <summary>
    /// 关闭连接
    /// </summary>
    public void Close()
    {
        _channel.Close();
        _connection.Close();
    }
}
```

### 调用测试
#### 生产者
``` CSharp
// IP
string ip;
// 端口
int port;
// 用户名
string userName;
// 密码
string password;
// 交换机
string exchangeName;
// 队列名称
string queueName;
// 持久化
bool durable = true;
// 生存时间
TimeSpan ttl = TimeSpan.FromDays(1);

// 创建消费者 RabbitMQ 对象
RabbitMQHelper _rabbitMQHelper = new RabbitMQHelper(ip, port, userName, password);
_rabbitMQHelper.RegisterProducer(exchangeName, queueName, durable, ttl);

// 路由键
string routingKey;
// 消息
string message;

// 发送消息
_rabbitMQHelper.SendMessage(routingKey, message);
``` 

#### 消费者
``` CSharp
// IP
string ip;
// 端口
int port;
// 用户名
string userName;
// 密码
string password;
// 队列名称
string queueName;
// 持久化
bool durable = true;
// 生存时间
TimeSpan ttl = TimeSpan.FromDays(1);

// 创建消费者 RabbitMQ 对象
RabbitMQHelper _rabbitMQHelper = new RabbitMQHelper(ip, port, userName, password);
_rabbitMQHelper.MessageCallback += RabbitMQHelper_MessageCallback;
_rabbitMQHelper.RegisterConsumer(queueName, durable, ttl);

/// <summary>
/// 队列消息回调
/// </summary>
/// <param name="message"></param>
private void RabbitMQHelper_MessageCallback(string message)
{
    // 接收队列数据
}
```