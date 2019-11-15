---
title: RabbitMQ Demo
date: 2018-07-07 12:00:46
tags: [c#,rabbitmq]
categories: C#.Net
---
### RabbitMQ 消息队列使用介绍
<!-- more -->
#### 简介
[RabbitMQ](https://www.rabbitmq.com/) 是实现了高级消息队列协议（AMQP）的开源消息代理软件（亦称面向消息的中间件）。RabbitMQ服务器是用Erlang语言编写的，而群集和故障转移是构建在开放电信平台框架上的。所有主要的编程语言均有与代理接口通讯的客户端库。具备异步、解耦等机制。
RabbitMQ在全球范围内在小型初创公司和大型企业中进行了超过35,000次RabbitMQ生产部署，是最受欢迎的开源消息代理。
RabbitMQ轻量级，易于在内部和云中部署。它支持多种消息传递协议。RabbitMQ可以部署在分布式和联合配置中，以满足高规模，高可用性要求。
RabbitMQ可在许多操作系统和云环境中运行，并为大多数流行语言提供各种开发人员工具。
#### 安装部署
请参阅[官方文档](https://www.rabbitmq.com/download.html)
##### Docker 部署
``` cmd
安装官方镜像
docker pull rabbitmq
启动 RabbitMQ
docker run -d --name myrabbitmq -p 5672:5672 -p 15672:15672 docker.io/rabbitmq:3-management
设置 RabbitMQ 默认账户密码为 guest/guest
docker run -d --hostname my-rabbit --name some-rabbit -e RABBITMQ_DEFAULT_USER=user -e RABBITMQ_DEFAULT_PASS=password rabbitmq:3-management
WEB 端登录
http://localhost:15672/#/
```
#### C#代码调用
引用 [RabbitMQ.Client](https://www.rabbitmq.com/dotnet.html) 库
##### 生产者
``` CSharp
/// <summary>
/// 生产者
/// </summary>
public static void Send()
{
    var factory = new ConnectionFactory() { HostName = "localhost" };
    using (var connection = factory.CreateConnection())
    using (var channel = connection.CreateModel())
    {
        channel.QueueDeclare(queue: "hello", durable: false, exclusive: false, autoDelete: false, arguments: null);

        string message = "Hello World!";
        var body = Encoding.UTF8.GetBytes(message);

        channel.BasicPublish(exchange: "", routingKey: "hello", basicProperties: null, body: body);
        Console.WriteLine(" [x] Sent {0}", message);
    }

    Console.WriteLine(" Press [enter] to exit.");
    Console.ReadLine();
}
```
##### 消费者
``` CSharp
/// <summary>
/// 消费者
/// </summary>
public static void Receive()
{
    var factory = new ConnectionFactory() { HostName = "localhost" };
    using (var connection = factory.CreateConnection())
    using (var channel = connection.CreateModel())
    {
        channel.QueueDeclare(queue: "hello", durable: false, exclusive: false, autoDelete: false, arguments: null);

        var consumer = new EventingBasicConsumer(channel);
        consumer.Received += (model, ea) =>
        {
            var body = ea.Body;
            var message = Encoding.UTF8.GetString(body);
            Console.WriteLine(" [x] Received {0}", message);
        };
        channel.BasicConsume(queue: "hello", autoAck: true, consumer: consumer);

        Console.WriteLine(" Press [enter] to exit.");
        Console.ReadLine();
    }
}
```