---
title: MQTT 协议
date: 2022-09-15 16:46:10
tags: [c#,mqtt]
categories: C#.Net
---
### 使用 MQTT 协议收发数据
<!-- more -->
#### 简介
[MQTT 协议](https://www.runoob.com/w3cnote/mqtt-intro.html) 是一种基于发布/订阅（publish/subscribe）模式的"轻量级"通讯协议，该协议构建于 TCP/IP 协议上，由 IBM 在 1999 年发布。由于规范很简单，非常适合低功耗和网络带宽有限的 IOT 物联网场景。实际应用于第三方提供的道闸与雷达数据传输。

#### 代码
* Nuget 引用第三方开源库 [MQTTnet](https://github.com/dotnet/MQTTnet)
* 默认服务端端口 1883，可任意修改为未被占用的端口。
* 通过 Topic(主题) 匹配数据，以 "+" 与 "#" 作为通配符，"+" 为单层通配符，"#" 为多层通配符。

##### 服务端
``` CSharp
static async Task Main(string[] args)
{
    var factory = new MqttFactory();

    var options = new MqttServerOptionsBuilder()
        .WithDefaultEndpoint()
        .WithDefaultEndpointPort(1883)
        .Build();

    using (var server = factory.CreateMqttServer(options))
    {
        server.ClientConnectedAsync += e =>
        {
            Console.WriteLine($"{e.ClientId} ClientConnected");
            return Task.FromResult(1);
        };
        server.ClientDisconnectedAsync += e =>
        {
            Console.WriteLine($"{e.ClientId} ClientDisconnected");
            return Task.FromResult(1);
        };
        server.ClientSubscribedTopicAsync += e =>
        {
            Console.WriteLine($"{e.ClientId} ClientSubscribedTopic");
            return Task.FromResult(1);
        };
        server.ClientUnsubscribedTopicAsync += e =>
        {
            Console.WriteLine($"{e.ClientId} ClientUnsubscribedTopic");
            return Task.FromResult(1);
        };
        await server.StartAsync();
        Console.WriteLine("Press Enter to exit.");
        Console.ReadLine();

        await server.StopAsync();
    }
}
```

##### 客户端
``` CSharp
static async Task Main(string[] args)
{
    Thread t1 = new Thread(StartConsumer);
    t1.Start();
    Thread t2 = new Thread(StartPublisher);
    t2.Start();
    Console.WriteLine("Press Enter to exit.");
    Console.ReadLine();
}

static async void StartConsumer()
{
    var factory = new MqttFactory();

    using (var client = factory.CreateManagedMqttClient())
    {
        var options = new MqttClientOptionsBuilder()
            .WithTcpServer("127.0.0.1", 1234)
            .Build();
        var managedClientOptions = new ManagedMqttClientOptionsBuilder()
            .WithClientOptions(options)
            .Build();
        client.ApplicationMessageReceivedAsync += e =>
        {
            Console.WriteLine($"Received message: {Encoding.UTF8.GetString(e.ApplicationMessage.Payload)}");
            return Task.FromResult(1);
        };

        await client.StartAsync(managedClientOptions);

        var subscribeOptions = factory.CreateTopicFilterBuilder()
            .WithTopic($"detrd/status/response/#")
            .Build();
        await client.SubscribeAsync(new List<MqttTopicFilter> { subscribeOptions });

        Console.WriteLine("MQTT client subscribed to topic.");
        Console.ReadLine();
    }
}

static async void StartPublisher()
{
    var factory = new MqttFactory();

    using (var client = factory.CreateManagedMqttClient())
    {
        var options = new MqttClientOptionsBuilder()
            .WithTcpServer("127.0.0.1", 1883)
            .Build();
        var managedClientOptions = new ManagedMqttClientOptionsBuilder()
            .WithClientOptions(options)
            .Build();
        await client.StartAsync(managedClientOptions);

        while (true)
        {
            await client.EnqueueAsync($"detrd/status/response/#", "data");

            SpinWait.SpinUntil(() => client.PendingApplicationMessagesCount == 0, 10000);
            Console.WriteLine("MQTT application message is published.");
            Thread.Sleep(5000);
        }
    }
}
```