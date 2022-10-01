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
Nuget 引用第三方开源库 [MQTTnet](https://github.com/dotnet/MQTTnet)
##### 服务端
``` CSharp
        static async Task Main(string[] args)
        {
            var factory = new MqttFactory();

            var options = new MqttServerOptionsBuilder()
                .WithDefaultEndpoint()
                .WithDefaultEndpointPort(1234)
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

```