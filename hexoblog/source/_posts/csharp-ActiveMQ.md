---
title: ActiveMQ Demo
date: 2019-12-31 17:50:00
tags: [c#,activemq]
categories: C#.Net
---
### ActiveMQ 消息队列使用介绍
<!-- more -->
#### 简介
[Apache ActiveMQ](https://activemq.apache.org/) 是一个开放源代码的消息中间件。

#### 安装部署
请参阅[官方文档](https://activemq.apache.org/components/classic/documentation)

##### Docker 部署
``` cmd
# 拉取官方 ActiveMQ
docker pull webcenter/activemq
# 启动 ActiveMQ
docker run -d --name myactivemq -p 61616:61616 -p 8161:8161 webcenter/activemq
# WEB 端登录 默认账户密码为 admin/admin
http://localhost:8161/
```

#### C# 代码调用
引用 [Apache.NMS.ActiveMQ](https://cwiki.apache.org/confluence/display/NMS/Apache.NMS.ActiveMQ) 库。

##### 生产者
``` CSharp
using Apache.NMS;
using Apache.NMS.ActiveMQ;
using System;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            //Create the Connection Factory
            IConnectionFactory factory = new ConnectionFactory("tcp://localhost:61616/");
            using (IConnection connection = factory.CreateConnection())
            {
                //Create the Session
                using (ISession session = connection.CreateSession())
                {
                    //Create the Producer for the topic/queue
                    IMessageProducer prod = session.CreateProducer(new Apache.NMS.ActiveMQ.Commands.ActiveMQTopic("testing"));
                    //Send Messages
                    int i = 0;
                    while (!Console.KeyAvailable)
                    {
                        ITextMessage msg = prod.CreateTextMessage();
                        msg.Text = i.ToString();
                        Console.WriteLine("Sending: " + i.ToString());
                        prod.Send(msg, Apache.NMS.MsgDeliveryMode.NonPersistent, Apache.NMS.MsgPriority.Normal, TimeSpan.MinValue);
                        System.Threading.Thread.Sleep(5000);
                        i++;
                    }
                }
            }
            Console.ReadLine();
        }
        catch (System.Exception e)
        {
            Console.WriteLine("{0}", e.Message);
            Console.ReadLine();
        }
    }
}
```

##### 消费者
``` CSharp
using Apache.NMS;
using Apache.NMS.ActiveMQ;
using System;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            //Create the Connection factory
            IConnectionFactory factory = new ConnectionFactory("tcp://localhost:61616/");
            //Create the connection
            using (IConnection connection = factory.CreateConnection())
            {
                connection.ClientId = "testing listener";
                connection.Start();
                //Create the Session
                using (ISession session = connection.CreateSession())
                {
                    //Create the Consumer
                    IMessageConsumer consumer = session.CreateDurableConsumer(new Apache.NMS.ActiveMQ.Commands.ActiveMQTopic("testing"), "testing listener", null, false);
                    consumer.Listener += new MessageListener(consumer_Listener);
                    Console.ReadLine();
                }
                connection.Stop();
                connection.Close();
            }
        }
        catch (System.Exception e)
        {
            Console.WriteLine(e.Message);
        }
    }
    static void consumer_Listener(IMessage message)
    {
        try
        {
            ITextMessage msg = (ITextMessage)message;
            Console.WriteLine("Receive: " + msg.Text);
        }
        catch (System.Exception e)
        {
            Console.WriteLine(e.Message);
        }
    }
}
```