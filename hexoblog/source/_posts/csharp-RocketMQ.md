---
title: RocketMQ Demo
date: 2023-08-16 14:42:00
tags: [c#,go,java,rocketmq]
categories: C#.Net
---
### RocketMQ 消息队列使用介绍
<!-- more -->
#### 简介
[Apache RocketMQ](https://rocketmq.apache.org/) 是阿里巴巴开发的分布式消息中间件，后捐赠给 Apache 基金会维护。
直接对接华为云，所以未在本地安装。测试对 C# 兼容不是很好，而且服务端都逐渐使用 docker 部署，所以最终选择使用 Golang 开发。

#### C# 代码调用
官方库 [rocketmq-client-csharp](https://github.com/apache/rocketmq-client-csharp) 的支持似乎并不好，调试了几次都运行不起来。
引用第三方 [NewLife.RocketMQ](https://github.com/NewLifeX/NewLife.RocketMQ) 库。测试不支持 ACL（权限控制），如开启 ACL 则无法连接。

##### 生产者
``` csharp
class Program
{
    public static string namesvr = "";
    public static string topic = "";

    static void Main(string[] args)
    {
        ThreadStart str = new ThreadStart(Producer);
        Thread ConstrolStr = new Thread(str);
        ConstrolStr.Start();
    }

    /// <summary>
    /// 生产者
    /// </summary>
    /// <param name="args"></param>
    static void Producer()
    {
        // MQ 对象
        var mq = new Producer()
        {
            // 主题
            Topic = topic,
            // 服务地址
            NameServerAddress = namesvr,
        };

        mq.Start();

        // 轮询发消息
        while (true)
        {
            try
            {
                var content = $"{DateTime.Now}";
                var message = new NewLife.RocketMQ.Protocol.Message();
                message.SetBody(content);
                // 发送消息（生产消息）
                var sr = mq.Publish(message);
                Console.WriteLine($"发送成功的消息，内容：{content}");
                Task.Delay(TimeSpan.FromMilliseconds(500)).Wait();
            }
            catch (Exception ex)
            {
                //throw;
            }
        }
    }
}
```

##### 消费者
``` csharp
class Program
{
    public static string namesvr = "";
    public static string topic = "";
    public static string group = "";

    static void Main(string[] args)
    {
        ThreadStart num = new ThreadStart(Consumer);
        Thread ConstrolNum = new Thread(num);
        ConstrolNum.Start();
    }

    /// <summary>
    /// 消费者
    /// </summary>
    /// <param name="args"></param>
    static void Consumer()
    {
        // 测试消费消息
        var consumer = new NewLife.RocketMQ.Consumer
        {
            Topic = topic,
            NameServerAddress = namesvr,
            // 设置每次接收消息只拉取一条信息
            BatchSize = 1,
            Group = group
        };
        consumer.OnConsume = (q, ms) =>
        {
            string mInfo = $"BrokerName={q.BrokerName},QueueId={q.QueueId},Length={ms.Length}";
            Console.WriteLine(mInfo);
            foreach (var item in ms.ToList())
            {
                string msg = string.Format($"接收到消息：msgId={item.MsgId},key={item.Keys}，产生时间【{item.BornTimestamp.ToDateTime()}】，内容：{item.BodyString}");
                Console.WriteLine(msg);
            }
            // return false; // 通知消息队：不消费消息
            return true; // 通知消息队：消费了消息
        };

        consumer.Start();
    }
}
```

#### Golang 代码调用
引用 [rocketmq-client-go](github.com/apache/rocketmq-client-go) 库。

##### 生产者
``` go
var server string = string()
var topic string = string()

p, _ := rocketmq.NewProducer(
    producer.WithNsResolver(primitive.NewPassthroughResolver([]string{server})),
    producer.WithRetry(2),
)
err := p.Start()
if err != nil {
    fmt.Printf("start producer error: %s", err.Error())
    os.Exit(1)
}
msg := &primitive.Message{
    Topic: topic, 
    Body:  []byte("Message"),
}
msg.WithTag("TagName")
msg.WithKeys([]string{"KeyName"})

for {
    res, err := p.SendSync(context.Background(), msg)
    if err != nil {
        fmt.Printf("send message error: %s\n", err)
    } else {
        fmt.Printf("send message success: result=%s\n", res.String())
    }

    time.Sleep(1000000000)
}

err = p.Shutdown()
if err != nil {
    fmt.Printf("shutdown producer error: %s", err.Error())
}
```

##### 消费者
``` go
var server string = string()
var topic string = string()
var group string = string()

c, _ := rocketmq.NewPushConsumer(
    consumer.WithGroupName(group),
    consumer.WithNsResolver(primitive.NewPassthroughResolver([]string{server})), 
)
err := c.Subscribe(topic, consumer.MessageSelector{}, func(ctx context.Context, 
    msgs ...*primitive.MessageExt) (consumer.ConsumeResult, error) {
    for i := range msgs {
        fmt.Printf("subscribe callback: %v \n", string(msgs[i].Message.Body))
    }
    return consumer.ConsumeSuccess, nil
})
if err != nil {
    fmt.Println(err.Error())
}
// Note: start after subscribe
err = c.Start()
if err != nil {
    fmt.Println(err.Error())
    os.Exit(-1)
}
time.Sleep(time.Hour)
err = c.Shutdown()
if err != nil {
    fmt.Printf("shutdown Consumer error: %s", err.Error())
}
```

#### Java 代码调用
通过 Maven 引用 rocketmq-client 等相关库。

##### Maven 引用库
``` xml
<dependencies>
    <dependency>
        <groupId>org.apache.rocketmq</groupId>
        <artifactId>rocketmq-client</artifactId>
        <version>4.9.0</version>
    </dependency>
    <dependency>
        <groupId>org.apache.rocketmq</groupId>
        <artifactId>rocketmq-acl</artifactId>
        <version>4.9.0</version>
    </dependency>
</dependencies>
```

##### 生产者
``` java
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.client.producer.DefaultMQProducer;
import org.apache.rocketmq.common.message.Message;
import org.apache.rocketmq.remoting.common.RemotingHelper;

import java.util.Date;
import java.text.SimpleDateFormat;

public class Main {
    public static void main(String[] args) {
        String producerGroup = "";
        String namesvr = "";
        String topic = "";
        String tag = "*";

        // 实例化一个生产者对象
        DefaultMQProducer producer = new DefaultMQProducer(producerGroup);

        // 设置 Name Server 地址
        producer.setNamesrvAddr(namesvr);

        // 启动生产者
        try {
            producer.start();
        } catch (MQClientException e) {
            throw new RuntimeException(e);
        }

        try {
            while (true) {
                // 创建一个日期对象
                Date currentDate = new Date();
                // 创建一个日期格式化对象，指定日期时间格式
                SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                // 使用日期格式化对象将日期对象格式化为字符串
                String formattedDate = dateFormat.format(currentDate);

                // 创建消息对象，指定消息主题、标签和内容
                Message message = new Message(
                        topic,  // 主题
                        tag,    // 标签
                        formattedDate.getBytes(RemotingHelper.DEFAULT_CHARSET)  // 内容
                );

                // 发送消息
                producer.send(message);
                System.out.println("消息发送成功");
                Thread.sleep(500);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // 关闭生产者
            producer.shutdown();
        }
    }
}
```

##### 生产者 ACL 验证
``` java
import org.apache.rocketmq.acl.common.AclClientRPCHook;
import org.apache.rocketmq.acl.common.SessionCredentials;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.client.producer.DefaultMQProducer;
import org.apache.rocketmq.common.message.Message;
import org.apache.rocketmq.remoting.RPCHook;
import org.apache.rocketmq.remoting.common.RemotingHelper;

import java.util.Date;
import java.text.SimpleDateFormat;

public class Main {
    public static void main(String[] args) {
        String producerGroup = "";
        String namesvr = "";
        String topic = "";
        String tag = "*";

        String ACL_ACCESS_KEY = "";
        String ACL_SECRET_KEY = "";

        RPCHook rpcHook = new AclClientRPCHook(new SessionCredentials(ACL_ACCESS_KEY, ACL_SECRET_KEY));
        // 实例化一个生产者对象
        DefaultMQProducer producer = new DefaultMQProducer(producerGroup, rpcHook);

        // 设置 Name Server 地址
        producer.setNamesrvAddr(namesvr);
        // 设置启用 TLS（传输层安全）
        producer.setUseTLS(true);

        // 启动生产者
        try {
            producer.start();
        } catch (MQClientException e) {
            throw new RuntimeException(e);
        }

        try {
            while (true) {
                // 创建一个日期对象
                Date currentDate = new Date();
                // 创建一个日期格式化对象，指定日期时间格式
                SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                // 使用日期格式化对象将日期对象格式化为字符串
                String formattedDate = dateFormat.format(currentDate);

                // 创建消息对象，指定消息主题、标签和内容
                Message message = new Message(
                        topic,  // 主题
                        tag,    // 标签
                        formattedDate.getBytes(RemotingHelper.DEFAULT_CHARSET)  // 内容
                );

                // 发送消息
                producer.send(message);
                System.out.println("消息发送成功");
                Thread.sleep(500);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // 关闭生产者
            producer.shutdown();
        }
    }
}
```

##### 消费者
``` java
import org.apache.rocketmq.client.consumer.DefaultMQPushConsumer;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyContext;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyStatus;
import org.apache.rocketmq.client.consumer.listener.MessageListenerConcurrently;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.common.message.MessageExt;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        String consumerGroup = "";
        String namesvr = "";
        String topic = "";
        String tag = "*";

        // 实例化一个消费者对象
        DefaultMQPushConsumer consumer = new DefaultMQPushConsumer(consumerGroup);

        // 设置 Name Server 地址
        consumer.setNamesrvAddr(namesvr);

        // 设置消息监听器
        consumer.registerMessageListener(new MessageListenerConcurrently() {
            @Override
            public ConsumeConcurrentlyStatus consumeMessage(List<MessageExt> msgs, ConsumeConcurrentlyContext context) {
                for (MessageExt message : msgs) {
                    System.out.println("收到消息：" + new String(message.getBody()));
                }

                // 消费成功后返回状态
                return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
            }
        });

        try {
            // 订阅主题和标签
            consumer.subscribe(topic, tag);
            // 启动消费者
            consumer.start();

            System.out.println("消费者启动成功");
        } catch (MQClientException e) {
            throw new RuntimeException(e);
        }
    }
}
```

##### 消费者 ACL 验证
``` java
import org.apache.rocketmq.acl.common.AclClientRPCHook;
import org.apache.rocketmq.acl.common.SessionCredentials;
import org.apache.rocketmq.client.consumer.DefaultMQPushConsumer;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyContext;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyStatus;
import org.apache.rocketmq.client.consumer.listener.MessageListenerConcurrently;
import org.apache.rocketmq.client.consumer.rebalance.AllocateMessageQueueAveragely;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.common.message.MessageExt;
import org.apache.rocketmq.remoting.RPCHook;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        String consumerGroup = "";
        String namesvr = "";
        String topic = "";
        String tag = "*";

        String ACL_ACCESS_KEY = "";
        String ACL_SECRET_KEY = "";

        RPCHook rpcHook = new AclClientRPCHook(new SessionCredentials(ACL_ACCESS_KEY, ACL_SECRET_KEY));
        // 添加ACL权限认证，并开启消息轨迹（ConsumerGroupName-消费者组，rpcHook-ACL认证，true-消息轨迹开启）
        DefaultMQPushConsumer consumer = new DefaultMQPushConsumer(null, consumerGroup, rpcHook, new AllocateMessageQueueAveragely(), true, null);

        // 设置 Name Server 地址
        consumer.setNamesrvAddr(namesvr);
        // 设置启用 TLS（传输层安全）
        consumer.setUseTLS(true);

        // 设置消息监听器
        consumer.registerMessageListener(new MessageListenerConcurrently() {
            @Override
            public ConsumeConcurrentlyStatus consumeMessage(List<MessageExt> msgs, ConsumeConcurrentlyContext context) {
                for (MessageExt message : msgs) {
                    System.out.println("收到消息：" + new String(message.getBody()));
                }

                // 消费成功后返回状态
                return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
            }
        });

        try {
            // 订阅主题和标签
            consumer.subscribe(topic, tag);
            // 启动消费者
            consumer.start();

            System.out.println("消费者启动成功");
        } catch (MQClientException e) {
            throw new RuntimeException(e);
        }
    }
}
```