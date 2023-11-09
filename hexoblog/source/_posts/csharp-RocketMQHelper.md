---
title: RocketMQ 帮助类
date: 2023-10-24 16:45:00
tags: [c#,rocketmq]
categories: C#.Net
---
### RocketMQ 消息队列帮助类使用介绍
<!-- more -->
### 简介
[RocketMQ Demo](https://sadness96.github.io/blog/2023/08/16/csharp-RocketMQ/) 写了关于 RocketMQ 的基础介绍与官方提供的最简生产者与消费者的 Demo，现在给 c# 与 java 的版本封装一下。

### 核心代码
#### C# RocketMQ 帮助类
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

##### 调用测试
###### 生产者
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

###### 消费者
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

#### JAVA RocketMQ 帮助类
``` Java
import org.apache.rocketmq.acl.common.AclClientRPCHook;
import org.apache.rocketmq.acl.common.SessionCredentials;
import org.apache.rocketmq.client.consumer.DefaultMQPushConsumer;
import org.apache.rocketmq.client.consumer.listener.MessageListenerConcurrently;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.client.producer.DefaultMQProducer;
import org.apache.rocketmq.common.message.Message;
import org.apache.rocketmq.remoting.RPCHook;
import org.apache.rocketmq.remoting.common.RemotingHelper;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyContext;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyStatus;
import org.apache.rocketmq.client.consumer.rebalance.AllocateMessageQueueAveragely;
import org.apache.rocketmq.common.message.MessageExt;
import java.util.List;

/**
 * RocketMQ 消息队列帮助类
 */
public class RocketMQHelper {
    /**
     * 生产者
     */
    private DefaultMQProducer producer;

    /**
     * 消费者
     */
    private DefaultMQPushConsumer consumer;

    /**
     * 消息回调
     */
    private MessageCallback messageCallback;

    /**
     * 注册生产者
     *
     * @param producerGroup 生产组
     * @param namesrv       服务地址
     */
    public void RegisterProducer(String producerGroup, String namesrv) {
        // 实例化一个生产者对象
        producer = new DefaultMQProducer(producerGroup);
        // 设置 Name Server 地址
        producer.setNamesrvAddr(namesrv);

        try {
            // 启动生产者
            producer.start();
        } catch (MQClientException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 注册生产者 ACL 验证
     *
     * @param producerGroup 生产组
     * @param namesrv       服务地址
     * @param aclAccessKey  Access 秘钥
     * @param aclSecretKey  Secret 秘钥
     */
    public void RegisterProducer(String producerGroup, String namesrv, String aclAccessKey, String aclSecretKey) {
        RPCHook rpcHook = new AclClientRPCHook(new SessionCredentials(aclAccessKey, aclSecretKey));
        // 实例化一个生产者对象
        producer = new DefaultMQProducer(producerGroup, rpcHook);
        // 设置 Name Server 地址
        producer.setNamesrvAddr(namesrv);
        // 设置启用 TLS（传输层安全）
        producer.setUseTLS(true);

        try {
            // 启动生产者
            producer.start();
        } catch (MQClientException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 注册消费者
     *
     * @param consumerGroup 消费组
     * @param namesrv       服务地址
     * @param topic         主题
     * @param tag           标签
     */
    public void RegisterConsumer(String consumerGroup, String namesrv, String topic, String tag) {
        // 实例化一个消费者对象
        consumer = new DefaultMQPushConsumer(consumerGroup);
        // 设置 Name Server 地址
        consumer.setNamesrvAddr(namesrv);
        // 设置消息监听器
        consumer.registerMessageListener(new MessageListenerConcurrently() {
            public ConsumeConcurrentlyStatus consumeMessage(List<MessageExt> msgs, ConsumeConcurrentlyContext context) {
                for (MessageExt message : msgs) {
                    if (messageCallback != null) {
                        // 调用消息回调函数传递消息
                        messageCallback.onMessageReceived(new String(message.getBody()));
                    }
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
        } catch (MQClientException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 注册消费者 ACL 验证
     *
     * @param consumerGroup 消费组
     * @param namesrv       服务地址
     * @param topic         主题
     * @param tag           标签
     * @param aclAccessKey  Access 秘钥
     * @param aclSecretKey  Secret 秘钥
     */
    public void RegisterConsumer(String consumerGroup, String namesrv, String topic, String tag, String aclAccessKey, String aclSecretKey) {
        RPCHook rpcHook = new AclClientRPCHook(new SessionCredentials(aclAccessKey, aclSecretKey));
        // 实例化一个消费者对象，添加ACL权限认证，并开启消息轨迹（ConsumerGroupName-消费者组，rpcHook-ACL认证，true-消息轨迹开启）
        consumer = new DefaultMQPushConsumer(null, consumerGroup, rpcHook, new AllocateMessageQueueAveragely(), true, null);
        // 设置 Name Server 地址
        consumer.setNamesrvAddr(namesrv);
        // 设置启用 TLS（传输层安全）
        consumer.setUseTLS(true);
        // 设置消息监听器
        consumer.registerMessageListener(new MessageListenerConcurrently() {
            public ConsumeConcurrentlyStatus consumeMessage(List<MessageExt> msgs, ConsumeConcurrentlyContext context) {
                for (MessageExt message : msgs) {
                    if (messageCallback != null) {
                        // 调用消息回调函数传递消息
                        messageCallback.onMessageReceived(new String(message.getBody()));
                    }
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
        } catch (MQClientException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 发送消息
     *
     * @param message 消息
     * @param topic   主题
     * @param tag     标签
     */
    public void SendMessage(String message, String topic, String tag) {
        try {
            // 发送消息
            producer.send(new Message(
                    topic,  // 主题
                    tag,    // 标签
                    message.getBytes(RemotingHelper.DEFAULT_CHARSET)  // 内容
            ));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 关闭连接
     */
    public void Close() {
        producer.shutdown();
        consumer.shutdown();
    }

    /**
     * 设置消息回调函数
     *
     * @param callback
     */
    public void setMessageCallback(MessageCallback callback) {
        this.messageCallback = callback;
    }

    /**
     * 定义消息回调接口
     */
    public interface MessageCallback {
        void onMessageReceived(String message);
    }
}
```

##### 调用测试
###### 生产者
``` Java
public static void main(String[] args) {
    String producerGroup = "";
    String namesvr = "";
    String topic = "";
    String tag = "*";

    RocketMQHelper rocketMQHelper = new RocketMQHelper();
    rocketMQHelper.RegisterProducer(producerGroup, namesvr);

    while (true) {
        Date currentDate = new Date();
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        String formattedDate = dateFormat.format(currentDate);
        rocketMQHelper.SendMessage(formattedDate, topic, tag);

        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
```

###### 消费者
``` Java
public static void main(String[] args) {
    String consumerGroup = "";
    String namesvr = "";
    String topic = "";
    String tag = "*";

    RocketMQHelper rocketMQHelper = new RocketMQHelper();
    rocketMQHelper.RegisterConsumer(consumerGroup, namesvr, topic, tag);

    rocketMQHelper.setMessageCallback(new RocketMQHelper.MessageCallback() {
        @Override
        public void onMessageReceived(String message) {
            System.out.println("Received: " + message);
        }
    });
}
```