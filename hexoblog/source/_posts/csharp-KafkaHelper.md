---
title: Kafka 帮助类
date: 2023-10-25 18:00:00
tags: [c#,kafka]
categories: C#.Net
---
### Kafka 消息队列帮助类使用介绍
<!-- more -->
### 简介
[Kafka Demo](https://sadness96.github.io/blog/2019/09/16/csharp-Kafka/) 写了关于 Kafka 的基础介绍与官方提供的最简生产者与消费者的 Demo，完善代码封装为帮助类。

### 核心代码
[KafkaHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Queue.Helper/Kafka/KafkaHelper.cs)

``` CSharp
/// <summary>
/// Kafka 消息队列帮助类
/// 创建日期:2023年10月25日
/// </summary>
public class KafkaHelper
{
    private string _bootstrapServers;
    Action<DeliveryReport<Null, string>> _handler;
    private IProducer<Null, string> _producerBuilder;

    /// <summary>
    /// 消息回调
    /// </summary>
    public event Action<string> MessageCallback;

    /// <summary>
    /// 构造函数
    /// </summary>
    /// <param name="bootstrapServers">服务地址</param>
    public KafkaHelper(string bootstrapServers)
    {
        // 127.0.0.1:9092
        _bootstrapServers = bootstrapServers;
    }

    /// <summary>
    /// 注册生产者
    /// </summary>
    public void RegisterProducer()
    {
        var conf = new ProducerConfig { BootstrapServers = _bootstrapServers };

        _handler = r => Console.WriteLine(!r.Error.IsError
            ? $"Delivered message to {r.TopicPartitionOffset}"
            : $"Delivery Error: {r.Error.Reason}");

        _producerBuilder = new ProducerBuilder<Null, string>(conf).Build();
    }

    /// <summary>
    /// 注册消费者
    /// </summary>
    /// <param name="groupId">消费组</param>
    /// <param name="topic">主题</param>
    /// <param name="autoOffsetReset">消费偏移量</param>
    /// <param name="kafkaSSL">SSL 验证</param>
    public void RegisterConsumer(string groupId, string topic, AutoOffsetReset autoOffsetReset = AutoOffsetReset.Earliest, KafkaSSL kafkaSSL = null)
    {
        var conf = new ConsumerConfig
        {
            GroupId = groupId,
            BootstrapServers = _bootstrapServers,
            // Note: The AutoOffsetReset property determines the start offset in the event
            // there are not yet any committed offsets for the consumer group for the
            // topic/partitions of interest. By default, offsets are committed
            // automatically, so in this example, consumption will only start from the
            // earliest message in the topic 'my-topic' the first time you run the program.
            AutoOffsetReset = autoOffsetReset
        };

        // 如果 Kafka 开启了 SSL 验证，则需要填写以下信息，否则删除
        if (kafkaSSL != null)
        {
            conf.SecurityProtocol = kafkaSSL.SecurityProtocol;
            conf.SaslMechanism = kafkaSSL.SaslMechanism;
            conf.SaslUsername = kafkaSSL.SaslUsername;
            conf.SaslPassword = kafkaSSL.SaslPassword;
            conf.SslCaLocation = kafkaSSL.SslCaLocation;
            conf.SslKeystorePassword = kafkaSSL.SslKeystorePassword;
            conf.SslEndpointIdentificationAlgorithm = kafkaSSL.SslEndpointIdentificationAlgorithm;
        }

        using (var c = new ConsumerBuilder<Ignore, string>(conf).Build())
        {
            c.Subscribe(topic);

            CancellationTokenSource cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) =>
            {
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
                        //Console.WriteLine($"Consumed message '{cr.Value}' at: '{cr.TopicPartitionOffset}'.");
                        MessageCallback?.Invoke(cr.Value);
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

    /// <summary>
    /// 发送消息
    /// </summary>
    public void SendMessage(string topic, string message)
    {
        _producerBuilder.Produce(topic, new Message<Null, string> { Value = message }, _handler);

        // wait for up to 10 seconds for any inflight messages to be delivered.
        _producerBuilder.Flush(TimeSpan.FromSeconds(10));
    }

    /// <summary>
    /// 关闭连接
    /// </summary>
    public void Close()
    {
        _producerBuilder.Dispose();
    }
}

/// <summary>
/// Kafka SSL 验证
/// </summary>
public class KafkaSSL
{
    /// <summary>
    /// Protocol used to communicate with brokers. default: plaintext importance: high
    /// </summary>
    public SecurityProtocol? SecurityProtocol { get; set; } = Confluent.Kafka.SecurityProtocol.SaslSsl;

    /// <summary>
    /// SASL mechanism to use for authentication. Supported: GSSAPI, PLAIN, SCRAM-SHA-256,
    /// SCRAM-SHA-512. **NOTE**: Despite the name, you may not configure more than one
    /// mechanism.
    /// </summary>
    public SaslMechanism? SaslMechanism { get; set; } = Confluent.Kafka.SaslMechanism.Plain;

    /// <summary>
    /// SASL username for use with the PLAIN and SASL-SCRAM-.. mechanisms default: ''
    /// importance: high
    /// </summary>
    public string SaslUsername { get; set; }

    /// <summary>
    /// SASL password for use with the PLAIN and SASL-SCRAM-.. mechanism default: ''
    /// importance: high
    /// </summary>
    public string SaslPassword { get; set; }

    /// <summary>
    /// File or directory path to CA certificate(s) for verifying the broker's key. Defaults:
    /// On Windows the system's CA certificates are automatically looked up in the Windows
    /// Root certificate store. On Mac OSX this configuration defaults to `probe`. It
    /// is recommended to install openssl using Homebrew, to provide CA certificates.
    /// On Linux install the distribution's ca-certificates package. If OpenSSL is statically
    /// linked or `ssl.ca.location` is set to `probe` a list of standard paths will be
    /// probed and the first one found will be used as the default CA certificate location
    /// path. If OpenSSL is dynamically linked the OpenSSL library's default path will
    /// be used (see `OPENSSLDIR` in `openssl version -a`). default: '' importance: low
    /// </summary>
    public string SslCaLocation { get; set; }

    /// <summary>
    /// Client's keystore (PKCS#12) password. default: '' importance: low
    /// </summary>
    public string SslKeystorePassword { get; set; }

    /// <summary>
    /// Endpoint identification algorithm to validate broker hostname using broker certificate.
    /// https - Server (broker) hostname verification as specified in RFC2818. none -
    /// No endpoint verification. OpenSSL >= 1.0.2 required. default: https importance:
    /// low
    /// </summary>
    public SslEndpointIdentificationAlgorithm? SslEndpointIdentificationAlgorithm { get; set; }
}
```

### 调用测试
#### 生产者
``` CSharp
static void Main(string[] args)
{
    // 服务地址
    string bootstrapServers = "127.0.0.1:9092";
    // 主题
    string topic;

    KafkaHelper kafkaHelper = new KafkaHelper(bootstrapServers);
    kafkaHelper.RegisterProducer();
    
    while (true)
    {
        kafkaHelper.SendMessage(topic, $"{DateTime.Now}");
        Task.Delay(TimeSpan.FromMilliseconds(500)).Wait();
    }
}
```

#### 消费者
``` CSharp
static void Main(string[] args)
{
    // 服务地址
    string bootstrapServers = "127.0.0.1:9092";
    // 主题
    string topic;
    // 消费组
    string group;

    KafkaHelper kafkaHelper = new KafkaHelper(bootstrapServers);
    kafkaHelper.MessageCallback += KafkaHelper_MessageCallback;
    Task.Run(() =>
    {
        kafkaHelper.RegisterConsumer(group, topic);
    });
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

#### 消费者（SSL 验证）
``` CSharp
static void Main(string[] args)
{
    // 服务地址
    string bootstrapServers = "127.0.0.1:9092";
    // 主题
    string topic;
    // 消费组
    string group;

    KafkaHelper kafkaHelper = new KafkaHelper(bootstrapServers);
    kafkaHelper.MessageCallback += KafkaHelper_MessageCallback;
    Task.Run(() =>
    {
        KafkaSSL kafkaSSL = new KafkaSSL();
        kafkaSSL.SecurityProtocol = SecurityProtocol.SaslSsl;
        kafkaSSL.SaslMechanism = SaslMechanism.Plain;
        kafkaSSL.SaslUsername = "xxx";
        kafkaSSL.SaslPassword = "xxx";
        kafkaSSL.SslCaLocation = "phy_ca.crt";
        kafkaSSL.SslKeystorePassword = "dms@kafka";
        kafkaSSL.SslEndpointIdentificationAlgorithm = null;

        kafkaHelper.RegisterConsumer(group, topic, kafkaSSL: kafkaSSL);
    });
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