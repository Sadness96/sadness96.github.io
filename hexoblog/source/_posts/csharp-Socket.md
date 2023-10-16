---
title: Socket 帮助类
date: 2023-09-15 16:10:00
tags: [c#,socket]
categories: C#.Net
---
### Socket Tcp 与 Udp 帮助类使用介绍
<!-- more -->
### 简介
Socket 是一种用于网络通信的编程接口，它允许不同计算机上的程序通过网络进行数据交换。

Socket 通常分为服务端与客户端：
* 服务端 Socket 负责监听客户端 Socket 的连接请求，并为每个连接请求创建一个新的 Socket 实例。服务端 Socket 处理客户端的请求并发送响应。
* 客户端 Socket 通常是由想要与服务器进行通信的应用程序创建的。客户端 Socket 负责与服务器 Socket 建立连接、发送请求和接收响应。

Socket的实现基于 TCP（传输控制协议）和 UDP（用户数据报协议）的网络通信：
* TCP（传输控制协议）：TCP 是一种面向连接的协议，它提供可靠的、有序的数据传输。在使用 TCP 时，建立连接是必要的，因此客户端 Socket 需要与服务器 Socket 建立连接，然后通过这个连接来发送和接收数据。TCP 可以确保数据的可靠性，保证数据按照发送的顺序到达目的地，还具备拥塞控制和错误恢复的机制。
* UDP（用户数据报协议）：UDP 是一种无连接的协议，它提供不可靠的、无序的数据传输。与 TCP 不同，使用 UDP 不需要事先建立连接，每个数据包（也称为数据报）都是独立发送的。UDP 更适合一些实时性要求较高、数据传输较少关注可靠性的应用场景，如视频流、音频传输和在线游戏。

### 核心代码
#### TCP Socket
基于 TCP 协议的 Socket 的实现

##### TCP Socket Server
[TcpServerHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Queue.Helper/Socket/TcpServerHelper.cs)

``` CSharp
/// <summary>
/// Socket Tcp 服务端帮助类
/// </summary>
public class TcpServerHelper
{
    private TcpListener listener;
    private List<TcpClient> clients;
    private Thread listenThread;

    /// <summary>
    /// 收到数据回调
    /// </summary>
    public event Action<EndPoint, string> OnDataReceived;

    /// <summary>
    ///  Socket Tcp 服务端构造函数
    /// </summary>
    /// <param name="port">服务端监听端口</param>
    public TcpServerHelper(int port)
    {
        Console.WriteLine($"Socket tcp Server listen port:{port}");

        listener = new TcpListener(IPAddress.Any, port);
        clients = new List<TcpClient>();
    }

    /// <summary>
    /// 启动
    /// </summary>
    public void Start()
    {
        listenThread = new Thread(ListenForClients);
        listenThread.Start();
    }

    /// <summary>
    /// 停止
    /// </summary>
    public void Stop()
    {
        listenThread?.Abort();

        lock (clients)
        {
            foreach (var client in clients)
            {
                client.Close();
            }

            clients.Clear();
        }
    }

    /// <summary>
    /// 监听客户端连接请求线程
    /// </summary>
    private void ListenForClients()
    {
        listener.Start();

        while (true)
        {
            TcpClient client = listener.AcceptTcpClient();
            lock (clients)
            {
                Console.WriteLine($"Add client connection:{client.Client.RemoteEndPoint}");
                clients.Add(client);
            }

            Thread clientThread = new Thread(HandleClientCommunication);
            clientThread.Start(client);
        }
    }

    /// <summary>
    /// 处理与客户端的通信线程
    /// </summary>
    /// <param name="clientObj"></param>
    private void HandleClientCommunication(object clientObj)
    {
        TcpClient client = (TcpClient)clientObj;
        string clientAddress = ((IPEndPoint)client.Client.RemoteEndPoint).Address.ToString();
        byte[] buffer = new byte[1024];

        while (true)
        {
            try
            {
                int bytesRead = client.GetStream().Read(buffer, 0, buffer.Length);
                if (bytesRead > 0)
                {
                    string receivedData = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    OnDataReceived?.Invoke(client.Client.RemoteEndPoint, receivedData);
                }
            }
            catch
            {
                break;
            }
        }

        lock (clients)
        {
            Console.WriteLine($"Break client connection:{client.Client.RemoteEndPoint}");
            clients.Remove(client);
        }

        client.Close();
    }

    /// <summary>
    /// 发送消息给所有客户端
    /// </summary>
    /// <param name="data">消息</param>
    public void SendDataToAll(string data)
    {
        lock (clients)
        {
            foreach (var client in clients)
            {
                SendData(client, data);
            }
        }
    }

    /// <summary>
    /// 发送消息给指定客户端
    /// </summary>
    /// <param name="client">客户端</param>
    /// <param name="data">消息</param>
    public void SendDataToClient(TcpClient client, string data)
    {
        if (!clients.Contains(client))
        {
            Console.WriteLine($"Client not connected to the server.");
            return;
        }

        SendData(client, data);
    }

    /// <summary>
    /// 发送消息
    /// </summary>
    /// <param name="client">客户端连接</param>
    /// <param name="data">数据</param>
    private void SendData(TcpClient client, string data)
    {
        byte[] buffer = Encoding.UTF8.GetBytes(data);
        client.GetStream().Write(buffer, 0, buffer.Length);
    }
}
```

##### TCP Socket Client
[TcpClientHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Queue.Helper/Socket/TcpClientHelper.cs)

``` CSharp
/// <summary>
/// Socket Tcp 客户端帮助类
/// </summary>
public class TcpClientHelper
{
    private TcpClient client;
    private byte[] buffer;

    /// <summary>
    /// 收到数据回调
    /// </summary>
    public event Action<string> OnDataReceived;

    /// <summary>
    /// 是否连接
    /// </summary>
    public bool IsConnected => client != null && client.Connected;

    /// <summary>
    /// 连接服务端
    /// </summary>
    /// <param name="ipAddress">IP</param>
    /// <param name="port">端口号</param>
    public void Connect(string ipAddress, int port)
    {
        buffer = new byte[1024];

        try
        {
            Console.WriteLine($"Socket tcp Client connection to {ipAddress}:{port}");

            client = new TcpClient();
            // 连接到服务端
            client.Connect(ipAddress, port);
            // 开始接收数据
            client.GetStream().BeginRead(buffer, 0, buffer.Length, ReceiveCallback, null);
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex.ToString());
        }
    }

    /// <summary>
    /// 断开连接释放资源
    /// </summary>
    public void Disconnect()
    {
        client?.Close();
        client = null;
    }

    /// <summary>
    /// 接收数据回调
    /// </summary>
    /// <param name="ar"></param>
    private void ReceiveCallback(IAsyncResult ar)
    {
        if (!IsConnected)
        {
            Console.WriteLine($"Client is not connected to a server.");
            return;
        }

        try
        {
            int bytesRead = client.GetStream().EndRead(ar);

            if (bytesRead > 0)
            {
                string receivedData = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                OnDataReceived?.Invoke(receivedData);
            }

            // 继续接收数据
            client.GetStream().BeginRead(buffer, 0, buffer.Length, ReceiveCallback, null);
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex.ToString());
        }
    }

    /// <summary>
    /// 发送消息
    /// </summary>
    /// <param name="data">消息</param>
    public void SendData(string data)
    {
        if (!IsConnected)
        {
            Console.WriteLine($"Client is not connected to a server.");
            return;
        }

        byte[] buffer = Encoding.UTF8.GetBytes(data);
        client.GetStream().Write(buffer, 0, buffer.Length);
    }
}
```

#### UDP Socket
基于 UDP 协议的 Socket 的实现

##### UDP Socket Server
[UdpServerHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Queue.Helper/Socket/UdpServerHelper.cs)

``` CSharp
/// <summary>
/// Socket UDP 服务端帮助类
/// </summary>
public class UdpServerHelper
{
    private UdpClient udpServer;
    private List<IPEndPoint> clients;
    private Thread receiveThread;

    /// <summary>
    /// 收到数据回调
    /// </summary>
    public event Action<IPEndPoint, string> OnDataReceived;

    /// <summary>
    /// Socket UDP 服务端构造函数
    /// </summary>
    /// <param name="port">服务端监听端口</param>
    public UdpServerHelper(int port)
    {
        Console.WriteLine($"Socket UDP Server listen port:{port}");

        udpServer = new UdpClient(port);
        clients = new List<IPEndPoint>();
    }

    /// <summary>
    /// 启动
    /// </summary>
    public void Start()
    {
        receiveThread = new Thread(ReceiveData);
        receiveThread.Start();
    }

    /// <summary>
    /// 停止
    /// </summary>
    public void Stop()
    {
        receiveThread?.Abort();
        udpServer?.Close();
    }

    /// <summary>
    /// 接收数据线程
    /// </summary>
    private void ReceiveData()
    {
        IPEndPoint clientEndPoint = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            try
            {
                byte[] receivedBytes = udpServer.Receive(ref clientEndPoint);
                string receivedData = Encoding.UTF8.GetString(receivedBytes);
                OnDataReceived?.Invoke(clientEndPoint, receivedData);

                if (!clients.Contains(clientEndPoint))
                {
                    Console.WriteLine($"Add client connection:{clientEndPoint}");
                    clients.Add(clientEndPoint);
                }

            }
            catch (SocketException)
            {
                // SocketException will be thrown when the thread is aborted or the underlying socket is closed
                Console.WriteLine($"Break client connection:{clientEndPoint}");
                clients.Remove(clientEndPoint);
            }
        }
    }

    /// <summary>
    /// 发送消息给所有客户端
    /// </summary>
    /// <param name="data">消息</param>
    public void SendDataToAll(string data)
    {
        lock (clients)
        {
            foreach (var client in clients)
            {
                SendData(client, data);
            }
        }
    }

    /// <summary>
    /// 发送消息给指定客户端
    /// </summary>
    /// <param name="client">客户端</param>
    /// <param name="data">消息</param>
    /// <returns>发送是否成功</returns>
    public bool SendDataToClient(IPEndPoint client, string data)
    {
        if (!clients.Contains(client))
        {
            Console.WriteLine($"Client not connected to the server.");
            return false;
        }

        SendData(client, data);
        return true;
    }

    /// <summary>
    /// 发送消息
    /// </summary>
    /// <param name="client">客户端连接</param>
    /// <param name="data">数据</param>
    private void SendData(IPEndPoint client, string data)
    {
        byte[] sendData = Encoding.UTF8.GetBytes(data);
        udpServer.Send(sendData, sendData.Length, client);
    }
}
```

##### UDP Socket Client
[UdpClientHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Queue.Helper/Socket/UdpClientHelper.cs)

``` CSharp
/// <summary>
/// Socket UDP 客户端帮助类
/// </summary>
public class UdpClientHelper
{
    private UdpClient udpClient;
    private IPEndPoint serverEndPoint;
    private Thread receiveThread;

    /// <summary>
    /// 收到数据回调
    /// </summary>
    public event Action<string> OnDataReceived;

    /// <summary>
    /// 连接到服务端
    /// </summary>
    /// <param name="ipAddress">IP</param>
    /// <param name="port">端口号</param>
    public void Connect(string ipAddress, int port)
    {
        serverEndPoint = new IPEndPoint(IPAddress.Parse(ipAddress), port);
        udpClient = new UdpClient();

        IPEndPoint localEndPoint = new IPEndPoint(IPAddress.Any, 0);
        udpClient.Client.Bind(localEndPoint);

        receiveThread = new Thread(ReceiveCallback);
        receiveThread.Start();
    }

    /// <summary>
    /// 断开连接释放资源
    /// </summary>
    public void Disconnect()
    {
        receiveThread?.Abort();
        udpClient?.Close();
        udpClient = null;
    }

    /// <summary>
    /// 接收数据回调
    /// </summary>
    private void ReceiveCallback()
    {
        IPEndPoint senderEndPoint = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            try
            {
                byte[] receivedData = udpClient.Receive(ref senderEndPoint);
                string message = Encoding.UTF8.GetString(receivedData);
                OnDataReceived?.Invoke(message);
            }
            catch (SocketException e)
            {
                // SocketException will be thrown when the thread is aborted or the underlying socket is closed
                Console.WriteLine($"UDP receive thread stopped: {e.Message}");
            }
        }
    }

    /// <summary>
    /// 发送消息
    /// </summary>
    /// <param name="data">消息</param>
    public void SendData(string data)
    {
        byte[] sendData = Encoding.UTF8.GetBytes(data);
        udpClient.Send(sendData, sendData.Length, serverEndPoint);
    }
}
```

### 调用测试
#### TCP Socket
##### TCP Socket Server
``` CSharp
// 监听端口
int listenPort;

// 启动 TCP Server 监听客户端连接
TcpServerHelper _tcpServerHelper = new TcpServerHelper(listenPort);
_tcpServerHelper.OnDataReceived += TcpServerHelper_OnDataReceived;
_tcpServerHelper.Start();

// 客户端
TcpClient client;
// 发送数据
string data;

// 发送消息给指定客户端
_tcpServerHelper.SendDataToClient(client, data);

// 发送消息给所有客户端
_tcpServerHelper.SendDataToAll(data);

/// <summary>
/// 消息回调
/// </summary>
/// <param name="arg1"></param>
/// <param name="arg2"></param>
private static void TcpServerHelper_OnDataReceived(System.Net.EndPoint arg1, string arg2)
{
    // 接收数据
}
```

##### TCP Socket Client
``` CSharp
// IP
string ip;
// 端口号
int port;

// 启动 TCP Client 连接到 Server
TcpClientHelper _tcpClientHelper = new TcpClientHelper();
_tcpClientHelper.OnDataReceived += TcpClientHelper_OnDataReceived;
_tcpClientHelper.Connect(ip, port);

// 发送数据
string data;

// 发送消息给服务端
_tcpClientHelper.SendData(data);

/// <summary>
/// 消息回调
/// </summary>
/// <param name="obj"></param>
private static void TcpClientHelper_OnDataReceived(string obj)
{
    // 接收数据
}
``` 

#### UDP Socket
##### UDP Socket Server
``` CSharp
// 监听端口
int listenPort;

// 启动 UDP Server 监听客户端连接
UdpServerHelper _udpServerHelper = = new UdpServerHelper(listenPort);
_udpServerHelper.OnDataReceived += UdpServerHelper_OnDataReceived;
_udpServerHelper.Start();

// 客户端地址
IPEndPoint client;
// 发送数据
string data；

// 发送消息给指定客户端
_udpServerHelper.SendDataToClient(client, data);

// 发送消息给所有客户端
_udpServerHelper.SendDataToAll(data);

/// <summary>
/// 消息回调
/// </summary>
/// <param name="arg1"></param>
/// <param name="arg2"></param>
/// <exception cref="NotImplementedException"></exception>
private static void UdpServerHelper_OnDataReceived(IPEndPoint arg1, string arg2)
{
    // 接收数据
}
``` 

##### UDP Socket Client
``` CSharp
// IP
string ip;
// 端口号
int port;

// 启动 UDP Client 连接到 Server
UdpClientHelper _udpClientHelper = new UdpClientHelper();
_udpClientHelper.OnDataReceived += UdpClientHelper_OnDataReceived;
_udpClientHelper.Connect(ip, port);

// 发送数据
string data;

// 发送消息给服务端
_udpClientHelper.SendData(data);

/// <summary>
/// 客户端监听服务端连接
/// </summary>
/// <param name="obj"></param>
private static void UdpClientHelper_OnDataReceived(string obj)
{
    // 接收数据
}
``` 