---
title: WebSocket 帮助类
date: 2023-10-10 17:00:00
tags: [c#,websocket]
categories: C#.Net
---
### WebSocket 帮助类使用介绍
<!-- more -->
### 简介
WebSocket 是一种在 Web 应用程序中实现双向通信的通信协议。
传统的 Web 通信方式是基于 HTTP（超文本传输协议）的请求-响应模式，即客户端发送请求，服务器返回响应。在这种模式下，服务器通常不能主动向客户端发送数据，而是需要客户端定期发送请求来获取最新数据。这对于实时性要求较高的应用场景（如聊天室、实时数据更新）并不是很高效。
WebSocket 解决了这个问题。WebSocket 建立在 HTTP 协议之上，但它是一种全双工通信协议，使得服务器和客户端可以在同一个持久连接上实时地交换数据，而不需要额外的请求和响应。
WebSocket 是基于 TCP（传输控制协议）作为传输层协议，通常分为服务端与客户端。

### 核心代码
#### WebSocket Server
[WebSocketServerHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Queue.Helper/Socket/WebSocketServerHelper.cs)

``` CSharp
/// <summary>
/// WebSocket 服务端帮助类
/// </summary>
public class WebSocketServerHelper
{
    private HttpListener listener;
    private CancellationTokenSource cancellationTokenSource;
    private List<WebSocket> webSockets;

    /// <summary>
    /// 收到数据回调
    /// </summary>
    public event Action<WebSocket, string> OnDataReceived;

    /// <summary>
    /// 启动
    /// 监听所有 IP 上的端口需要以管理员权限运行
    /// </summary>
    /// <param name="port"></param>
    /// <returns></returns>
    public async Task Start(int port)
    {
        Console.WriteLine($"WebSocket Server listen port:{port}");

        listener = new HttpListener();
        listener.Prefixes.Add($"http://*:{port}/");
        listener.Start();

        cancellationTokenSource = new CancellationTokenSource();
        webSockets = new List<WebSocket>();

        while (!cancellationTokenSource.IsCancellationRequested)
        {
            HttpListenerContext context = await listener.GetContextAsync();

            if (context.Request.IsWebSocketRequest)
            {
                ProcessWebSocketRequest(context);
            }
            else
            {
                context.Response.StatusCode = 400;
                context.Response.Close();
            }
        }
    }

    /// <summary>
    /// 停止
    /// </summary>
    public void Stop()
    {
        cancellationTokenSource.Cancel();
        listener.Stop();
    }

    /// <summary>
    /// 处理 WebSocket 请求
    /// </summary>
    /// <param name="context"></param>
    private async void ProcessWebSocketRequest(HttpListenerContext context)
    {
        WebSocketContext webSocketContext = await context.AcceptWebSocketAsync(subProtocol: null);
        WebSocket webSocket = webSocketContext.WebSocket;
        webSockets.Add(webSocket);
        Console.WriteLine($"Add client connection:{context.Request.RemoteEndPoint}");

        try
        {
            while (webSocket.State == WebSocketState.Open)
            {
                byte[] buffer = new byte[1024];
                WebSocketReceiveResult result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationTokenSource.Token);

                if (result.MessageType == WebSocketMessageType.Text)
                {
                    string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    OnDataReceived.Invoke(webSocket, message);
                }
                else if (result.MessageType == WebSocketMessageType.Close)
                {
                    webSockets.Remove(webSocket);
                    Console.WriteLine($"Break client connection:{context.Request.RemoteEndPoint}");
                    await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", cancellationTokenSource.Token);
                }
            }
        }
        catch (Exception ex)
        {
            webSockets.Remove(webSocket);
            Console.WriteLine($"Break client connection:{context.Request.RemoteEndPoint}");
        }
    }

    /// <summary>
    /// 发送消息给所有客户端
    /// </summary>
    /// <param name="data">消息</param>
    /// <returns></returns>
    public async Task SendToAllAsync(string data)
    {
        List<Task> tasks = new List<Task>();

        if (webSockets != null && webSockets.Count >= 1)
        {
            foreach (WebSocket webSocket in webSockets)
            {
                tasks.Add(webSocket.SendAsync(new ArraySegment<byte>(Encoding.UTF8.GetBytes(data)), WebSocketMessageType.Text, true, cancellationTokenSource.Token));
            }

            await Task.WhenAll(tasks);
        }
    }

    /// <summary>
    /// 发送消息给指定客户端
    /// </summary>
    /// <param name="client">客户端</param>
    /// <param name="data">消息</param>
    /// <returns></returns>
    public async Task SendToClientAsync(WebSocket client, string data)
    {
        try
        {
            await client.SendAsync(new ArraySegment<byte>(Encoding.UTF8.GetBytes(data)), WebSocketMessageType.Text, true, cancellationTokenSource.Token);
        }
        catch (Exception)
        {
            // 客户端已断开连接
        }
    }
}
``` 

#### WebSocket Client
[WebSocketClientHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Queue.Helper/Socket/WebSocketClientHelper.cs)

``` CSharp
/// <summary>
/// WebSocket 客户端帮助类
/// </summary>
public class WebSocketClientHelper
{
    private ClientWebSocket clientWebSocket;
    private CancellationTokenSource cancellationTokenSource;

    /// <summary>
    /// 收到数据回调
    /// </summary>
    public event Action<string> OnDataReceived;

    /// <summary>
    /// 连接服务端
    /// </summary>
    /// <param name="url">ws://localhost:8080/</param>
    /// <returns></returns>
    public async Task Connect(string url)
    {
        clientWebSocket = new ClientWebSocket();
        cancellationTokenSource = new CancellationTokenSource();

        await clientWebSocket.ConnectAsync(new Uri(url), cancellationTokenSource.Token);

        await Task.Factory.StartNew(async () =>
        {
            while (clientWebSocket.State == WebSocketState.Open)
            {
                byte[] buffer = new byte[1024];
                WebSocketReceiveResult result = await clientWebSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationTokenSource.Token);

                if (result.MessageType == WebSocketMessageType.Text)
                {
                    string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    OnDataReceived?.Invoke(message);
                }
                else if (result.MessageType == WebSocketMessageType.Close)
                {
                    await clientWebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", cancellationTokenSource.Token);
                }
            }
        }, cancellationTokenSource.Token);
    }

    /// <summary>
    /// 发送消息
    /// </summary>
    /// <param name="message">消息</param>
    /// <returns></returns>
    public async Task Send(string message)
    {
        byte[] buffer = Encoding.UTF8.GetBytes(message);
        await clientWebSocket.SendAsync(new ArraySegment<byte>(buffer), WebSocketMessageType.Text, true, cancellationTokenSource.Token);
    }

    /// <summary>
    /// 关闭
    /// </summary>
    /// <returns></returns>
    public async Task Close()
    {
        cancellationTokenSource.Cancel();
        await clientWebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", cancellationTokenSource.Token);
    }
}
``` 

### 调用测试
#### WebSocket Server
``` CSharp
// 监听端口
int listenPort;

// 启动 WebSocket Server 监听客户端连接
WebSocketServerHelper _webSocketServerHelper = new WebSocketServerHelper();
_webSocketServerHelper.OnDataReceived += WebSocketServer_OnDataReceived;
Task.Run(async () =>
{
    await _webSocketServerHelper.Start(listenPort);
    Console.WriteLine($"启动 WebSocket Server 监听：{listenPort}");
});

// 客户端
WebSocket client;
// 发送数据
string data;

Task.Run(async () =>
{
    // 发送消息给指定客户端
    await _webSocketServerHelper.SendToClientAsync(client, data);

    // 发送消息给所有客户端
    await _webSocketServerHelper.SendToAllAsync(data);
});

/// <summary>
/// 消息回调
/// </summary>
/// <param name="arg1"></param>
/// <param name="arg2"></param>
/// <exception cref="NotImplementedException"></exception>
private static void WebSocketServer_OnDataReceived(System.Net.WebSockets.WebSocket arg1, string arg2)
{
    // 接收数据
}
``` 

#### WebSocket Client
``` CSharp
// IP
string ip;
// 端口号
int port;
// WebSocket URL
string url = $"ws://{ip}:{port}/";

// 启动 WebSocket Client 连接到 Server
WebSocketClientHelper _websocketClient = new WebSocketClientHelper();
_websocketClient.Connect(url);
_websocketClient.OnDataReceived += WebsocketClient_OnDataReceived;

// 发送数据
string message;

// 发送消息给服务端
_websocketClient.Send(message);

/// <summary>
/// 消息回调
/// </summary>
/// <param name="obj"></param>
/// <exception cref="NotImplementedException"></exception>
private void WebsocketClient_OnDataReceived(string obj)
{
    // 接收数据
}
``` 