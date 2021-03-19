---
title: 内网穿透 UDP 打洞
date: 2021-01-26 16:08:10
tags: [c#,udp,hole]
categories: C#.Net
---
### Nat 穿透（UDP 打洞）
<!-- more -->
### 简介
最开始源于一次去用户现场出差，[TeamViewer](https://www.teamviewer.com/en/) 与 [向日葵](https://sunlogin.oray.com/) 都是开通了会员的，但还是受到各种稀奇古怪的问题阻挠无法远程到家里的电脑和公司的电脑，回来后尝试做一个内网穿透工具以 [Remote Desktop Connection(RDP)](https://support.microsoft.com/en-us/windows/how-to-use-remote-desktop-5fe128d5-8fb1-7a23-3b8a-41e636865e8c) 方式连接作为以后远程的备选方案。
但是后期测试时发现 [Remote Desktop Connection(RDP)](https://support.microsoft.com/en-us/windows/how-to-use-remote-desktop-5fe128d5-8fb1-7a23-3b8a-41e636865e8c) 协议首先通过 TCP 进行第一次通讯建立连接以及输入用户名密码，验证用户凭据后重新以 TCP 进行远程通讯，UDP 仅作为辅助，所以改为 TCP 转发方式继续开发，UDP 打洞成功，或许以后或许可以在此基础上做些其他功能。

### 代码仓库
不确定当前仓库权限是否公有
``` cmd
git clone https://github.com/iceelves/PenetrateRemote.git
git reset fd217598
```

### 网络类型
#### 检测网络 Nat 类型
使用开源项目 [NatTypeTester](https://github.com/HMBSbige/NatTypeTester) 可检测本 地网络 Nat 类型。

#### Nat 类型
<img src="https://sadness96.github.io/images/blog/csharp-UDPHole/NatXmind.png"/>

1. 全锥型(Full Cone)
    如果在 NAT 网关已经建立了一个 NAT 映射，那么任何外网的机器都可以通过这个映射来访问内网的电脑。
1. 受限锥型(Restricted Cone)(IP受限锥型)
    如果在 NAT 网关已经建立了一个 NAT 映射，那么只有与其建立映射的 IP 才能通过 NAT 访问内网的电脑。
1. 端口受限锥型(Port Restricted Cone)(IP + PORT受限锥型)
    在 IP 限制型锥型的基础上，再进行端口的限制。
1. 对称型(Symmetric)
    即对 IP 和端口都有限制，只有和其建立连接的 IP 和端口向其发送数据才不会被丢弃。

#### 不同 Nat 类型穿透性
盗的图，出处见水印
<img src="https://sadness96.github.io/images/blog/csharp-UDPHole/NatPierceThrough.webp"/>

### 运行流程
如图所示：局域网 NatA 中的机器 192.168.5.13(以下简称 ClientA) 请求与局域网 NatB 中的机器 192.168.1.100(以下简称 ClientB) 通过 UDP 打洞通信。
<img src="https://sadness96.github.io/images/blog/csharp-UDPHole/NatVisio.png"/>

1. 局域网内两台电脑运行 Client 程序，向服务器建立 Socket UDP 连接。
1. ClientA 向服务端发出请求与 ClientB 建立连接。
1. 服务端验证两台电脑在线后将 ClientB Socket UDP 信息发送给 ClientA。
1. ClientA 发送登录信息到 ClientB，此时 ClientB 接收不到消息。
1. ClientA 向服务端发送反向打洞请求，让 ClientB 向自己发送数据。
1. 服务端将 ClientA 信息发送到 ClientB。
1. ClientB 向 ClientA 发送登录信息后，双方建立打洞通信，关闭服务端程序后依然可以通信。

### 核心代码
#### Server
``` csharp 
public class Udp_Server : IDisposable
{
    /// <summary>
    /// 构造函数
    /// </summary>
    public Udp_Server(int ListeningPort)
    {
        _listeningPort = ListeningPort;
        _remotePoint = new IPEndPoint(IPAddress.Any, 0);
        _serverThread = new Thread(Run);
    }

    /// <summary>
    /// 析构函数
    /// </summary>
    ~Udp_Server()
    {
        Dispose();
    }

    /// <summary>
    /// 监听端口号
    /// </summary>
    private int _listeningPort;

    /// <summary>
    /// 服务器端消息监听
    /// </summary>
    private UdpClient _server;

    /// <summary>
    /// 计时器
    /// </summary>
    private readonly Thread _serverThread;

    /// <summary>
    /// 远程用户请求的IP地址及端口 
    /// </summary>
    private IPEndPoint _remotePoint;

    private static ConcurrentDictionary<string, HoleUserInfo> _loginUser;
    /// <summary>
    /// 登陆打洞用户
    /// </summary>
    public static ConcurrentDictionary<string, HoleUserInfo> LoginUser
    {
        get
        {
            if (_loginUser == null)
            {
                _loginUser = new ConcurrentDictionary<string, HoleUserInfo>();
            }
            return _loginUser;
        }
        set => _loginUser = value;
    }

    /// <summary>
    /// 启动 UDP 监听
    /// </summary>
    public void Start()
    {
        try
        {
            // 删除超时数据
            System.Timers.Timer overtime = new System.Timers.Timer();
            overtime.Interval = 60000 * 5;
            overtime.Elapsed += new System.Timers.ElapsedEventHandler(timer_Overtime);
            overtime.Start();
            // 启动服务
            _server = new UdpClient(_listeningPort);
            _serverThread.Start();
            NLogHelper.SaveInfo($"服务启动，监听端口：{_listeningPort}");
        }
        catch (Exception ex)
        {
            NLogHelper.SaveError(ex.ToString());
        }
    }

    /// <summary>
    /// 移除超时数据线程
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void timer_Overtime(object sender, System.Timers.ElapsedEventArgs e)
    {
        List<string> removeKeys = LoginUser.Where(o => DateTime.Now - o.Value.LastLoginTime >= new TimeSpan(0, 5, 0)).Select(o => o.Key).ToList();
        if (removeKeys != null && removeKeys.Count >= 1)
        {
            foreach (var item in removeKeys)
            {
                LoginUser.TryRemove(item, out _);
            }
        }
    }

    /// <summary>
    /// 线程主方法
    /// </summary>
    private void Run()
    {
        byte[] msgBuffer;
        while (true)
        {
            try
            {
                // 接受消息
                msgBuffer = _server.Receive(ref _remotePoint);
                // 将消息转换为对象 
                var msgObject = ObjectSerializerHelper.JsonDeserialize(msgBuffer);
                if (msgObject == null)
                {
                    continue;
                }
                // 解析消息
                if (msgObject.ModelName == typeof(MessageLogin).Name)
                {
                    // 用户登陆/心跳
                    var lginMsg = JsonConvert.DeserializeObject<MessageLogin>(msgObject.ModelData);
                    if (LoginUser.ContainsKey(lginMsg.MacAddress))
                    {
                        // 更新数据
                        var vLoginUser = LoginUser.Where(o => o.Key.Equals(lginMsg.MacAddress)).First();
                        vLoginUser.Value.UserName = lginMsg.UserName;
                        vLoginUser.Value.NetPoint = new IPEndPoint(_remotePoint.Address, _remotePoint.Port);
                        vLoginUser.Value.LastLoginTime = DateTime.Now;
                    }
                    else
                    {
                        // 新增数据
                        HoleUserInfo holeUserInfo = new HoleUserInfo
                        {
                            MacAddress = lginMsg.MacAddress,
                            UserName = lginMsg.UserName,
                            NetPoint = new IPEndPoint(_remotePoint.Address, _remotePoint.Port),
                            LastLoginTime = DateTime.Now
                        };
                        LoginUser.TryAdd(lginMsg.MacAddress, holeUserInfo);
                    }
                }
                else if (msgObject.ModelName == typeof(MessageHolePunchingRequest).Name)
                {
                    // 打洞请求
                    var holeMsg = JsonConvert.DeserializeObject<MessageHolePunchingRequest>(msgObject.ModelData);
                    var vSelectUsers = LoginUser.Where(o => o.Key.Equals(holeMsg.TargetMacAddress))?.ToList();
                    if (vSelectUsers != null && vSelectUsers.Count >= 1)
                    {
                        var vSelectUser = vSelectUsers.First();
                        byte[] buffer = ObjectSerializerHelper.JsonSerialize(new MessageHolePunching() { MacAddress = holeMsg.RequestorMacAddress, RequestorNetIP = _remotePoint.Address.ToString(), RequestorNetPort = _remotePoint.Port, IsFirst = holeMsg.IsAtoB });
                        _server.Send(buffer, buffer.Length, vSelectUser.Value.NetPoint);
                    }
                }
            }
            catch (Exception ex)
            {
                NLogHelper.SaveError(ex.ToString());
            }
        }
    }

    /// <summary>
    /// 释放资源
    /// </summary>
    public void Dispose()
    {
        try
        {
            if (_server != null)
            {
                _serverThread.Abort();
                _server.Close();
                NLogHelper.SaveInfo($"服务停止！");
            }
        }
        catch (Exception ex)
        {
            NLogHelper.SaveError(ex.ToString());
        }
    }
}
```

#### Clent
``` csharp 
public class Udp_Client : IDisposable
{
    /// <summary>
    /// 构造函数
    /// </summary>
    public Udp_Client(MessageLogin MessageLogin)
    {
        _messageLogin = MessageLogin;
        // 任何与本地连接的用户IP地址
        _remotePoint = new IPEndPoint(IPAddress.Any, 0);
        // 服务器地址
        string serverIP = ConfigurationManager.AppSettings["server_remote"];
        _hostPoint = new IPEndPoint(IPAddress.Parse(serverIP.Split(':')[0]), int.Parse(serverIP.Split(':')[1]));
        // 不指定端口,系统自动分配
        _client = new UdpClient();
        // 监听线程
        _listenThread = new Thread(Run);
    }

    /// <summary>
    /// 析构函数
    /// </summary>
    ~Udp_Client()
    {
        Dispose();
    }

    /// <summary>
    /// 客户端监听
    /// </summary>
    private readonly UdpClient _client;

    /// <summary>
    /// 主机IP 
    /// </summary>
    private readonly IPEndPoint _hostPoint;

    /// <summary>
    /// 接收任何远程机器的数据
    /// </summary>
    private IPEndPoint _remotePoint;

    /// <summary>
    /// 监听线程 
    /// </summary>
    private readonly Thread _listenThread;

    /// <summary>
    /// 登陆信息
    /// </summary>
    private MessageLogin _messageLogin { get; set; }

    /// <summary> 
    /// 启动客户端 
    /// </summary> 
    public void Start()
    {
        // 登陆打洞用户
        byte[] buffer = ObjectSerializerHelper.JsonSerialize(_messageLogin);
        _client.Send(buffer, buffer.Length, _hostPoint);
        // 启动心跳线程
        DispatcherTimer timerHeartbeat = new DispatcherTimer();
        timerHeartbeat.Interval = new TimeSpan(TimeSpan.TicksPerMinute);
        timerHeartbeat.Tick += TimerHeartbeat_Tick;
        timerHeartbeat.Start();
        // 启动主线程
        if (_listenThread.ThreadState == ThreadState.Unstarted)
        {
            _listenThread.Start();
        }
    }

    /// <summary>
    /// 打洞请求
    /// </summary>
    /// <param name="MacAddress">目标 Mac 地址</param>
    public void HolePunching(string MacAddress)
    {
        byte[] buffer = ObjectSerializerHelper.JsonSerialize(new MessageHolePunchingRequest() { RequestorMacAddress = _messageLogin.MacAddress, TargetMacAddress = MacAddress, IsAtoB = true });
        _client.Send(buffer, buffer.Length, _hostPoint);
    }

    /// <summary>
    /// 心跳计时器
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void TimerHeartbeat_Tick(object sender, EventArgs e)
    {
        byte[] buffer = ObjectSerializerHelper.JsonSerialize(_messageLogin);
        _client.Send(buffer, buffer.Length, _hostPoint);
    }

    /// <summary>
    /// 线程主方法
    /// </summary>
    private void Run()
    {
        byte[] msgBuffer;
        while (true)
        {
            try
            {
                // 接受消息
                msgBuffer = _client.Receive(ref _remotePoint);
                // 将消息转换为对象 
                var msgObject = ObjectSerializerHelper.JsonDeserialize(msgBuffer);
                if (msgObject == null)
                {
                    continue;
                }
                // 解析消息
                if (msgObject.ModelName == typeof(MessageHolePunching).Name)
                {
                    // 接受打洞请求
                    var holeMsg = JsonConvert.DeserializeObject<MessageHolePunching>(msgObject.ModelData);
                    byte[] bufferUserLogin = ObjectSerializerHelper.JsonSerialize(_messageLogin);
                    _client.Send(bufferUserLogin, bufferUserLogin.Length, new IPEndPoint(IPAddress.Parse(holeMsg.RequestorNetIP), holeMsg.RequestorNetPort));
                    // 反向打洞
                    if (holeMsg.IsFirst)
                    {
                        byte[] bufferHolePunchingRequest = ObjectSerializerHelper.JsonSerialize(new MessageHolePunchingRequest() { RequestorMacAddress = _messageLogin.MacAddress, TargetMacAddress = holeMsg.MacAddress, IsAtoB = false });
                        _client.Send(bufferHolePunchingRequest, bufferHolePunchingRequest.Length, _hostPoint);
                    }
                }
                else if (msgObject.ModelName == typeof(MessageLogin).Name)
                {
                    // 打洞连接成功,接收到对方登陆信息
                    var lginMsg = JsonConvert.DeserializeObject<MessageLogin>(msgObject.ModelData);
                    NLogHelper.SaveInfo($"P2P打洞成功：{JsonConvert.SerializeObject(lginMsg)}");
                }
            }
            catch (Exception ex)
            {
                NLogHelper.SaveError(ex.ToString());
            }
        }
    }

    /// <summary>
    /// 释放资源
    /// </summary>
    public void Dispose()
    {
        try
        {
            if (_client != null)
            {
                _listenThread.Abort();
                _client.Close();
                NLogHelper.SaveInfo($"客户端停止！");
            }
        }
        catch (Exception ex)
        {
            NLogHelper.SaveError(ex.ToString());
        }
    }
}
```