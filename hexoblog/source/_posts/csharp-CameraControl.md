---
title: 监控相机控制
date: 2023-12-09 12:45:00
tags: [c#]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-CameraControl/CameraControlTest.jpg"/>

<!-- more -->
### 简介
与主流监控相机厂商（[海康](https://www.hikvision.com/cn/)、[大华](https://www.dahuatech.com/)、[宇视](https://cn.uniview.com/)） SDK 和开放行业接口 [Onvif](https://www.onvif.org/) 对接，实现画面预览、PTZ 云台控制等操作，二次封装后提供引用库、WebAPI 接口、Socket 等多种方式调用控制。

### 核心代码
注意核心代码并不完整，仅做参考
#### 接口
``` csharp
/// <summary>
/// 相机控制接口
/// </summary>
public interface ICameraControl
{
    /// <summary>
    /// 登录
    /// </summary>
    /// <param name="channels">通道号集合</param>
    /// <returns>是否登录成功</returns>
    bool Login(out List<CameraChannel> channels);

    /// <summary>
    /// 退出登录
    /// </summary>
    /// <returns>是否退出成功</returns>
    bool Logout();

    /// <summary>
    /// 获取流 RTSP 地址
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="streamUris">流地址集合</param>
    /// <returns>是否获取成功</returns>
    bool GetStreamUri(int channel, out List<string> streamUris);

    /// <summary>
    /// 实时预览
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="hPlayWnd">渲染句柄</param>
    /// <returns>是否成功</returns>
    bool RealPlay(int channel, IntPtr hPlayWnd);

    /// <summary>
    /// 停止实时预览
    /// </summary>
    /// <returns>是否成功</returns>
    bool StopRealPlay();

    /// <summary>
    /// 云台控制
    /// 方向、变焦、变倍、光圈
    /// 灯光、雨刷、风扇、加热、除雪
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzCommand">控制命令</param>
    /// <param name="isStart">是否启动</param>
    /// <param name="speed">速度</param>
    /// <returns>是否调用成功</returns>
    bool PtzControl(int channel, PTZCommand ptzCommand, bool isStart, int speed = 1);

    /// <summary>
    /// 获取云台 PTZ 值
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzValue">PTZ 参数</param>
    /// <returns>是否调用成功</returns>
    bool GetPTZValue(int channel, out PTZValue ptzValue);

    /// <summary>
    /// 设置云台 PTZ 值
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzValue">PTZ 参数</param>
    /// <returns>是否设置成功</returns>
    bool SetPTZValue(int channel, PTZValue ptzValue);

    /// <summary>
    /// 获取云台预置位列表
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzPresets">预置位列表</param>
    /// <returns>是否调用成功</returns>
    bool GetPTZPreset(int channel, out List<PTZPreset> ptzPresets);

    /// <summary>
    /// 云台预置位控制
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzCommand">控制命令</param>
    /// <param name="presetIndex">预置位序号</param>
    /// <param name="presetName">预置位名称，设置预置位时使用</param>
    /// <returns>是否调用成功</returns>
    bool PTZPresetControl(int channel, PTZCommand ptzCommand, int presetIndex, string presetName);
}
```

#### 海康
使用 [官方 SDK](https://open.hikvision.com/download/5cda567cf47ae80dd41a54b3?type=10) 对接。

``` csharp
/// <summary>
/// 海康相机控制
/// </summary>
public class HikvisionControl : ICameraControl
{
    // 用户 ID 值
    private Int32 m_lUserID = -1;
    // 预览句柄
    private Int32 m_lRealHandle = -1;

    public CHCNetSDK.NET_DVR_USER_LOGIN_INFO struLogInfo;
    public CHCNetSDK.NET_DVR_DEVICEINFO_V40 DeviceInfo;

    public CHCNetSDK.NET_DVR_PRESET_NAME[] m_struPreSetCfg = new CHCNetSDK.NET_DVR_PRESET_NAME[300];

    /// <summary>
    /// 海康相机控制构造函数
    /// </summary>
    /// <param name="ip">IP</param>
    /// <param name="userName">用户名</param>
    /// <param name="password">密码</param>
    /// <param name="port">端口</param>
    public HikvisionControl(string ip, string userName, string password, string port = "8000")
    {
        struLogInfo = new CHCNetSDK.NET_DVR_USER_LOGIN_INFO();

        // 设备IP地址或者域名
        byte[] byIP = Encoding.Default.GetBytes(ip);
        struLogInfo.sDeviceAddress = new byte[129];
        byIP.CopyTo(struLogInfo.sDeviceAddress, 0);

        // 设备用户名
        byte[] byUserName = Encoding.Default.GetBytes(userName);
        struLogInfo.sUserName = new byte[64];
        byUserName.CopyTo(struLogInfo.sUserName, 0);

        // 设备密码
        byte[] byPassword = Encoding.Default.GetBytes(password);
        struLogInfo.sPassword = new byte[64];
        byPassword.CopyTo(struLogInfo.sPassword, 0);

        // 设备服务端口号
        struLogInfo.wPort = ushort.Parse(port);

        // 是否异步登录：0- 否，1- 是
        struLogInfo.bUseAsynLogin = false;

        // 初始化
        CHCNetSDK.NET_DVR_Init();

        // 保存SDK日志 To save the SDK log
        //CHCNetSDK.NET_DVR_SetLogToFile(3, "C:\\SdkLog\\", true);
    }

    /// <summary>
    /// 登录
    /// </summary>
    /// <param name="channels">通道号集合</param>
    /// <returns>是否登录成功</returns>
    public bool Login(out List<CameraChannel> channels)
    {
        channels = new List<CameraChannel>();

        DeviceInfo = new CHCNetSDK.NET_DVR_DEVICEINFO_V40();
        // 登录设备 Login the device
        m_lUserID = CHCNetSDK.NET_DVR_Login_V40(ref struLogInfo, ref DeviceInfo);
        if (m_lUserID < 0)
        {
            var iLastErr = CHCNetSDK.NET_DVR_GetLastError();
            return false;
        }
        else
        {
            // 模拟通道号
            var vChanNum = DeviceInfo.struDeviceV30.byChanNum;
            // 模拟通道起始通道号
            var vStartChan = DeviceInfo.struDeviceV30.byStartChan;
            // 添加模拟通道号
            for (int i = 0; i < vChanNum; i++)
            {
                channels.Add(new CameraChannel()
                {
                    ChannelID = vStartChan + i,
                    ChannelName = $"Channel {vStartChan + i}"
                });
            }
            return true;
        }
    }

    /// <summary>
    /// 退出登录
    /// </summary>
    /// <returns>是否退出成功</returns>
    public bool Logout()
    {
        // 注销用户与释放 SDK 资源
        return CHCNetSDK.NET_DVR_Logout(m_lUserID) && CHCNetSDK.NET_DVR_Cleanup();
    }

    /// <summary>
    /// 获取流 RTSP 地址
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="streamUris">流地址集合</param>
    /// <returns>是否获取成功</returns>
    public bool GetStreamUri(int channel, out List<string> streamUris)
    {
        streamUris = new List<string>();
        return false;
    }

    /// <summary>
    /// 实时预览
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="hPlayWnd">渲染句柄</param>
    /// <returns>是否成功</returns>
    public bool RealPlay(int channel, IntPtr hPlayWnd)
    {
        if (m_lRealHandle >= 0)
        {
            // 已经开启预览
            return false;
        }

        CHCNetSDK.NET_DVR_PREVIEWINFO lpPreviewInfo = new CHCNetSDK.NET_DVR_PREVIEWINFO();
        lpPreviewInfo.hPlayWnd = hPlayWnd;//预览窗口
        lpPreviewInfo.lChannel = channel;//预览的设备通道
        lpPreviewInfo.dwStreamType = 0;//码流类型：0-主码流，1-子码流，2-码流3，3-码流4，以此类推
        lpPreviewInfo.dwLinkMode = 0;//连接方式：0- TCP方式，1- UDP方式，2- 多播方式，3- RTP方式，4-RTP/RTSP，5-RSTP/HTTP 
        lpPreviewInfo.bBlocked = true; //0- 非阻塞取流，1- 阻塞取流
        lpPreviewInfo.dwDisplayBufNum = 1; //播放库播放缓冲区最大缓冲帧数
        lpPreviewInfo.byProtoType = 0;
        lpPreviewInfo.byPreviewMode = 0;

        IntPtr pUser = new IntPtr();//用户数据

        // 打开预览 Start live view 
        m_lRealHandle = CHCNetSDK.NET_DVR_RealPlay_V40(m_lUserID, ref lpPreviewInfo, null/*RealData*/, pUser);
        if (m_lRealHandle < 0)
        {
            var iLastErr = $"NET_DVR_RealPlay_V40 failed, error code= {CHCNetSDK.NET_DVR_GetLastError()}";
            return false;
        }
        else
        {
            return true;
        }
    }

    /// <summary>
    /// 停止实时预览
    /// </summary>
    /// <returns>是否成功</returns>
    public bool StopRealPlay()
    {
        if (m_lRealHandle < 0)
        {
            // 未开启预览
            return false;
        }

        if (!CHCNetSDK.NET_DVR_StopRealPlay(m_lRealHandle))
        {
            var iLastErr = $"NET_DVR_StopRealPlay failed, error code= {CHCNetSDK.NET_DVR_GetLastError()}";
            return false;
        }
        m_lRealHandle = -1;
        return true;
    }

    /// <summary>
    /// 云台控制
    /// 方向、变焦、变倍、光圈
    /// 灯光、雨刷、风扇、加热、除雪
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzCommand">控制命令</param>
    /// <param name="isStart">是否启动</param>
    /// <param name="speed">速度</param>
    /// <returns>是否调用成功</returns>
    public bool PtzControl(int channel, PTZCommand ptzCommand, bool isStart, int speed)
    {
        uint dwPTZCommand = 0;
        switch (ptzCommand)
        {
            case PTZCommand.Light: dwPTZCommand = CHCNetSDK.LIGHT_PWRON; break;
            case PTZCommand.Wiper: dwPTZCommand = CHCNetSDK.WIPER_PWRON; break;
            case PTZCommand.Fan: dwPTZCommand = CHCNetSDK.FAN_PWRON; break;
            case PTZCommand.Heater: dwPTZCommand = CHCNetSDK.HEATER_PWRON; break;
            case PTZCommand.SnowRemoval: dwPTZCommand = 0; break;
            case PTZCommand.Aux1: dwPTZCommand = CHCNetSDK.AUX_PWRON1; break;
            case PTZCommand.Aux2: dwPTZCommand = CHCNetSDK.AUX_PWRON2; break;
            case PTZCommand.SetPreset: dwPTZCommand = CHCNetSDK.SET_PRESET; break;
            case PTZCommand.ClePreset: dwPTZCommand = CHCNetSDK.CLE_PRESET; break;
            case PTZCommand.GotoPreset: dwPTZCommand = CHCNetSDK.GOTO_PRESET; break;

            case PTZCommand.ZoomIn: dwPTZCommand = CHCNetSDK.ZOOM_IN; break;
            case PTZCommand.ZoomOut: dwPTZCommand = CHCNetSDK.ZOOM_OUT; break;
            case PTZCommand.FocusNear: dwPTZCommand = CHCNetSDK.FOCUS_NEAR; break;
            case PTZCommand.FocusFar: dwPTZCommand = CHCNetSDK.FOCUS_FAR; break;
            case PTZCommand.IrisOpen: dwPTZCommand = CHCNetSDK.IRIS_OPEN; break;
            case PTZCommand.IrisClose: dwPTZCommand = CHCNetSDK.IRIS_CLOSE; break;

            case PTZCommand.TiltUp: dwPTZCommand = CHCNetSDK.TILT_UP; break;
            case PTZCommand.TiltDown: dwPTZCommand = CHCNetSDK.TILT_DOWN; break;
            case PTZCommand.PanLeft: dwPTZCommand = CHCNetSDK.PAN_LEFT; break;
            case PTZCommand.PanRight: dwPTZCommand = CHCNetSDK.PAN_RIGHT; break;
            case PTZCommand.UpLeft: dwPTZCommand = CHCNetSDK.UP_LEFT; break;
            case PTZCommand.UpRight: dwPTZCommand = CHCNetSDK.UP_RIGHT; break;
            case PTZCommand.DownLeft: dwPTZCommand = CHCNetSDK.DOWN_LEFT; break;
            case PTZCommand.DownRight: dwPTZCommand = CHCNetSDK.DOWN_RIGHT; break;
            case PTZCommand.PanAuto: dwPTZCommand = CHCNetSDK.PAN_AUTO; break;
        }

        return CHCNetSDK.NET_DVR_PTZControlWithSpeed_Other(m_lUserID, channel, dwPTZCommand, isStart ? (uint)0 : 1, (uint)speed);
    }

    /// <summary>
    /// 获取云台 PTZ 值
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzValue">PTZ 参数</param>
    /// <returns>是否调用成功</returns>
    public bool GetPTZValue(int channel, out PTZValue ptzValue)
    {
        CHCNetSDK.NET_DVR_PTZPOS m_struPtzCfg = new CHCNetSDK.NET_DVR_PTZPOS();

        uint dwReturn = 0;
        int nSize = Marshal.SizeOf(m_struPtzCfg);
        IntPtr ptrPtzCfg = Marshal.AllocHGlobal(nSize);
        Marshal.StructureToPtr(m_struPtzCfg, ptrPtzCfg, false);

        if (!CHCNetSDK.NET_DVR_GetDVRConfig(m_lUserID, CHCNetSDK.NET_DVR_GET_PTZPOS, channel, ptrPtzCfg, (uint)nSize, ref dwReturn))
        {
            // 获取参数失败
            var iLastErr = "NET_DVR_GetDVRConfig failed, error code:" + CHCNetSDK.NET_DVR_GetLastError();
            ptzValue = new PTZValue();
            return false;
        }
        else
        {
            m_struPtzCfg = (CHCNetSDK.NET_DVR_PTZPOS)Marshal.PtrToStructure(ptrPtzCfg, typeof(CHCNetSDK.NET_DVR_PTZPOS));
            // 成功获取显示 ptz 参数
            ptzValue = new PTZValue()
            {
                P = (float)(Convert.ToUInt16(Convert.ToString(m_struPtzCfg.wPanPos, 16)) * 0.1m),
                T = (float)(Convert.ToUInt16(Convert.ToString(m_struPtzCfg.wTiltPos, 16)) * 0.1m),
                Z = (float)(Convert.ToUInt16(Convert.ToString(m_struPtzCfg.wZoomPos, 16)) * 0.1m)
            };
            return true;
        }
    }

    /// <summary>
    /// 设置云台 PTZ 值
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzValue">PTZ 参数</param>
    /// <returns>是否设置成功</returns>
    public bool SetPTZValue(int channel, PTZValue ptzValue)
    {
        // 海康设置云台时 PTZ 参数值小数超过 1 位为会报错
        ptzValue.P = (float)Math.Round(ptzValue.P, 1);
        ptzValue.T = (float)Math.Round(ptzValue.T, 1);
        ptzValue.Z = (float)Math.Round(ptzValue.Z, 1);

        CHCNetSDK.NET_DVR_PTZPOS m_struPtzCfg = new CHCNetSDK.NET_DVR_PTZPOS();

        // 操作类型，仅在设置时有效。1-定位PTZ参数，2-定位P参数，3-定位T参数，4-定位Z参数，5-定位PT参数 
        m_struPtzCfg.wAction = 1;
        // 实际显示的PTZ值是获取到的十六进制值的十分之一，如获取的水平参数P的值是0x1750，实际显示的P值为175度；获取到的垂直参数T的值是0x0789，实际显示的T值为78.9度；获取到的变倍参数Z的值是0x1100，实际显示的Z值为110倍。
        m_struPtzCfg.wPanPos = Convert.ToUInt16(Convert.ToString(ptzValue.P * 10), 16);
        m_struPtzCfg.wTiltPos = Convert.ToUInt16(Convert.ToString(ptzValue.T * 10), 16);
        m_struPtzCfg.wZoomPos = Convert.ToUInt16(Convert.ToString(ptzValue.Z * 10), 16);

        int nSize = Marshal.SizeOf(m_struPtzCfg);
        IntPtr ptrPtzCfg = Marshal.AllocHGlobal(nSize);
        Marshal.StructureToPtr(m_struPtzCfg, ptrPtzCfg, false);

        if (!CHCNetSDK.NET_DVR_SetDVRConfig(m_lUserID, CHCNetSDK.NET_DVR_SET_PTZPOS, channel, ptrPtzCfg, (uint)nSize))
        {
            var iLastErr = "NET_DVR_SetDVRConfig failed, error code:" + CHCNetSDK.NET_DVR_GetLastError();
            return false;
        }
        else
        {
            return true;
        }
    }

    /// <summary>
    /// 获取云台预置位列表
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzPresets">预置位列表</param>
    /// <returns>是否调用成功</returns>
    public bool GetPTZPreset(int channel, out List<PTZPreset> ptzPresets)
    {
        uint dwReturn = 0;
        int nSize = Marshal.SizeOf(m_struPreSetCfg[0]);
        int nOutBufSize = nSize * 300;
        IntPtr ptrPreSetCfg = Marshal.AllocHGlobal(nOutBufSize);

        for (int i = 0; i < 300; i++)
        {
            Marshal.StructureToPtr(m_struPreSetCfg[i], ptrPreSetCfg + (i * nSize), false);
        }

        if (!CHCNetSDK.NET_DVR_GetDVRConfig(m_lUserID, CHCNetSDK.NET_DVR_GET_PRESET_NAME, channel, ptrPreSetCfg, (uint)nOutBufSize, ref dwReturn))
        {
            // 获取参数失败
            var iLastErr = "NET_DVR_GetDVRConfig failed, error code:" + CHCNetSDK.NET_DVR_GetLastError();
            ptzPresets = new List<PTZPreset>();
            return false;
        }
        else
        {
            ptzPresets = new List<PTZPreset>();
            for (int i = 0; i < 300; i++)
            {
                m_struPreSetCfg[i] = (CHCNetSDK.NET_DVR_PRESET_NAME)Marshal.PtrToStructure(ptrPreSetCfg + (i * nSize), typeof(CHCNetSDK.NET_DVR_PRESET_NAME));

                // 添加数据
                ptzPresets.Add(new PTZPreset()
                {
                    ID = m_struPreSetCfg[i].wPresetNum,
                    Name = HikvisionEncoding.GetGBKString(m_struPreSetCfg[i].byName)
                });
            }
            return true;
        }
    }

    /// <summary>
    /// 云台预置位控制
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzCommand">控制命令</param>
    /// <param name="presetIndex">预置位序号</param>
    /// <param name="presetName">预置位名称，设置预置位时使用</param>
    /// <returns>是否调用成功</returns>
    public bool PTZPresetControl(int channel, PTZCommand ptzCommand, int presetIndex, string presetName)
    {
        uint dwPTZCommand = 0;
        switch (ptzCommand)
        {
            case PTZCommand.SetPreset:
                dwPTZCommand = CHCNetSDK.SET_PRESET;
                // 设置预置位前先修改预置位名称
                CHCNetSDK.NET_DVR_PRESET_NAME struPreSetCfg = m_struPreSetCfg[presetIndex - 1];
                struPreSetCfg.byRes = new byte[58];
                struPreSetCfg.byRes1 = new byte[2];
                int nSize = Marshal.SizeOf(struPreSetCfg);
                IntPtr ptrPreSetCfg = Marshal.AllocHGlobal(nSize);
                struPreSetCfg.dwSize = (uint)nSize;
                byte[] byPresetName;
                HikvisionEncoding.GetGBKBuffer(presetName, 32, out byPresetName);
                struPreSetCfg.byName = new byte[32];
                byPresetName.CopyTo(struPreSetCfg.byName, 0);
                Marshal.StructureToPtr(struPreSetCfg, ptrPreSetCfg, false);
                CHCNetSDK.NET_DVR_SetDVRConfig(m_lUserID, CHCNetSDK.NET_DVR_SET_PRESET_NAME, channel, ptrPreSetCfg, (uint)nSize);
                Marshal.FreeHGlobal(ptrPreSetCfg);
                break;
            case PTZCommand.ClePreset: dwPTZCommand = CHCNetSDK.CLE_PRESET; break;
            case PTZCommand.GotoPreset: dwPTZCommand = CHCNetSDK.GOTO_PRESET; break;
        }

        return CHCNetSDK.NET_DVR_PTZPreset_Other(m_lUserID, channel, dwPTZCommand, (uint)presetIndex);
    }
}
```

#### 宇视
使用 [官方 SDK](https://cn.uniview.com/Service/Service_Training/Download/SDK/2/201603/796481_194214_0.htm) 对接。

``` csharp
/// <summary>
/// 宇视相机控制
/// </summary>
public class UniviewControl : ICameraControl
{
    IntPtr lpDevHandle = IntPtr.Zero;

    NETDEV_DEVICE_LOGIN_INFO_S pstDevLoginInfo;

    // 预览句柄
    IntPtr lpRealHandle;

    /// <summary>
    /// 缩放最大倍数，用于计算球机 Z 值
    /// </summary>
    public int max_zoom = 40;

    /// <summary>
    /// 宇视相机控制构造函数
    /// </summary>
    /// <param name="ip">IP</param>
    /// <param name="userName">用户名</param>
    /// <param name="password">密码</param>
    /// <param name="port">端口</param>
    public UniviewControl(string ip, string userName, string password, int port = 80)
    {
        pstDevLoginInfo = new NETDEV_DEVICE_LOGIN_INFO_S();
        pstDevLoginInfo.szIPAddr = ip;
        pstDevLoginInfo.dwPort = port;
        pstDevLoginInfo.szUserName = userName;
        pstDevLoginInfo.szPassword = password;
        pstDevLoginInfo.dwLoginProto = (int)NETDEV_LOGIN_PROTO_E.NETDEV_LOGIN_PROTO_ONVIF;

        // 初始化
        NETDEVSDK.NETDEV_Init();
    }

    /// <summary>
    /// 登录
    /// </summary>
    /// <param name="channels">通道号集合</param>
    /// <returns>是否登录成功</returns>
    public bool Login(out List<CameraChannel> channels)
    {
        channels = new List<CameraChannel>();

        // 登录设备
        NETDEV_SELOG_INFO_S pstSELogInfo = new NETDEV_SELOG_INFO_S();
        lpDevHandle = NETDEVSDK.NETDEV_Login_V30(ref pstDevLoginInfo, ref pstSELogInfo);
        if (lpDevHandle == IntPtr.Zero)
        {
            var iLastErr = NETDEVSDK.NETDEV_GetLastError();
            return false;
        }
        else
        {
            // 获取设备通道号
            // 测试宇视调用预置位，需要在登录后执行以下获取通道详细列表后才可以使用
            int pdwChlCount = 256;
            IntPtr pstVideoChlList = new IntPtr();
            pstVideoChlList = Marshal.AllocHGlobal(256 * Marshal.SizeOf(typeof(NETDEV_VIDEO_CHL_DETAIL_INFO_S)));
            int iRet = NETDEVSDK.NETDEV_QueryVideoChlDetailList(lpDevHandle, ref pdwChlCount, pstVideoChlList);
            if (NETDEVSDK.TRUE == iRet)
            {
                var m_channelNumber = pdwChlCount;
                NETDEV_VIDEO_CHL_DETAIL_INFO_S stCHLItem = new NETDEV_VIDEO_CHL_DETAIL_INFO_S();
                for (int i = 0; i < pdwChlCount; i++)
                {
                    IntPtr ptrTemp = new IntPtr(pstVideoChlList.ToInt64() + Marshal.SizeOf(typeof(NETDEV_VIDEO_CHL_DETAIL_INFO_S)) * i);
                    stCHLItem = (NETDEV_VIDEO_CHL_DETAIL_INFO_S)Marshal.PtrToStructure(ptrTemp, typeof(NETDEV_VIDEO_CHL_DETAIL_INFO_S));

                    channels.Add(new CameraChannel()
                    {
                        ChannelID = stCHLItem.dwChannelID,
                        ChannelName = !string.IsNullOrEmpty(stCHLItem.szChnName) ? stCHLItem.szChnName : $"Channel {stCHLItem.dwChannelID}"
                    });
                }
            }
            Marshal.FreeHGlobal(pstVideoChlList);
            // 获取缩放最大倍数
            Task.Run(async () =>
            {
                max_zoom = await GetMaxZoom(pstDevLoginInfo.szIPAddr, pstDevLoginInfo.szUserName, pstDevLoginInfo.szPassword);
            });
            return true;
        }
    }

    /// <summary>
    /// 退出登录
    /// </summary>
    /// <returns>是否退出成功</returns>
    public bool Logout()
    {
        // 注销用户与释放 SDK 资源
        return NETDEVSDK.NETDEV_Logout(lpDevHandle) >= 1 && NETDEVSDK.NETDEV_Cleanup() >= 1;
    }

    /// <summary>
    /// 获取流 RTSP 地址
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="streamUris">流地址集合</param>
    /// <returns>是否获取成功</returns>
    public bool GetStreamUri(int channel, out List<string> streamUris)
    {
        streamUris = new List<string>();
        return false;
    }

    /// <summary>
    /// 实时预览
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="hPlayWnd">渲染句柄</param>
    /// <returns>是否成功</returns>
    public bool RealPlay(int channel, IntPtr hPlayWnd)
    {
        if (lpRealHandle != IntPtr.Zero)
        {
            // 已经开启预览
            return false;
        }

        NETDEV_PREVIEWINFO_S stPreviewInfo = new NETDEV_PREVIEWINFO_S();
        stPreviewInfo.dwChannelID = channel;
        stPreviewInfo.dwLinkMode = (int)NETDEV_PROTOCAL_E.NETDEV_TRANSPROTOCAL_RTPTCP;
        stPreviewInfo.dwStreamType = (int)NETDEV_LIVE_STREAM_INDEX_E.NETDEV_LIVE_STREAM_INDEX_MAIN;
        stPreviewInfo.hPlayWnd = hPlayWnd;
        lpRealHandle = NETDEVSDK.NETDEV_RealPlay(lpDevHandle, ref stPreviewInfo, IntPtr.Zero, IntPtr.Zero);
        return lpRealHandle != IntPtr.Zero;
    }

    /// <summary>
    /// 停止实时预览
    /// </summary>
    /// <returns>是否成功</returns>
    public bool StopRealPlay()
    {
        if (lpRealHandle == IntPtr.Zero)
        {
            // 未开启预览
            return false;
        }

        if (NETDEVSDK.FALSE == NETDEVSDK.NETDEV_StopRealPlay(lpRealHandle))
        {
            var iLastErr = $"stop real play: {NETDEVSDK.NETDEV_GetLastError()}";
            return false;
        }
        else
        {
            lpRealHandle = IntPtr.Zero;
            return true;
        }
    }

    /// <summary>
    /// 云台控制
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzCommand">控制命令</param>
    /// <param name="isStart">是否启动</param>
    /// <param name="speed">速度</param>
    /// <returns>是否调用成功</returns>
    public bool PtzControl(int channel, PTZCommand ptzCommand, bool isStart, int speed)
    {
        int dwPTZCommand = 0;
        switch (ptzCommand)
        {
            case PTZCommand.Light: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_LIGHTON : (int)NETDEV_PTZ_E.NETDEV_PTZ_LIGHTOFF; break;
            case PTZCommand.Wiper: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_BRUSHON : (int)NETDEV_PTZ_E.NETDEV_PTZ_BRUSHOFF; break;
            case PTZCommand.Fan: dwPTZCommand = 0; break;
            case PTZCommand.Heater: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_HEATON : (int)NETDEV_PTZ_E.NETDEV_PTZ_HEATOFF; break;
            case PTZCommand.SnowRemoval: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_SNOWREMOINGON : (int)NETDEV_PTZ_E.NETDEV_PTZ_SNOWREMOINGOFF; break;
            case PTZCommand.Aux1: dwPTZCommand = 0; break;
            case PTZCommand.Aux2: dwPTZCommand = 0; break;
            case PTZCommand.SetPreset: dwPTZCommand = (int)NETDEV_PTZ_PRESETCMD_E.NETDEV_PTZ_SET_PRESET; break;
            case PTZCommand.ClePreset: dwPTZCommand = (int)NETDEV_PTZ_PRESETCMD_E.NETDEV_PTZ_CLE_PRESET; break;
            case PTZCommand.GotoPreset: dwPTZCommand = (int)NETDEV_PTZ_PRESETCMD_E.NETDEV_PTZ_GOTO_PRESET; break;

            case PTZCommand.ZoomIn: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_ZOOMTELE : (int)NETDEV_PTZ_E.NETDEV_PTZ_ZOOMTELE_STOP; break;
            case PTZCommand.ZoomOut: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_ZOOMWIDE : (int)NETDEV_PTZ_E.NETDEV_PTZ_ZOOMWIDE_STOP; break;
            case PTZCommand.FocusNear: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_FOCUSNEAR : (int)NETDEV_PTZ_E.NETDEV_PTZ_FOCUSNEAR_STOP; break;
            case PTZCommand.FocusFar: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_FOCUSFAR : (int)NETDEV_PTZ_E.NETDEV_PTZ_FOCUSFAR_STOP; break;
            case PTZCommand.IrisOpen: dwPTZCommand = 0; break;
            case PTZCommand.IrisClose: dwPTZCommand = 0; break;

            case PTZCommand.TiltUp: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_TILTUP : (int)NETDEV_PTZ_E.NETDEV_PTZ_ALLSTOP; break;
            case PTZCommand.TiltDown: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_TILTDOWN : (int)NETDEV_PTZ_E.NETDEV_PTZ_ALLSTOP; break;
            case PTZCommand.PanLeft: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_PANLEFT : (int)NETDEV_PTZ_E.NETDEV_PTZ_ALLSTOP; break;
            case PTZCommand.PanRight: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_PANRIGHT : (int)NETDEV_PTZ_E.NETDEV_PTZ_ALLSTOP; break;
            case PTZCommand.UpLeft: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_LEFTUP : (int)NETDEV_PTZ_E.NETDEV_PTZ_ALLSTOP; break;
            case PTZCommand.UpRight: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_RIGHTUP : (int)NETDEV_PTZ_E.NETDEV_PTZ_ALLSTOP; break;
            case PTZCommand.DownLeft: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_LEFTDOWN : (int)NETDEV_PTZ_E.NETDEV_PTZ_ALLSTOP; break;
            case PTZCommand.DownRight: dwPTZCommand = isStart ? (int)NETDEV_PTZ_E.NETDEV_PTZ_RIGHTDOWN : (int)NETDEV_PTZ_E.NETDEV_PTZ_ALLSTOP; break;
            case PTZCommand.PanAuto: dwPTZCommand = 0; break;
        }

        if (NETDEVSDK.FALSE == NETDEVSDK.NETDEV_PTZControl_Other(lpDevHandle, channel, dwPTZCommand, speed))
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// 获取云台 PTZ 值
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzValue">PTZ 参数</param>
    /// <returns>是否调用成功</returns>
    public bool GetPTZValue(int channel, out PTZValue ptzValue)
    {
        NETDEV_PTZ_STATUS_S pstPTZStaus = new NETDEV_PTZ_STATUS_S();
        int iRet = NETDEVSDK.NETDEV_PTZGetStatus(lpDevHandle, channel, ref pstPTZStaus);
        if (NETDEVSDK.TRUE != iRet)
        {
            var iLastErr = $"PTZ Get Status fail: {NETDEVSDK.NETDEV_GetLastError()}";
            ptzValue = new PTZValue();
            return false;
        }
        else
        {
            ptzValue = new PTZValue()
            {
                P = (float)Math.Round(decimal.Parse($"{pstPTZStaus.fPanTiltX + 1.0}") * 180m, 2),
                T = (float)Math.Round(decimal.Parse($"{pstPTZStaus.fPanTiltY}") * 90m, 2),
                Z = (float)Math.Round(pstPTZStaus.fZoomX * (max_zoom - 1) + 1, 2)
            };
            return true;
        }
    }

    /// <summary>
    /// 设置云台 PTZ 值
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzValue">PTZ 参数</param>
    /// <returns>是否设置成功</returns>
    public bool SetPTZValue(int channel, PTZValue ptzValue)
    {
        NETDEV_PTZ_ABSOLUTE_MOVE_S pstAbsoluteMove = new NETDEV_PTZ_ABSOLUTE_MOVE_S();
        pstAbsoluteMove.fPanTiltX = ptzValue.P / 180f - 1.0f;
        pstAbsoluteMove.fPanTiltY = ptzValue.T / 90f;
        pstAbsoluteMove.fZoomX = (ptzValue.Z - 1) / (max_zoom - 1);
        int iRet = NETDEVSDK.NETDEV_PTZAbsoluteMove(lpDevHandle, channel, pstAbsoluteMove);
        if (NETDEVSDK.TRUE != iRet)
        {
            var iLastErr = $"PTZ Absolute Move fail: {NETDEVSDK.NETDEV_GetLastError()}";
            return false;
        }
        return true;
    }

    /// <summary>
    /// 获取云台预置位列表
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzPresets">预置位列表</param>
    /// <returns>是否调用成功</returns>
    public bool GetPTZPreset(int channel, out List<PTZPreset> ptzPresets)
    {
        ptzPresets = new List<PTZPreset>();

        NETDEV_PTZ_ALLPRESETS_S stPtzPresets = new NETDEV_PTZ_ALLPRESETS_S();
        int iRet = NETDEVSDK.NETDEV_GetPTZPresetList(lpDevHandle, channel, ref stPtzPresets);
        if (NETDEVSDK.TRUE != iRet)
        {
            var vError = NETDEVSDK.NETDEV_GetLastError();
            if (113 == vError)
            {
                // NETDEV_E_NOT_SUPPORT 113 设备不支持该功能
                var iLastErr = $"Get PTZ Preset List fail: {vError}";
                return false;
            }
        }
        else
        {
            for (int i = 0; i < stPtzPresets.dwSize; i++)
            {
                ptzPresets.Add(new PTZPreset()
                {
                    ID = stPtzPresets.astPreset[i].dwPresetID,
                    Name = UniviewEncoding.GetDefaultString(stPtzPresets.astPreset[i].szPresetName)
                });
            }
        }

        // 根据预置位编号排序
        ptzPresets = ptzPresets.OrderBy(x => x.ID).ToList();
        return true;
    }

    /// <summary>
    /// 云台预置位控制
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzCommand">控制命令</param>
    /// <param name="presetIndex">预置位序号</param>
    /// <param name="presetName">预置位名称，设置预置位时使用</param>
    /// <returns>是否调用成功</returns>
    public bool PTZPresetControl(int channel, PTZCommand ptzCommand, int presetIndex, string presetName)
    {
        int dwPTZCommand = 0;
        switch (ptzCommand)
        {
            case PTZCommand.SetPreset: dwPTZCommand = (int)NETDEV_PTZ_PRESETCMD_E.NETDEV_PTZ_SET_PRESET; break;
            case PTZCommand.ClePreset: dwPTZCommand = (int)NETDEV_PTZ_PRESETCMD_E.NETDEV_PTZ_CLE_PRESET; break;
            case PTZCommand.GotoPreset: dwPTZCommand = (int)NETDEV_PTZ_PRESETCMD_E.NETDEV_PTZ_GOTO_PRESET; break;
        }

        byte[] byPresetName;
        UniviewEncoding.GetUTF8Buffer(presetName, NETDEVSDK.NETDEV_LEN_32, out byPresetName);

        int iRet = NETDEVSDK.NETDEV_PTZPreset_Other(lpDevHandle, channel, dwPTZCommand, byPresetName, presetIndex);
        if (NETDEVSDK.TRUE != iRet)
        {
            var iLastErr = $"Go to preset fail: {NETDEVSDK.NETDEV_GetLastError()}";
            return false;
        }
        return true;
    }

    /// <summary>
    /// 获取缩放最大倍数
    /// 通过 LAPI Digest 鉴权获取
    /// </summary>
    /// <param name="ip">球机 IP</param>
    /// <param name="userName">用户名</param>
    /// <param name="password">密码</param>
    /// <returns></returns>
    private async Task<int> GetMaxZoom(string ip, string userName, string password)
    {
        try
        {
            // 发送 GET 请求到 API 的 URL
            var baseAddress = new Uri($"http://{ip}");

            // 创建一个具有 Digest 鉴权的 HttpClientHandler
            var handler = new HttpClientHandler
            {
                Credentials = new NetworkCredential(userName, password),
                PreAuthenticate = true
            };

            var client = new HttpClient(handler)
            {
                BaseAddress = baseAddress
            };

            var response = await client.GetAsync("/LAPI/V1.0/Channel/0/System/DigitalZoom");
            // 检查响应状态码
            if (response.IsSuccessStatusCode)
            {
                // 处理成功响应
                var responseBody = await response.Content.ReadAsStringAsync();
                var vDigitalZoom = JsonConvert.DeserializeObject<DigitalZoomModel>(responseBody);
                return (int)vDigitalZoom?.Response?.Data?.MaxDigitalZoom;
            }
            else
            {
                // 错误响应
                return 0;
            }
        }
        catch (Exception)
        {
            // 处理异常
            return 0;
        }
    }
}
```

#### 大华
使用 [官方 SDK](https://support.dahuatech.com/tools/sdkExploit) 对接（缺少调试设备，暂时未对接完）。

#### Onvif
参考 [网络接口协议](https://www.onvif.org/profiles/specifications/) 在项目中使用 WCF 引用 WSDL。
Onvif 与直接对接厂商 SDK 不同，它只提供了网络接口协议与设备通讯，所以预览视频只能通过协议获取到 RTSP 地址后播放。

``` csharp
/// <summary>
/// Onvif 相机控制
/// </summary>
public class OnvifControl : ICameraControl
{
    /// <summary>
    /// Onvif 相机
    /// </summary>
    public OnvifCamera onvifCamera;

    /// <summary>
    /// Profiles
    /// </summary>
    public CameraControl.Onvif.Client.Common.Profile[] profiles;

    /// <summary>
    /// Onvif 相机控制构造函数
    /// </summary>
    /// <param name="ip">IP</param>
    /// <param name="userName">用户名</param>
    /// <param name="password">密码</param>
    public OnvifControl(string ip, string userName, string password)
    {
        onvifCamera = new OnvifCamera(ip, userName, password);
    }

    /// <summary>
    /// 登录
    /// </summary>
    /// <param name="channels">通道号集合</param>
    /// <returns>是否登录成功</returns>
    public bool Login(out List<CameraChannel> channels)
    {
        channels = new List<CameraChannel>();

        try
        {
            var vProfilesTask = Task.Run(async () => await onvifCamera.Media.GetProfilesAsync());
            var vProfiles = vProfilesTask.Result;

            profiles = vProfiles.Profiles;

            for (int i = 0; i < profiles.Length; i++)
            {
                channels.Add(new CameraChannel()
                {
                    ChannelID = i,
                    ChannelName = profiles[i].token
                });
            }
            return true;
        }
        catch (Exception ex)
        {
            var err = $"GetProfilesAsync Error:{ex}";
            return false;
        }
    }

    /// <summary>
    /// 退出登录
    /// </summary>
    /// <returns>是否退出成功</returns>
    public bool Logout()
    {
        // Onvif 没有退出，无论如何都返回失败
        return false;
    }

    /// <summary>
    /// 获取流 RTSP 地址
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="streamUris">流地址集合</param>
    /// <returns>是否获取成功</returns>
    public bool GetStreamUri(int channel, out List<string> streamUris)
    {
        streamUris = new List<string>();

        try
        {
            var token = profiles[channel].token;
            var vPresetsTask = Task.Run(async () => await onvifCamera.Media.GetStreamUriAsync(new CameraControl.Onvif.Client.Common.StreamSetup()
            {
                Stream = CameraControl.Onvif.Client.Common.StreamType.RTPUnicast,
                Transport = new CameraControl.Onvif.Client.Common.Transport()
                {
                    Protocol = CameraControl.Onvif.Client.Common.TransportProtocol.UDP
                }
            }, token));
            var vPresets = vPresetsTask.Result;
            streamUris.Add(vPresets.Uri.Insert(7, $"{onvifCamera.UserName}:{onvifCamera.Password}@"));
            return true;
        }
        catch (Exception ex)
        {
            var err = $"GetStreamUriAsync Error:{ex}";
            return false;
        }
    }

    /// <summary>
    /// 实时预览
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="hPlayWnd">渲染句柄</param>
    /// <returns>是否成功</returns>
    public bool RealPlay(int channel, IntPtr hPlayWnd)
    {
        // Onvif 目前只能获取到 RTSP 地址，无法直接预览
        return false;
    }

    /// <summary>
    /// 停止实时预览
    /// </summary>
    /// <returns>是否成功</returns>
    public bool StopRealPlay()
    {
        // Onvif 目前只能获取到 RTSP 地址，无法直接预览
        return false;
    }

    /// <summary>
    /// 云台控制
    /// 方向、变焦、变倍、光圈
    /// 灯光、雨刷、风扇、加热、除雪
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzCommand">控制命令</param>
    /// <param name="isStart">是否启动</param>
    /// <param name="speed">速度</param>
    /// <returns>是否调用成功</returns>
    public bool PtzControl(int channel, PTZCommand ptzCommand, bool isStart, int speed = 1)
    {
        var token = profiles[channel].token;

        if (isStart)
        {
            float vP = 0, vT = 0, vZ = 0;

            // Onvif Speed 参数范围 [0~100]，需要转换为 [0~1]
            var vSpeed = (float)Math.Round((double)speed / 100, 1);

            switch (ptzCommand)
            {
                case PTZCommand.ZoomIn: vZ = vSpeed; break;
                case PTZCommand.ZoomOut: vZ = -vSpeed; break;

                case PTZCommand.TiltUp: vT = vSpeed; break;
                case PTZCommand.TiltDown: vT = -vSpeed; break;
                case PTZCommand.PanLeft: vP = -vSpeed; break;
                case PTZCommand.PanRight: vP = vSpeed; break;
                case PTZCommand.UpLeft: vP = -vSpeed; vT = vSpeed; break;
                case PTZCommand.UpRight: vP = vSpeed; vT = vSpeed; break;
                case PTZCommand.DownLeft: vP = -vSpeed; vT = -vSpeed; break;
                case PTZCommand.DownRight: vP = vSpeed; vT = -vSpeed; break;
            }

            CameraControl.Onvif.Client.Common.PTZSpeed ptzSpeed = new CameraControl.Onvif.Client.Common.PTZSpeed()
            {
                PanTilt = new CameraControl.Onvif.Client.Common.Vector2D()
                {
                    x = vP,
                    y = vT
                },
                Zoom = new CameraControl.Onvif.Client.Common.Vector1D()
                {
                    x = vZ
                }
            };

            Task.Run(async () => await onvifCamera.Ptz.ContinuousMoveAsync(token, ptzSpeed));
        }
        else
        {
            Task.Run(async () => await onvifCamera.Ptz.StopAsync(token, true, true));
        }

        return true;
    }

    /// <summary>
    /// 获取云台 PTZ 值
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzValue">PTZ 参数</param>
    /// <returns>是否调用成功</returns>
    public bool GetPTZValue(int channel, out PTZValue ptzValue)
    {
        ptzValue = new PTZValue();

        var token = profiles[channel].token;
        var vGetStatusTask = Task.Run(async () => await onvifCamera.Ptz.GetStatusAsync(token));
        var vGetStatus = vGetStatusTask.Result;

        if (vGetStatus != null && vGetStatus.Position != null)
        {
            ptzValue.P = vGetStatus.Position.PanTilt.x;
            ptzValue.T = vGetStatus.Position.PanTilt.y;
            ptzValue.Z = vGetStatus.Position.Zoom.x;
        }
        return true;
    }

    /// <summary>
    /// 设置云台 PTZ 值
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzValue">PTZ 参数</param>
    /// <returns>是否设置成功</returns>
    public bool SetPTZValue(int channel, PTZValue ptzValue)
    {
        // 获取球机绝对移动范围
        var vGetNodesTask = Task.Run(async () => await onvifCamera.Ptz.GetNodesAsync());
        var vGetNodes = vGetNodesTask.Result;
        if (vGetNodes != null && vGetNodes.PTZNode != null && vGetNodes.PTZNode.Count() >= 1)
        {
            var vPanTilt = vGetNodes.PTZNode[0].SupportedPTZSpaces.AbsolutePanTiltPositionSpace;
            var vZoom = vGetNodes.PTZNode[0].SupportedPTZSpaces.AbsoluteZoomPositionSpace;
        }

        // 调用球机绝对移动
        var token = profiles[channel].token;
        var vectorPT = new CameraControl.Onvif.Client.Common.PTZVector { PanTilt = new CameraControl.Onvif.Client.Common.Vector2D { x = ptzValue.P, y = ptzValue.T } };
        var speedPT = new CameraControl.Onvif.Client.Common.PTZSpeed { PanTilt = new CameraControl.Onvif.Client.Common.Vector2D { x = 1f, y = 1f } };
        Task.Run(async () => await onvifCamera.Ptz.AbsoluteMoveAsync(token, vectorPT, speedPT));
        Thread.Sleep(1000);
        var vectorZ = new CameraControl.Onvif.Client.Common.PTZVector { Zoom = new CameraControl.Onvif.Client.Common.Vector1D { x = ptzValue.Z } };
        var speedZ = new CameraControl.Onvif.Client.Common.PTZSpeed { Zoom = new CameraControl.Onvif.Client.Common.Vector1D { x = 1f } };
        Task.Run(async () => await onvifCamera.Ptz.AbsoluteMoveAsync(token, vectorZ, speedZ));
        return true;
    }

    /// <summary>
    /// 获取云台预置位列表
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzPresets">预置位列表</param>
    /// <returns>是否调用成功</returns>
    public bool GetPTZPreset(int channel, out List<PTZPreset> ptzPresets)
    {
        ptzPresets = new List<PTZPreset>();

        var token = profiles[channel].token;
        var vPresetsTask = Task.Run(async () => await onvifCamera.Ptz.GetPresetsAsync(token));
        var vPresets = vPresetsTask.Result;

        foreach (var item in vPresets.Preset)
        {
            ptzPresets.Add(new PTZPreset()
            {
                ID = int.Parse(item.token),
                Name = item.Name
            });
        }

        // 根据预置位编号排序
        ptzPresets = ptzPresets.OrderBy(x => x.ID).ToList();
        return true;
    }

    /// <summary>
    /// 云台预置位控制
    /// </summary>
    /// <param name="channel">通道号</param>
    /// <param name="ptzCommand">控制命令</param>
    /// <param name="presetIndex">预置位序号</param>
    /// <param name="presetName">预置位名称，设置预置位时使用</param>
    /// <returns>是否调用成功</returns>
    public bool PTZPresetControl(int channel, PTZCommand ptzCommand, int presetIndex, string presetName)
    {
        var token = profiles[channel].token;

        switch (ptzCommand)
        {
            case PTZCommand.SetPreset:
                CameraControl.Onvif.Client.Ptz.SetPresetRequest setPresetRequest = new CameraControl.Onvif.Client.Ptz.SetPresetRequest()
                {
                    ProfileToken = token,
                    PresetToken = $"{presetIndex}",
                    PresetName = presetName
                };
                Task.Run(async () => await onvifCamera.Ptz.SetPresetAsync(setPresetRequest));
                break;
            case PTZCommand.ClePreset:
                Task.Run(async () => await onvifCamera.Ptz.RemovePresetAsync(token, $"{presetIndex}"));
                break;
            case PTZCommand.GotoPreset:
                Task.Run(async () => await onvifCamera.Ptz.GotoPresetAsync(token, $"{presetIndex}", new CameraControl.Onvif.Client.Common.PTZSpeed()));
                break;
        }

        return true;
    }
}
```
