---
title: Xbox 控制器捕获
date: 2021-06-20 13:50:00
tags: [c#,wpf,helper]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-XInput/XboxOneController.png"/>

<!-- more -->
### 简介
写这个帮助类是对 [摇杆控制器捕获](https://sadness96.github.io/blog/2021/05/22/csharp-DirectInput/) 接口的一个延伸，之前有了解到 OBS 插件 [input-overlay](https://github.com/univrsal/input-overlay) 可以显示手柄信息用于直播，所以我也做了一个UI用于显示。

### 实现方式
手柄型号为 Xbox One，使用 [XInput API](https://docs.microsoft.com/en-us/windows/win32/xinput/getting-started-with-xinput) 与手柄通信。

### 核心代码
完整代码查看：[GamepadController](https://github.com/Sadness96/GamepadController)
``` csharp
/// <summary>
/// 从Windows的Xbox控制器接收输入
/// https://docs.microsoft.com/en-us/windows/win32/xinput/getting-started-with-xinput
/// </summary>
public class XInputHelper
{
    /// <summary>
    /// 是否连接控制器
    /// </summary>
    public bool isGetJoystick = false;

    /// <summary>
    /// 连接到的控制器
    /// </summary>
    public Controller controller;

    /// <summary>
    /// 控制器状态捕获计时器
    /// </summary>
    private Timer _timer;

    /// <summary>
    /// 当前按键状态
    /// 用于判断两次按键差异
    /// </summary>
    private GamepadButtonFlags ButtonsData;

    /// <summary>
    /// LT 按键状态
    /// 用于判断两次按键差异
    /// </summary>
    private byte LeftTriggerData;

    /// <summary>
    /// RT 按键状态
    /// 用于判断两次按键差异
    /// </summary>
    private byte RightTriggerData;

    /// <summary>
    /// 左摇杆 X 坐标状态
    /// 用于判断两次摇杆坐标差异
    /// </summary>
    private short LeftThumbXData;

    /// <summary>
    /// 左摇杆 Y 坐标状态
    /// 用于判断两次摇杆坐标差异
    /// </summary>
    private short LeftThumbYData;

    /// <summary>
    /// 右摇杆 X 坐标状态
    /// 用于判断两次摇杆坐标差异
    /// </summary>
    private short RightThumbXData;

    /// <summary>
    /// 右摇杆 Y 坐标状态
    /// 用于判断两次摇杆坐标差异
    /// </summary>
    private short RightThumbYData;

    /// <summary>
    /// 按钮变化事件
    /// </summary>
    public event Action<GamepadButtonFlags> ButtonsChange;

    /// <summary>
    /// LT 按键变化事件
    /// </summary>
    public event Action<byte> LeftTriggerChange;

    /// <summary>
    /// LT 按键变化事件
    /// </summary>
    public event Action<byte> RightTriggerChange;

    /// <summary>
    /// 左摇杆 X 变化事件
    /// </summary>
    public event Action<short> LeftThumbXChange;

    /// <summary>
    /// 左摇杆 Y 变化事件
    /// </summary>
    public event Action<short> LeftThumbYChange;

    /// <summary>
    /// 右摇杆 X 变化事件
    /// </summary>
    public event Action<short> RightThumbXChange;

    /// <summary>
    /// 右摇杆 Y 变化事件
    /// </summary>
    public event Action<short> RightThumbYChange;

    /// <summary>
    /// 连接控制器
    /// </summary>
    /// <returns></returns>
    public bool ConnectGamepad()
    {
        if (!isGetJoystick && _timer == null)
        {
            controller = new Controller(UserIndex.One);
            if (controller != null)
            {
                isGetJoystick = true;
                _timer = new Timer(obj => Update());
                _timer.Change(0, 1000 / 60);
            }
        }
        return isGetJoystick;
    }

    /// <summary>
    /// 断开控制器
    /// </summary>
    /// <returns></returns>
    public void BreakOffGamepad()
    {
        if (_timer != null)
        {
            _timer.Dispose();
            _timer = null;
        }
        if (isGetJoystick)
        {
            isGetJoystick = false;
        }
    }

    /// <summary>
    /// 捕获控制器数据
    /// </summary>
    private void Update()
    {
        try
        {
            #region 其他功能
            //// 获取电池电量和电池类型
            //var vGetBatteryInformation = controller.GetBatteryInformation(BatteryDeviceType.Gamepad);
            //// 获取功能按钮和震动马达信息(1)
            //var vGetCapabilities = controller.GetCapabilities(DeviceQueryType.Any);
            //// 获取功能按钮和震动马达信息(2) 
            //Capabilities vCapabilities;
            //var vIsGetCapabilities = controller.GetCapabilities(DeviceQueryType.Any, out vCapabilities);
            //// 获取按键信息
            //Keystroke vKeystroke;
            //var vGetKeystroke = controller.GetKeystroke(DeviceQueryType.Any, out vKeystroke);
            //// 获取状态(1)
            //var vGetState = controller.GetState();
            //// 获取状态(2)
            //State vState;
            //var vIsGetState = controller.GetState(out vState);
            //// 设置震动马达
            ////var vVibration = vGetCapabilities.Vibration;
            ////var vSetVibration = controller.SetVibration(vGetCapabilities.Vibration);
            #endregion

            // 获取状态
            var vGetState = controller.GetState();
            // 按钮
            var vButtons = vGetState.Gamepad.Buttons;
            if (ButtonsData != vButtons)
            {
                ButtonsData = vButtons;
                ButtonsChange?.Invoke(ButtonsData);
            }
            // LT 按键 0-255
            var vLeftTrigger = vGetState.Gamepad.LeftTrigger;
            if (LeftTriggerData != vLeftTrigger)
            {
                LeftTriggerData = vLeftTrigger;
                LeftTriggerChange?.Invoke(LeftTriggerData);
            }
            // RT 按键 0-255
            var vRightTrigger = vGetState.Gamepad.RightTrigger;
            if (RightTriggerData != vRightTrigger)
            {
                RightTriggerData = vRightTrigger;
                RightTriggerChange?.Invoke(RightTriggerData);
            }
            // 左摇杆 X
            var vLeftThumbX = vGetState.Gamepad.LeftThumbX;
            if (!LeftThumbXData.Equals(vLeftThumbX))
            {
                LeftThumbXData = vLeftThumbX;
                LeftThumbXChange?.Invoke(LeftThumbXData);
            }
            // 左摇杆 Y
            var vLeftThumbY = vGetState.Gamepad.LeftThumbY;
            if (!LeftThumbYData.Equals(vLeftThumbY))
            {
                LeftThumbYData = vLeftThumbY;
                LeftThumbYChange?.Invoke(LeftThumbYData);
            }
            // 右摇杆 X
            var vRightThumbX = vGetState.Gamepad.RightThumbX;
            if (!RightThumbXData.Equals(vRightThumbX))
            {
                RightThumbXData = vRightThumbX;
                RightThumbXChange?.Invoke(RightThumbXData);
            }
            // 右摇杆 Y
            var vRightThumbY = vGetState.Gamepad.RightThumbY;
            if (!RightThumbYData.Equals(vRightThumbY))
            {
                RightThumbYData = vRightThumbY;
                RightThumbYChange?.Invoke(RightThumbYData);
            }
        }
        catch (Exception ex)
        {
            BreakOffGamepad();
        }
    }
}
```

### 游戏演示
<img src="https://sadness96.github.io/images/blog/csharp-XInput/CupheadRecord.gif"/>