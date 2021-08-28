---
title: 摇杆控制器捕获
date: 2021-05-22 22:30:00
tags: [c#,wpf,helper]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-DirectInput/QuanbaQ1Controller.png"/>

<!-- more -->
### 简介
原本工作需要对接一个球机的操作键盘，奈何厂商一直没做出来，听到键盘的操作描述与街机摇杆相似，刚好手边有一个，就先写一个Demo尝试一下。
<img src="https://sadness96.github.io/images/blog/csharp-DirectInput/722多光谱球机.jpg"/>

### 实现方式
手头的摇杆型号为：[拳霸 Q1W](http://www.qanba.com/ProductDetail/2059904.html)，接入方式为 xbox360 驱动，使用 DirectInput 获取摇杆操作信息。
由于穷，只买了这一种摇杆，所以其他的品牌或型号的按键可能不太匹配，修改前端的判断即可。

### 核心代码
完整代码查看：[GamepadController](https://github.com/Sadness96/GamepadController)
``` csharp
/// <summary>
/// DirectInput是用于输入设备（包括鼠标，键盘，操纵杆和其他游戏控制器）
/// https://docs.microsoft.com/en-us/previous-versions/windows/desktop/ee418273(v=vs.85)
/// </summary>
public class DirectInputHelper
{
    /// <summary>
    /// 是否连接控制器
    /// </summary>
    public bool isGetJoystick = false;

    /// <summary>
    /// 连接到的控制器
    /// </summary>
    private Joystick curJoystick;

    /// <summary>
    /// 控制器状态捕获计时器
    /// </summary>
    private Timer _timer;

    /// <summary>
    /// 当前摇杆状态
    /// 用于判断两次摇杆差异
    /// </summary>
    private int[] RockerData;

    /// <summary>
    /// 当前按键状态
    /// 用于判断两次按键差异
    /// </summary>
    private bool[] ButtonData;

    /// <summary>
    /// 摇杆变化事件
    /// </summary>
    public event Action<int[]> RockerChange;

    /// <summary>
    /// 按钮变化事件
    /// </summary>
    public event Action<bool[]> ButtonChange;

    /// <summary>
    /// 连接控制器
    /// </summary>
    /// <returns></returns>
    public bool ConnectGamepad()
    {
        if (!isGetJoystick && _timer == null)
        {
            var vDirectInput = new DirectInput();
            var allDevices = vDirectInput.GetDevices();
            foreach (var item in allDevices)
            {
                if (item.Type == DeviceType.Gamepad)
                {
                    curJoystick = new Joystick(vDirectInput, item.InstanceGuid);
                    curJoystick.Acquire();
                    isGetJoystick = true;
                    _timer = new Timer(obj => Update());
                    _timer.Change(0, 1000 / 60);
                }
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
            var joys = curJoystick.GetCurrentState();
            // 摇杆
            if (RockerData == null || !Enumerable.SequenceEqual(RockerData, joys.PointOfViewControllers))
            {
                RockerData = joys.PointOfViewControllers;
                RockerChange.Invoke(RockerData);
            }
            // 按钮
            if (ButtonData == null || !Enumerable.SequenceEqual(ButtonData, joys.Buttons))
            {
                ButtonData = joys.Buttons;
                ButtonChange.Invoke(ButtonData);
            }
        }
        catch (Exception)
        {
            BreakOffGamepad();
        }
    }
}
```

### 游戏演示
<img src="https://sadness96.github.io/images/blog/csharp-DirectInput/MameRecord.gif"/>