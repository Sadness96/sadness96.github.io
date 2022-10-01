---
title: 获取系统状态信息
date: 2018-06-18 16:00:58
tags: [c#,helper]
categories: C#.Net
---
### 显示 CPU 与 ARM 占用率
<!-- more -->
#### 简介
在一些占用系统资源较大的软件上可以增加 CPU 与 ARM 的占用率，用于美化界面的同时可以实时监视系统占用状况。
依赖于 [PerformanceCounter](https://learn.microsoft.com/en-us/dotnet/api/system.diagnostics.performancecounter?view=dotnet-plat-ext-6.0) 方法查询系统性能计数器，获取 CPU 与 ARM 基础信息，通过计算后获取占用率信息。

#### 代码
``` CSharp
/// <summary>
/// 系统使用率统计
/// </summary>
public class SystemStateHelper
{
    /// <summary>
    /// 获取全局占用率
    /// </summary>
    public SystemStateHelper()
    {
        Task.Run(() =>
        {
            PerformanceCounter CpuCounter = new PerformanceCounter("Processor Information", "% Processor Utility", "_Total");
            PerformanceCounter RamCounter = new PerformanceCounter("Memory", "Available MBytes");
            double TotalMemoryMBytesCapacity = GetTotalMemoryMBytesCapacity();

            while (true)
            {
                var cpuUsage = CpuCounter.NextValue();
                cpuUsage = cpuUsage >= 100 ? 100 : cpuUsage;

                var ramAvailable = RamCounter.NextValue();
                var memUsage = Math.Round((TotalMemoryMBytesCapacity - ramAvailable) / TotalMemoryMBytesCapacity, 4) * 100;
                memUsage = memUsage >= 100 ? 100 : memUsage;

                CpuCounterChange?.Invoke(cpuUsage);
                RamCounterChange?.Invoke(memUsage);
                Thread.Sleep(500);
            }
        });
    }

    /// <summary>
    /// 获取指定 pid 进程占用率
    /// </summary>
    /// <param name="pid">pid</param>
    public SystemStateHelper(int pid)
    {
        Task.Run(() =>
        {
            const float mega = 1024 * 1024;
            var vInstanceName = GetProcessInstanceName(pid);

            if (!string.IsNullOrEmpty(vInstanceName))
            {
                PerformanceCounter cpuPerformanceCounter = new PerformanceCounter("Process", "% Processor Time", vInstanceName);
                PerformanceCounter memoryPerformanceCounter = new PerformanceCounter("Process", "Working Set - Private", vInstanceName);

                while (true)
                {
                    try
                    {
                        float mainCpu = cpuPerformanceCounter.NextValue() / Environment.ProcessorCount;
                        mainCpu = mainCpu >= 100 ? 100 : mainCpu;

                        float mainRam = memoryPerformanceCounter.NextValue() / mega;

                        CpuCounterChange.Invoke(mainCpu);
                        RamCounterChange.Invoke(mainRam);
                    }
                    catch (Exception)
                    {
                        // pid 查询不到进程
                    }

                    Thread.Sleep(500);
                }
            }
        });
    }

    /// <summary>
    /// CPU 使用率
    /// </summary>
    public event Action<double> CpuCounterChange;

    /// <summary>
    /// 内存使用率
    /// </summary>
    public event Action<double> RamCounterChange;

    /// <summary>
    /// 获取总内存字节容量
    /// </summary>
    /// <returns></returns>
    private double GetTotalMemoryMBytesCapacity()
    {
        using (var mc = new ManagementClass("Win32_PhysicalMemory"))
        {
            using (var moc = mc.GetInstances())
            {
                double totalCapacity = 0d;
                foreach (var mo in moc)
                {
                    var moCapacity = long.Parse(mo.Properties["Capacity"].Value.ToString());
                    totalCapacity += Math.Round(moCapacity / 1024.0 / 1024, 1);
                }
                return totalCapacity;
            }
        }
    }

    /// <summary>
    /// 获取进程实例名称
    /// </summary>
    /// <param name="pid"></param>
    /// <returns></returns>
    private string GetProcessInstanceName(int pid)
    {
        PerformanceCounterCategory processCategory = new PerformanceCounterCategory("Process");
        string[] runnedInstances = processCategory.GetInstanceNames();

        foreach (string runnedInstance in runnedInstances)
        {
            using (PerformanceCounter performanceCounter = new PerformanceCounter("Process", "ID Process", runnedInstance, true))
            {
                try
                {
                    if ((int)performanceCounter?.RawValue == pid)
                    {
                        return runnedInstance;
                    }
                }
                catch (Exception)
                { }
            }
        }
        return "";
    }
}
```

#### 调用说明
* 调用 SystemStateHelper() 获取全局占用率，返回结果为 CPU 与 ARM 占用率百分比，两位小数。
* 调用 SystemStateHelper(int pid) 获取指定 pid 进程占用率，返回结果为 CPU 占用率百分比，两位小数，ARM 占用字节，单位 MB。
* 通过 PerformanceCounter 方法查询占用率仅支持通过名字查询，但是运行多个相同名字的进程，会隐性增加命名后缀例如 #1、#2、#3，所以调用 GetProcessInstanceName(pid) 方法可以获取 pid 对应精确名称。
* 通过 pid 查询占用率时仅会查询自身进程的占用率，但是通常一个大型系统运行起来会调用多个不同进程组件，会存在与任务管理器显示不一致，可以先查询 pid 关联所有子线程，查询到占用率后叠加显示。
    ``` CSharp
    /// <summary>
    /// 获取指定 pid 关联子进程信息
    /// </summary>
    public static Dictionary<int, string> GetAllProcess(int pid)
    {
        Dictionary<int, string> dicAllProcess = new Dictionary<int, string>();
        ManagementObjectSearcher searcher = new ManagementObjectSearcher($"Select * From Win32_Process Where ParentProcessID={pid}");
        foreach (ManagementObject mo in searcher.Get())
        {
            dicAllProcess.Add(int.Parse($"{mo["ProcessID"]}"), $"{mo["Name"]}");
        }
        return dicAllProcess;
    }
    ```