---
title: 超时自动删除集合数据
date: 2021-06-14 14:28:14
tags: [c#]
categories: C#.Net
---
### 重写 Dictionary 达到数据超时自动删除
<!-- more -->
#### 简介
在实际应用中存在一种需求，是数据存在有效时间，只管添加，超时后需要自动删除。通过重写集合的方式达到数据集合超时自动删除数据的需求。

#### 代码
在新增数据时创建一个针对于这条数据的计时器，调用时使用重写的 Add 方法传入数据有效时间即可。

``` CSharp
/// <summary>
/// 重写 Dictionary
/// </summary>
/// <typeparam name="TKey"></typeparam>
/// <typeparam name="TValue"></typeparam>
public class TimeOutDictionary<TKey, TValue> : Dictionary<TKey, TValue>
{
    private static object locker = new object();

    /// <summary>
    /// 重写
    /// </summary>
    /// <param name="key"></param>
    /// <param name="value"></param>
    /// <param name="timeSpan">超时时间</param>
    public void Add(TKey key, TValue value, TimeSpan timeSpan)
    {
        lock (locker)
        {
            if (base.ContainsKey(key))
            {
                base[key] = value;
            }
            else
            {
                base.Add(key, value);
            }

            SetTimeoutDelete(timeSpan.TotalMilliseconds, key);
        }
    }

    /// <summary>
    /// 在指定时间过后删除数据
    /// </summary>
    /// <param name="interval">事件之间经过的时间（以毫秒为单位）</param>
    /// <param name="key">要删除的数据Key</param>
    public void SetTimeoutDelete(double interval, TKey key)
    {
        Timer timer = new Timer(interval);
        timer.Elapsed += delegate (object sender, ElapsedEventArgs e)
        {
            lock (locker)
            {
                timer.Enabled = false;
                base.Remove(key);
            }
        };
        timer.Enabled = true;
    }
}
``` 