---
title: Windows屏幕截图帮助类
date: 2017-06-21 16:47:10
tags: [c#,helper,redis]
categories: C#.Net
---
### 截取Windows屏幕全屏或指定区域帮助类
<!-- more -->
#### 简介
平时习惯于用QQ截图，但是公司一台电脑没有外网链接，登录不了QQ截图就很麻烦，通常是按PrtSc键截取全屏幕，然后在粘贴在Windows自带的画图应用中截取区域。原本计划自己实现一个仿制QQ截图（画笔、框选、编辑文字、提取颜色）的功能，但是一直也没有付诸行动，但是需要用到的技术栈都已经整理。屏幕录像的原理也是按照固定的[FPS](https://baike.baidu.com/item/%E5%B8%A7%E7%8E%87/1052590)写入视频流，但是实际测试截取速度比较慢，无法稳定在30FPS以上。
#### 帮助类
[ScreenshotHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Screenshot/ScreenshotHelper.cs)
``` CSharp
/// <summary>
/// 全屏幕截图
/// </summary>
/// <returns>截图Bitmap</returns>
public static Bitmap ScreenshotFullScreen()
{
    try
    {
        //得到屏幕整体宽度
        double dPrimaryScreenWidth = SystemParameters.PrimaryScreenWidth;
        //得到屏幕整体高度
        double dPrimaryScreenHeight = SystemParameters.PrimaryScreenHeight;
        //初始化使用指定的大小(屏幕大小)的 System.Drawing.Bitmap 类的新实例.
        Bitmap bitmapScreenshot = new Bitmap((int)dPrimaryScreenWidth, (int)dPrimaryScreenHeight);
        //从指定的载入原创建新的 System.Drawing.Graphics.
        Graphics graphicsScreenshot = Graphics.FromImage(bitmapScreenshot);
        //获取或设置绘制到此 System.Drawing.Graphics 的渲染质量:高质量 低速度合成.
        graphicsScreenshot.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
        //截取电脑屏幕:从屏幕到 System.Drawing.Graphics 的绘图图面.
        graphicsScreenshot.CopyFromScreen((int)0, (int)0, (int)0, (int)0, new System.Drawing.Size((int)dPrimaryScreenWidth, (int)dPrimaryScreenHeight));
        return bitmapScreenshot;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 截取指定位置截图
/// </summary>
/// <param name="iStartX">截取起始坐标X</param>
/// <param name="iStartY">截取起始坐标Y</param>
/// <param name="iInterceptWidth">截取宽度</param>
/// <param name="iInterceptHeight">截取高度</param>
/// <returns>截图Bitmap</returns>
public static Bitmap ScreenshotsSpecifyLocation(int iStartX, int iStartY, int iInterceptWidth, int iInterceptHeight)
{
    try
    {
        //初始化使用指定的大小(屏幕大小)的 System.Drawing.Bitmap 类的新实例.
        Bitmap bitmapScreenshot = new Bitmap((int)iInterceptWidth, (int)iInterceptHeight);
        //从指定的载入原创建新的 System.Drawing.Graphics.
        Graphics graphicsScreenshot = Graphics.FromImage(bitmapScreenshot);
        //获取或设置绘制到此 System.Drawing.Graphics 的渲染质量:高质量 低速度合成.
        graphicsScreenshot.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
        //截取电脑屏幕:从屏幕到 System.Drawing.Graphics 的绘图图面.
        graphicsScreenshot.CopyFromScreen(iStartX, iStartY, (int)0, (int)0, new System.Drawing.Size((int)iInterceptWidth, (int)iInterceptHeight));
        return bitmapScreenshot;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}
```