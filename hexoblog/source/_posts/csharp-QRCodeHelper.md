---
title: 二维码帮助类
date: 2017-06-06 21:24:00
tags: [c#,helper,qrcode]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-DevFramework/%E7%94%9F%E6%88%90%E4%BA%8C%E7%BB%B4%E7%A0%81.png"/>

<!-- more -->
### 基于 ZXing.QrCode 库制作的横条码/二维码生成识别工具
#### 简介
[二维码（QR Code码）](https://baike.baidu.com/item/%E4%BA%8C%E7%BB%B4%E7%A0%81) 在现在生活中十分常见，扫一扫即可付款。原本想要学习一下二维码原理自己造轮子，后来还是决定站在巨人的肩膀上使用Google的开源库ZXing来生成和识别二维码/横条码。
#### 帮助类
[QRCodeHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/QRCode/QRCodeHelper.cs) 帮助类提供二维码生成，带LOGO二维码，以及图像识别功能。
``` CSharp
/// <summary>
/// 注销对象方法API
/// </summary>
[DllImport("gdi32")]
private static extern int DeleteObject(IntPtr o);

/// <summary>
/// 生成二维码
/// </summary>
/// <param name="strContent">二维码文本</param>
/// <param name="iWidth">二维码宽度</param>
/// <param name="iHeigth">二维码高度</param>
/// <returns>二维码位图</returns>
public static Bitmap GetQRCode(string strContent, int iWidth, int iHeigth)
{
    try
    {
        //构造二维码写码器
        MultiFormatWriter writer = new MultiFormatWriter();
        Dictionary<EncodeHintType, object> hint = new Dictionary<EncodeHintType, object>();
        hint.Add(EncodeHintType.CHARACTER_SET, "UTF-8");
        hint.Add(EncodeHintType.ERROR_CORRECTION, ErrorCorrectionLevel.H);
        hint.Add(EncodeHintType.MARGIN, 1);
        //生成二维码 
        BitMatrix bitMatrix = writer.encode(strContent, BarcodeFormat.QR_CODE, iWidth, iHeigth, hint);
        BarcodeWriter barcodeWriter = new BarcodeWriter();
        Bitmap bitmapQRCode = barcodeWriter.Write(bitMatrix);
        //获取二维码实际尺寸(去掉二维码两边空白后的实际尺寸)
        int[] rectangle = bitMatrix.getEnclosingRectangle();
        //将img转换成bmp格式，否则后面无法创建Graphics对象
        Bitmap bitmapQRCodeBMP = new Bitmap(bitmapQRCode.Width, bitmapQRCode.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        using (Graphics g = Graphics.FromImage(bitmapQRCodeBMP))
        {
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            g.DrawImage(bitmapQRCode, 0, 0);
        }
        return bitmapQRCodeBMP;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 生成二维码(带LOGO)
/// </summary>
/// <param name="strContent">二维码文本</param>
/// <param name="iWidth">二维码宽度</param>
/// <param name="iHeigth">二维码高度</param>
/// <param name="strLogoPath">LOGO图片路径</param>
/// <returns>二维码位图</returns>
public static Bitmap GetQRCode_logo(string strContent, int iWidth, int iHeigth, string strLogoPath)
{
    try
    {
        //构造二维码写码器
        MultiFormatWriter writer = new MultiFormatWriter();
        Dictionary<EncodeHintType, object> hint = new Dictionary<EncodeHintType, object>();
        hint.Add(EncodeHintType.CHARACTER_SET, "UTF-8");
        hint.Add(EncodeHintType.ERROR_CORRECTION, ErrorCorrectionLevel.H);
        hint.Add(EncodeHintType.MARGIN, 1);
        //生成二维码 
        BitMatrix bitMatrix = writer.encode(strContent, BarcodeFormat.QR_CODE, iWidth, iHeigth, hint);
        BarcodeWriter barcodeWriter = new BarcodeWriter();
        Bitmap bitmapQRCode = barcodeWriter.Write(bitMatrix);
        //获取二维码实际尺寸(去掉二维码两边空白后的实际尺寸)
        int[] rectangle = bitMatrix.getEnclosingRectangle();
        //将img转换成bmp格式，否则后面无法创建Graphics对象
        Bitmap bitmapQRCodeBMP = new Bitmap(bitmapQRCode.Width, bitmapQRCode.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        using (Graphics g = Graphics.FromImage(bitmapQRCodeBMP))
        {
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            g.DrawImage(bitmapQRCode, 0, 0);
        }
        //获得LOGO位图并计算插入图片的大小和位置
        Bitmap bitmapLogo = new Bitmap(strLogoPath);
        int middleW = Math.Min((int)(rectangle[2] / 3.5), bitmapLogo.Width);
        int middleH = Math.Min((int)(rectangle[3] / 3.5), bitmapLogo.Height);
        int middleL = (bitmapQRCode.Width - middleW) / 2;
        int middleT = (bitmapQRCode.Height - middleH) / 2;
        //将二维码插入图片(白底)
        Graphics myGraphic = Graphics.FromImage(bitmapQRCodeBMP);
        myGraphic.DrawImage(bitmapLogo, middleL, middleT, middleW, middleH);
        return bitmapQRCodeBMP;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 从位图获取图像源
/// </summary>
/// <param name="bitmapQRCode">位图二维码</param>
/// <returns>图像源(用以显示在控件上)</returns>
public static ImageSource GetImageSourceFromBitmap(Bitmap bitmapQRCode)
{
    try
    {
        IntPtr ipQRCode = bitmapQRCode.GetHbitmap();
        BitmapSource bitmapSource = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(ipQRCode, IntPtr.Zero, Int32Rect.Empty, System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());
        DeleteObject(ipQRCode);
        return bitmapSource;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 识别二维码和条形码
/// </summary>
/// <param name="bitmapQRCode">二维码和条形码位图</param>
/// <returns>成功返回二维码和条形码内容,失败返回NULL或Empty</returns>
public static string BarcodeReader(Bitmap bitmapQRCode)
{
    try
    {
        BarcodeReader barcodeReader = new BarcodeReader();
        barcodeReader.Options.CharacterSet = "UTF-8";
        Result resultQRCode = barcodeReader.Decode(bitmapQRCode);
        if (resultQRCode == null)
        {
            return string.Empty;
        }
        else
        {
            return resultQRCode.Text;
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 识别二维码和条形码
/// </summary>
/// <param name="strQRCodePath">二维码和条形码文件路径</param>
/// <returns>成功返回二维码和条形码内容,失败返回NULL或Empty</returns>
public static string BarcodeReader(string strQRCodePath)
{
    try
    {
        BarcodeReader barcodeReader = new BarcodeReader();
        barcodeReader.Options.CharacterSet = "UTF-8";
        Bitmap bitmapQRCode = new Bitmap(strQRCodePath);
        Result resultQRCode = barcodeReader.Decode(bitmapQRCode);
        if (resultQRCode == null)
        {
            return string.Empty;
        }
        else
        {
            return resultQRCode.Text;
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}

/// <summary>
/// 保存二维码和条形码位图到指定位置
/// </summary>
/// <param name="strSavePath">文件保存位置</param>
/// <param name="bitmapQRCode">二维码和条形码位图</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool SaveBitmap(string strSavePath, Bitmap bitmapQRCode)
{
    try
    {
        ImageFormat imageFormat;
        switch (System.IO.Path.GetExtension(strSavePath))
        {
            case ".bmp": imageFormat = ImageFormat.Bmp; break;
            case ".emf": imageFormat = ImageFormat.Emf; break;
            case ".exif": imageFormat = ImageFormat.Exif; break;
            case ".gif": imageFormat = ImageFormat.Gif; break;
            case ".icon": imageFormat = ImageFormat.Icon; break;
            case ".jpeg": imageFormat = ImageFormat.Jpeg; break;
            case ".jpg": imageFormat = ImageFormat.Jpeg; break;
            case ".memorybmp": imageFormat = ImageFormat.MemoryBmp; break;
            case ".png": imageFormat = ImageFormat.Png; break;
            case ".tiff": imageFormat = ImageFormat.Tiff; break;
            case ".wmf": imageFormat = ImageFormat.Wmf; break;
            default: imageFormat = ImageFormat.Png; break;
        }
        bitmapQRCode.Save(strSavePath, imageFormat);
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
```