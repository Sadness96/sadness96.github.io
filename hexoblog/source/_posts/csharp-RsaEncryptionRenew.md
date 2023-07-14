---
title: c# RSA 加密解密帮助类更新
date: 2023-06-25 22:51:20
tags: [c#,helper,rsa]
categories: C#.Net
---
### 更新 RSA 帮助类解决加密文本长度
<!-- more -->
### 简介
由于旧版本 [RSA 帮助类](https://liujiahua.com/blog/2018/01/10/csharp-EncryptionHelper/#RSA) 对加密的文字长度有限制，所以更新代码以加密更长的字符。

### 代码
``` CSharp
/// <summary>
/// RSA加密解密帮助类
/// </summary>
public class RSAHelper
{
    /// <summary>
    /// Rsa 生成秘钥
    /// </summary>
    /// <param name="xmlPublicKey">公钥</param>
    /// <param name="xmlPrivateKey">私钥</param>
    public static void GenerateRSAKeys(out string xmlPublicKey, out string xmlPrivateKey)
    {
        try
        {
            using RSA rsa = RSA.Create();
            xmlPublicKey = rsa.ToXmlString(false);
            xmlPrivateKey = rsa.ToXmlString(true);
        }
        catch (Exception ex)
        {
            xmlPublicKey = string.Empty;
            xmlPrivateKey = string.Empty;
        }
    }

    /// <summary>
    /// RSA 加密
    /// </summary>
    /// <param name="xmlPublicKey">公钥</param>
    /// <param name="strPlaintext">明文</param>
    /// <returns>RSA密文</returns>
    public static string RSAEncrypt(string xmlPublicKey, string strPlaintext)
    {
        try
        {
            byte[] originalData = Encoding.UTF8.GetBytes(strPlaintext);
            using RSA rsa = RSA.Create();
            rsa.FromXmlString(xmlPublicKey);
            byte[] encryptedData = rsa.Encrypt(originalData, RSAEncryptionPadding.OaepSHA1);
            return Convert.ToBase64String(encryptedData);
        }
        catch (Exception ex)
        {
            return string.Empty;
        }
    }

    /// <summary>
    /// RSA 解密
    /// </summary>
    /// <param name="xmlPrivateKey">私钥</param>
    /// <param name="strCiphertext">RSA密文</param>
    /// <returns>明文</returns>
    public static string RSADecrypt(string xmlPrivateKey, string strCiphertext)
    {
        try
        {
            byte[] encryptedData = Convert.FromBase64String(strCiphertext);
            using RSA rsa = RSA.Create();
            rsa.FromXmlString(xmlPrivateKey);
            byte[] decryptedData = rsa.Decrypt(encryptedData, RSAEncryptionPadding.OaepSHA1);
            return Encoding.UTF8.GetString(decryptedData);
        }
        catch (Exception ex)
        {
            return string.Empty;
        }
    }
}
```