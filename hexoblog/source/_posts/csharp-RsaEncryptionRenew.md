---
title: c# RSA 加密解密帮助类更新
date: 2023-06-25 22:51:20
tags: [c#,helper,rsa]
categories: C#.Net
---
### 更新 RSA 帮助类解决加密文本长度
<!-- more -->
### 简介
由于旧版本 [RSA 帮助类](https://sadness96.github.io/blog/2018/01/10/csharp-EncryptionHelper/#RSA) 对加密的文字长度有限制，所以更新代码以加密更长的字符。

### 代码
#### RSA 库
依旧是基于 System.Security.Cryptography.RSA 库

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

#### Portable.BouncyCastle 库
由于 System.Security.Cryptography.RSA 库只能使用自身生成的 xml 公钥与私钥，而想使用 OpenSSL 生成的 pem 公钥与私钥，则无法读取，使用 Portable.BouncyCastle 库可以读取编解码，但是实测似乎与 [C++ OpenSSL](https://sadness96.github.io/blog/2023/03/21/cpp-RsaOpenSSL/) 无法互相加密解密。

``` CSharp
/// <summary>
/// 基于 BouncyCastle 库加密解密 RSA
/// </summary>
public class BouncyCastleHelper
{
    /// <summary>
    /// 读取 OpenSSL 生成的私钥文件 private.pem 为 AsymmetricKeyParameter
    /// </summary>
    /// <param name="privateKeyFilePath">OpenSSL 生成 RSA 私钥路径</param>
    /// <returns></returns>
    public static AsymmetricKeyParameter ReadPrivateKeyFile(string privateKeyFilePath)
    {
        using var reader = File.OpenText(privateKeyFilePath);
        var pemReader = new PemReader(reader);
        var keyObject = pemReader.ReadObject();

        if (keyObject is AsymmetricCipherKeyPair keyPair)
        {
            return keyPair.Private;
        }
        else if (keyObject is AsymmetricKeyParameter keyParam)
        {
            return keyParam;
        }
        else
        {
            throw new InvalidOperationException("Invalid private key format.");
        }
    }

    /// <summary>
    /// 读取 OpenSSL 生成的公钥文件 public.pem 为 AsymmetricKeyParameter
    /// </summary>
    /// <param name="publicKeyFilePath">OpenSSL 生成 RSA 公钥路径</param>
    /// <returns></returns>
    public static AsymmetricKeyParameter ReadPublicKeyFile(string publicKeyFilePath)
    {
        using var reader = File.OpenText(publicKeyFilePath);
        var pemReader = new PemReader(reader);
        var keyObject = pemReader.ReadObject();

        if (keyObject is AsymmetricKeyParameter keyParam)
        {
            return keyParam;
        }
        else
        {
            throw new InvalidOperationException("Invalid public key format.");
        }
    }

    /// <summary>
    /// 读取 OpenSSL 生成的私钥文件为 AsymmetricKeyParameter
    /// </summary>
    /// <param name="privateKey">OpenSSL 生成 RSA 私钥</param>
    /// <returns></returns>
    public static AsymmetricKeyParameter ReadPrivateKey(string privateKey)
    {
        StringReader stringReader = new StringReader(privateKey);
        var pemReader = new PemReader(stringReader);
        var keyObject = pemReader.ReadObject();

        if (keyObject is AsymmetricCipherKeyPair keyPair)
        {
            return keyPair.Private;
        }
        else if (keyObject is AsymmetricKeyParameter keyParam)
        {
            return keyParam;
        }
        else
        {
            throw new InvalidOperationException("Invalid private key format.");
        }
    }

    /// <summary>
    /// 读取 OpenSSL 生成的公钥为 AsymmetricKeyParameter
    /// </summary>
    /// <param name="publicKey">OpenSSL 生成 RSA 公钥</param>
    /// <returns></returns>
    public static AsymmetricKeyParameter ReadPublicKey(string publicKey)
    {
        StringReader stringReader = new StringReader(publicKey);
        var pemReader = new PemReader(stringReader);
        var keyObject = pemReader.ReadObject();

        if (keyObject is AsymmetricKeyParameter keyParam)
        {
            return keyParam;
        }
        else
        {
            throw new InvalidOperationException("Invalid public key format.");
        }
    }

    /// <summary>
    /// 根据 AsymmetricKeyParameter 生成 System.Security.Cryptography.RSA 秘钥
    /// </summary>
    /// <param name="xmlPublicKey">公钥</param>
    /// <param name="xmlPrivateKey">私钥</param>
    public static void GenerateRSAKeys(AsymmetricKeyParameter rsaKey, out string xmlPublicKey, out string xmlPrivateKey)
    {
        try
        {
            if (rsaKey.IsPrivate)
            {
                using RSA rsa = RSA.Create(DotNetUtilities.ToRSAParameters((RsaPrivateCrtKeyParameters)rsaKey));
                xmlPublicKey = rsa.ToXmlString(false);
                xmlPrivateKey = rsa.ToXmlString(true);
            }
            else
            {
                using RSA rsa = RSA.Create(DotNetUtilities.ToRSAParameters((RsaKeyParameters)rsaKey));
                xmlPublicKey = rsa.ToXmlString(false);
                xmlPrivateKey = string.Empty;
            }
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
    /// <param name="publicKey">公钥</param>
    /// <param name="strPlaintext">明文</param>
    /// <returns>RSA密文</returns>
    public static string RSAEncrypt(AsymmetricKeyParameter publicKey, string strPlaintext)
    {
        try
        {
            byte[] inputData = Encoding.UTF8.GetBytes(strPlaintext);
            IAsymmetricBlockCipher cipher = new RsaEngine();
            cipher.Init(true, publicKey);
            byte[] encryptedData = cipher.ProcessBlock(inputData, 0, inputData.Length);
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
    /// <param name="privateKey">私钥</param>
    /// <param name="strCiphertext">RSA密文</param>
    /// <returns>明文</returns>
    public static string RSADecrypt(AsymmetricKeyParameter privateKey, string strCiphertext)
    {
        try
        {
            byte[] encryptedData = Convert.FromBase64String(strCiphertext);
            IAsymmetricBlockCipher cipher = new RsaEngine();
            cipher.Init(false, privateKey);
            byte[] decryptedData = cipher.ProcessBlock(encryptedData, 0, encryptedData.Length);
            return Encoding.UTF8.GetString(decryptedData);
        }
        catch (Exception ex)
        {
            return string.Empty;
        }
    }
}
```