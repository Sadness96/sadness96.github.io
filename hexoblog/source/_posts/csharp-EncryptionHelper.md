---
title: 加密解密帮助类
date: 2018-01-10 11:41:20
tags: [c#,helper,aes,base64,crc32,des,folder,md5,rsa,sha1]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-DevFramework/%E5%8A%A0%E5%AF%86%E8%A7%A3%E5%AF%86%E5%B7%A5%E5%85%B7.png"/>

### 提供几种对称加密与非对称加密算法，以及单项加密与文件夹加密
<!-- more -->
#### 简介
工作中需要各种方式的加密（传输文本加密，文件加密，图片Base64编码，文件MD5与SHA1值计算），既有对称式加密与非对称式加密，也有单向加密，应用于各种使用环境。
#### 警告
由于2017年5月12日的比特币勒索病毒 [WannaCry](https://baike.baidu.com/item/WannaCry/20797421?fr=aladdin) 爆发，100多个国家和地区超过10万台电脑遭到了勒索病毒攻击、感染。其原理就是加密电脑中的文件，以用秘钥勒索比特币。技术本质并无好坏之分，多行善事。
#### 帮助类、介绍
##### 对称式加密
[对称加密算法](https://baike.baidu.com/item/%E5%AF%B9%E7%A7%B0%E5%8A%A0%E5%AF%86%E7%AE%97%E6%B3%95/211953?fr=aladdin) 解密使用相同密钥及相同算法的逆算法对密文进行解密。
###### AES
[AES](https://baike.baidu.com/item/aes/5903?fr=aladdin) 一种区块加密标准，替代原先的DES，对称密钥加密中最流行的算法之一。
[AESHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Encryption/AESHelper.cs)
``` CSharp
/// <summary>
/// AES加密
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <param name="strKey">秘钥</param>
/// <returns>AES密文</returns>
public static string AESEncrypt(string strPlaintext, string strKey)
{
    try
    {
        if (string.IsNullOrEmpty(strPlaintext))
        {
            return string.Empty;
        }
        strKey = strKey.Length < 32 ? strKey.PadRight(32, '0') : strKey.Substring(0, 32);
        Byte[] toEncryptArray = Encoding.UTF8.GetBytes(strPlaintext);
        RijndaelManaged rijndaelManaged = new RijndaelManaged
        {
            Key = Encoding.UTF8.GetBytes(strKey),
            Mode = CipherMode.ECB,
            Padding = PaddingMode.PKCS7
        };
        ICryptoTransform pCryptoTransform = rijndaelManaged.CreateEncryptor();
        Byte[] resultArray = pCryptoTransform.TransformFinalBlock(toEncryptArray, 0, toEncryptArray.Length);
        return Convert.ToBase64String(resultArray, 0, resultArray.Length);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// AES解密
/// </summary>
/// <param name="strCiphertext">AES密文</param>
/// <param name="strKey">秘钥</param>
/// <returns>明文</returns>
public static string AESDecrypt(string strCiphertext, string strKey)
{
    try
    {
        if (string.IsNullOrEmpty(strCiphertext))
        {
            return string.Empty;
        }
        strKey = strKey.Length < 32 ? strKey.PadRight(32, '0') : strKey.Substring(0, 32);
        Byte[] toEncryptArray = Convert.FromBase64String(strCiphertext);
        RijndaelManaged rijndaelManaged = new RijndaelManaged
        {
            Key = Encoding.UTF8.GetBytes(strKey),
            Mode = CipherMode.ECB,
            Padding = PaddingMode.PKCS7
        };
        ICryptoTransform pCryptoTransform = rijndaelManaged.CreateDecryptor();
        Byte[] resultArray = pCryptoTransform.TransformFinalBlock(toEncryptArray, 0, toEncryptArray.Length);
        return Encoding.UTF8.GetString(resultArray);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 文件AES加密
/// </summary>
/// <param name="strFilePath">文件路径</param>
/// <param name="strSaveFilePath">加密文件目录</param>
/// <param name="strKey">秘钥</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool FileAESEncrypt(string strFilePath, string strSaveFilePath, string strKey)
{
    try
    {
        //设置Aes秘钥和格式
        strKey = strKey.Length < 32 ? strKey.PadRight(32, '0') : strKey.Substring(0, 32);
        RijndaelManaged rijndaelManaged = new RijndaelManaged
        {
            Key = Encoding.UTF8.GetBytes(strKey),
            Mode = CipherMode.ECB,
            Padding = PaddingMode.PKCS7
        };
        //读取文本加密数据
        FileStream fileStream = File.OpenRead(strFilePath);
        byte[] byteFileStream = new byte[fileStream.Length];
        fileStream.Read(byteFileStream, 0, (int)fileStream.Length);
        fileStream.Close();
        using (var memoryStream = new MemoryStream())
        {
            using (var cryptoStream = new CryptoStream(memoryStream, rijndaelManaged.CreateEncryptor(), CryptoStreamMode.Write))
            {
                cryptoStream.Write(byteFileStream, 0, byteFileStream.Length);
                cryptoStream.FlushFinalBlock();
                fileStream = File.OpenWrite(strSaveFilePath);
                foreach (byte byteMemoryStream in memoryStream.ToArray())
                {
                    fileStream.WriteByte(byteMemoryStream);
                }
                fileStream.Close();
                cryptoStream.Close();
                memoryStream.Close();
                return true;
            }
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 文件AES解密
/// </summary>
/// <param name="strFilePath">被加密的文件路径</param>
/// <param name="strSaveFilePath">解密文件目录</param>
/// <param name="strKey">秘钥</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool FileAESDecrypt(string strFilePath, string strSaveFilePath, string strKey)
{
    try
    {
        strKey = strKey.Length < 32 ? strKey.PadRight(32, '0') : strKey.Substring(0, 32);
        RijndaelManaged rijndaelManaged = new RijndaelManaged
        {
            Key = Encoding.UTF8.GetBytes(strKey),
            Mode = CipherMode.ECB,
            Padding = PaddingMode.PKCS7
        };
        FileStream fileStream = File.OpenRead(strFilePath);
        byte[] byteFileStream = new byte[fileStream.Length];
        fileStream.Read(byteFileStream, 0, (int)fileStream.Length);
        fileStream.Close();
        using (var memoryStream = new MemoryStream())
        {
            using (var cryptoStream = new CryptoStream(memoryStream, rijndaelManaged.CreateDecryptor(), CryptoStreamMode.Write))
            {
                cryptoStream.Write(byteFileStream, 0, byteFileStream.Length);
                cryptoStream.FlushFinalBlock();
                fileStream = File.OpenWrite(strSaveFilePath);
                foreach (byte byteMemoryStream in memoryStream.ToArray())
                {
                    fileStream.WriteByte(byteMemoryStream);
                }
                fileStream.Close();
                cryptoStream.Close();
                memoryStream.Close();
                return true;
            }
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
```
###### DES
[DES](https://baike.baidu.com/item/DES) 一种使用密钥加密的块算法,1977年被美国联邦政府的国家标准局确定为联邦资料处理标准（FIPS），并授权在非密级政府通信中使用。
[DESHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Encryption/DESHelper.cs)
``` CSharp
/// <summary>
/// DES加密
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <param name="strKey">秘钥(8位)</param>
/// <param name="strIV">向量(8位)</param>
/// <returns>DES密文</returns>
public static string DESEncrypt(string strPlaintext, string strKey, string strIV)
{
    try
    {
        DESCryptoServiceProvider desCrypto = new DESCryptoServiceProvider();
        desCrypto.Key = UTF8Encoding.Default.GetBytes(strKey);
        desCrypto.IV = UTF8Encoding.UTF8.GetBytes(strIV);
        using (ICryptoTransform cryptoTransform = desCrypto.CreateEncryptor())
        {
            byte[] byteBaseUTF8 = Encoding.UTF8.GetBytes(strPlaintext);
            using (var memoryStream = new MemoryStream())
            {
                using (var cryptoStream = new CryptoStream(memoryStream, cryptoTransform, CryptoStreamMode.Write))
                {
                    cryptoStream.Write(byteBaseUTF8, 0, byteBaseUTF8.Length);
                    cryptoStream.FlushFinalBlock();
                }
                return Convert.ToBase64String(memoryStream.ToArray());
            }
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// DES解密
/// </summary>
/// <param name="strCiphertext">DES密文</param>
/// <param name="strKey">秘钥(8位)</param>
/// <param name="strIV">向量(8位)</param>
/// <returns>明文</returns>
public static string DESDecrypt(string strCiphertext, string strKey, string strIV)
{
    try
    {
        DESCryptoServiceProvider desCrypto = new DESCryptoServiceProvider();
        desCrypto.Key = UTF8Encoding.Default.GetBytes(strKey);
        desCrypto.IV = UTF8Encoding.UTF8.GetBytes(strIV);
        using (ICryptoTransform cryptoTransform = desCrypto.CreateDecryptor())
        {
            byte[] byteBase64 = Convert.FromBase64String(strCiphertext);
            using (var memoryStream = new MemoryStream())
            {
                using (var cryptoStream = new CryptoStream(memoryStream, cryptoTransform, CryptoStreamMode.Write))
                {
                    cryptoStream.Write(byteBase64, 0, byteBase64.Length);
                    cryptoStream.FlushFinalBlock();
                }
                return Encoding.UTF8.GetString(memoryStream.ToArray());
            }
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 文件DES加密
/// </summary>
/// <param name="strFilePath">文件路径</param>
/// <param name="strSaveFilePath">加密文件目录</param>
/// <param name="strKey">秘钥(8位)</param>
/// <param name="strIV">向量(8位)</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool FileDESEncrypt(string strFilePath, string strSaveFilePath, string strKey, string strIV)
{
    try
    {
        DESCryptoServiceProvider desCrypto = new DESCryptoServiceProvider();
        desCrypto.Key = UTF8Encoding.Default.GetBytes(strKey);
        desCrypto.IV = UTF8Encoding.UTF8.GetBytes(strIV);
        FileStream fileStream = File.OpenRead(strFilePath);
        byte[] byteFileStream = new byte[fileStream.Length];
        fileStream.Read(byteFileStream, 0, (int)fileStream.Length);
        fileStream.Close();
        using (var memoryStream = new MemoryStream())
        {
            using (var cryptoStream = new CryptoStream(memoryStream, desCrypto.CreateEncryptor(), CryptoStreamMode.Write))
            {
                cryptoStream.Write(byteFileStream, 0, byteFileStream.Length);
                cryptoStream.FlushFinalBlock();
                fileStream = File.OpenWrite(strSaveFilePath);
                foreach (byte byteMemoryStream in memoryStream.ToArray())
                {
                    fileStream.WriteByte(byteMemoryStream);
                }
                fileStream.Close();
                cryptoStream.Close();
                memoryStream.Close();
                return true;
            }
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 文件DES解密
/// </summary>
/// <param name="strFilePath">被加密的文件路径</param>
/// <param name="strSaveFilePath">解密文件目录</param>
/// <param name="strKey">秘钥(8位)</param>
/// <param name="strIV">向量(8位)</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool FileDESDecrypt(string strFilePath, string strSaveFilePath, string strKey, string strIV)
{
    try
    {
        DESCryptoServiceProvider desCrypto = new DESCryptoServiceProvider();
        desCrypto.Key = UTF8Encoding.Default.GetBytes(strKey);
        desCrypto.IV = UTF8Encoding.UTF8.GetBytes(strIV);
        FileStream fileStream = File.OpenRead(strFilePath);
        byte[] byteFileStream = new byte[fileStream.Length];
        fileStream.Read(byteFileStream, 0, (int)fileStream.Length);
        fileStream.Close();
        using (var memoryStream = new MemoryStream())
        {
            using (var cryptoStream = new CryptoStream(memoryStream, desCrypto.CreateDecryptor(), CryptoStreamMode.Write))
            {
                cryptoStream.Write(byteFileStream, 0, byteFileStream.Length);
                cryptoStream.FlushFinalBlock();
                fileStream = File.OpenWrite(strSaveFilePath);
                foreach (byte byteMemoryStream in memoryStream.ToArray())
                {
                    fileStream.WriteByte(byteMemoryStream);
                }
                fileStream.Close();
                cryptoStream.Close();
                memoryStream.Close();
                return true;
            }
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
```
##### 非对称式加密
[非对称加密算法](https://baike.baidu.com/item/%E9%9D%9E%E5%AF%B9%E7%A7%B0%E5%8A%A0%E5%AF%86%E7%AE%97%E6%B3%95) 需要两个密钥：公开密钥（publickey:简称公钥）和私有密钥（privatekey:简称私钥）。公钥与私钥是一对，如果用公钥对数据进行加密，只有用对应的私钥才能解密。
###### RSA
[RSA](https://baike.baidu.com/item/RSA%E7%AE%97%E6%B3%95?fromtitle=RSA&fromid=210678) 一种非对称加密算法。在公开密钥加密和电子商业中RSA被广泛使用。
[RSAHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Encryption/RSAHelper.cs) 只能使用产生出的密钥，且加密更加复杂所以只能加密短文本（测试加密长度上限为58字节）。
``` CSharp
/// <summary>
/// RSA产生秘钥
/// </summary>
/// <param name="xmlPublicKey">公钥</param>
/// <param name="xmlPrivateKey">私钥</param>
public static void RSAKey(out string xmlPublicKey, out string xmlPrivateKey)
{
    try
    {
        RSACryptoServiceProvider rsaCrypto = new RSACryptoServiceProvider();
        xmlPublicKey = rsaCrypto.ToXmlString(false);
        xmlPrivateKey = rsaCrypto.ToXmlString(true);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        xmlPublicKey = string.Empty;
        xmlPrivateKey = string.Empty;
    }
}

/// <summary>
/// RSA加密
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <param name="xmlPublicKey">公钥</param>
/// <returns>RSA密文</returns>
public static string RSAEncrypt(string strPlaintext, string xmlPublicKey)
{
    try
    {
        RSACryptoServiceProvider rsaCrypto = new RSACryptoServiceProvider();
        rsaCrypto.FromXmlString(xmlPublicKey);
        UnicodeEncoding unicodeEncoding = new UnicodeEncoding();
        byte[] byteBaseUnicode = unicodeEncoding.GetBytes(strPlaintext);
        byte[] byteBaseEncrypt = rsaCrypto.Encrypt(byteBaseUnicode, false);
        string strRSAEncrypt = Convert.ToBase64String(byteBaseEncrypt);
        return strRSAEncrypt;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// RSA解密
/// </summary>
/// <param name="strCiphertext">RSA密文</param>
/// <param name="xmlPrivateKey">私钥</param>
/// <returns>明文</returns>
public static string RSADecrypt(string strCiphertext, string xmlPrivateKey)
{
    try
    {
        RSACryptoServiceProvider rsaCrypto = new RSACryptoServiceProvider();
        rsaCrypto.FromXmlString(xmlPrivateKey);
        byte[] byteBase64 = Convert.FromBase64String(strCiphertext);
        byte[] byteBaseDecrypt = rsaCrypto.Decrypt(byteBase64, false);
        string strRSADecrypt = (new UnicodeEncoding()).GetString(byteBaseDecrypt);
        return strRSADecrypt;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}
```
##### 单项加密
[密码散列函数](https://baike.baidu.com/item/%E5%AF%86%E7%A0%81%E6%95%A3%E5%88%97%E5%87%BD%E6%95%B0) 一种单向函数，也就是说极其难以由散列函数输出的结果，回推输入的数据是什么。多用于文件效验完整性。
###### MD5
[MD5](https://baike.baidu.com/item/MD5) 一种被广泛使用的密码散列函数，可以产生出一个128位（16字节）的散列值（hash value），用于确保信息传输完整一致。
[MD5Helper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Encryption/MD5Helper.cs)
``` CSharp
/// <summary>
/// MD5加密(16位小写)
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <returns>MD5密文(16位小写)</returns>
public static string MD5Encrypt_16Lower(string strPlaintext)
{
    try
    {
        MD5CryptoServiceProvider md5Crypto = new MD5CryptoServiceProvider();
        string strCiphertext = BitConverter.ToString(md5Crypto.ComputeHash(UTF8Encoding.Default.GetBytes(strPlaintext)), 4, 8);
        strCiphertext = strCiphertext.Replace("-", "");
        strCiphertext = strCiphertext.ToLower();
        return strCiphertext;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// MD5加密(16位大写)
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <returns>MD5密文(16位小写)</returns>
public static string MD5Encrypt_16Upper(string strPlaintext)
{
    try
    {
        MD5CryptoServiceProvider md5Crypto = new MD5CryptoServiceProvider();
        string strCiphertext = BitConverter.ToString(md5Crypto.ComputeHash(UTF8Encoding.Default.GetBytes(strPlaintext)), 4, 8);
        strCiphertext = strCiphertext.Replace("-", "");
        strCiphertext = strCiphertext.ToUpper();
        return strCiphertext;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// MD5加密(32位小写)
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <returns>MD5密文(32位小写)</returns>
public static string MD5Encrypt_32Lower(string strPlaintext)
{
    try
    {
        MD5CryptoServiceProvider md5Crypto = new MD5CryptoServiceProvider();
        string strCiphertext = BitConverter.ToString(md5Crypto.ComputeHash(UTF8Encoding.Default.GetBytes(strPlaintext)));
        strCiphertext = strCiphertext.Replace("-", "");
        strCiphertext = strCiphertext.ToLower();
        return strCiphertext;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// MD5加密(32位大写)
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <returns>MD5密文(32位小写)</returns>
public static string MD5Encrypt_32Upper(string strPlaintext)
{
    try
    {
        MD5CryptoServiceProvider md5Crypto = new MD5CryptoServiceProvider();
        string strCiphertext = BitConverter.ToString(md5Crypto.ComputeHash(UTF8Encoding.Default.GetBytes(strPlaintext)));
        strCiphertext = strCiphertext.Replace("-", "");
        strCiphertext = strCiphertext.ToUpper();
        return strCiphertext;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 获取文件MD5值(32位小写)
/// </summary>
/// <param name="strFilePath">文件路径</param>
/// <returns>文件MD5值(32位小写)</returns>
public static string FileMD5Encrypt_32Lower(string strFilePath)
{
    try
    {
        FileStream fileStream = new FileStream(strFilePath, FileMode.Open, FileAccess.Read);
        System.Security.Cryptography.MD5 md5 = new System.Security.Cryptography.MD5CryptoServiceProvider();
        byte[] byteHash = md5.ComputeHash(fileStream);
        fileStream.Close();
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < byteHash.Length; i++)
        {
            stringBuilder.Append(byteHash[i].ToString("x2"));
        }
        return stringBuilder.ToString();
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 获取文件MD5值(32位大写)
/// </summary>
/// <param name="strFilePath">文件路径</param>
/// <returns>文件MD5值(32位大写)</returns>
public static string FileMD5Encrypt_32Upper(string strFilePath)
{
    try
    {
        FileStream fileStream = new FileStream(strFilePath, FileMode.Open, FileAccess.Read);
        System.Security.Cryptography.MD5 md5 = new System.Security.Cryptography.MD5CryptoServiceProvider();
        byte[] byteHash = md5.ComputeHash(fileStream);
        fileStream.Close();
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < byteHash.Length; i++)
        {
            stringBuilder.Append(byteHash[i].ToString("X2"));
        }
        return stringBuilder.ToString();
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}
```
###### SHA1
[SHA1](https://baike.baidu.com/item/SHA1) 安全哈希算法（Secure Hash Algorithm）主要适用于数字签名标准 （Digital Signature Standard DSS）里面定义的数字签名算法（Digital Signature Algorithm DSA）。
[SHA1Helper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Encryption/SHA1Helper.cs)
``` CSharp
/// <summary>
/// SHA1加密(40位小写)
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <returns>SHA1密文(40位小写)</returns>
public static string SHA1Encrypt_40Lower(string strPlaintext)
{
    try
    {
        System.Security.Cryptography.SHA1 sha1Crypto = new SHA1CryptoServiceProvider();
        byte[] bytes_sha1_in = UTF8Encoding.Default.GetBytes(strPlaintext);
        byte[] bytes_sha1_out = sha1Crypto.ComputeHash(bytes_sha1_in);
        string str_sha1_out = BitConverter.ToString(bytes_sha1_out);
        str_sha1_out = str_sha1_out.Replace("-", "");
        str_sha1_out = str_sha1_out.ToLower();
        return str_sha1_out;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// SHA1加密(40位大写)
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <returns>SHA1密文(40位大写)</returns>
public static string SHA1Encrypt_40Upper(string strPlaintext)
{
    try
    {
        System.Security.Cryptography.SHA1 sha1Crypto = new SHA1CryptoServiceProvider();
        byte[] bytes_sha1_in = UTF8Encoding.Default.GetBytes(strPlaintext);
        byte[] bytes_sha1_out = sha1Crypto.ComputeHash(bytes_sha1_in);
        string str_sha1_out = BitConverter.ToString(bytes_sha1_out);
        str_sha1_out = str_sha1_out.Replace("-", "");
        str_sha1_out = str_sha1_out.ToUpper();
        return str_sha1_out;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 获取文件SHA1值(40位小写)
/// </summary>
/// <param name="strFilePath">文件路径</param>
/// <returns>文件SHA1值(40位小写)</returns>
public static string FileSHA1Encrypt_40Lower(string strFilePath)
{
    try
    {
        FileStream fileStream = new FileStream(strFilePath, FileMode.Open, FileAccess.Read);
        System.Security.Cryptography.SHA1 sha1 = new SHA1CryptoServiceProvider();
        byte[] byteHash = sha1.ComputeHash(fileStream);
        fileStream.Close();
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < byteHash.Length; i++)
        {
            stringBuilder.Append(byteHash[i].ToString("x2"));
        }
        return stringBuilder.ToString();
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 获取文件SHA1值(40位大写)
/// </summary>
/// <param name="strFilePath">文件路径</param>
/// <returns>文件SHA1值(40位大写)</returns>
public static string FileSHA1Encrypt_40Upper(string strFilePath)
{
    try
    {
        FileStream fileStream = new FileStream(strFilePath, FileMode.Open, FileAccess.Read);
        System.Security.Cryptography.SHA1 sha1 = new SHA1CryptoServiceProvider();
        byte[] byteHash = sha1.ComputeHash(fileStream);
        fileStream.Close();
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < byteHash.Length; i++)
        {
            stringBuilder.Append(byteHash[i].ToString("X2"));
        }
        return stringBuilder.ToString();
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}
```
###### CRC32
[CRC32](https://baike.baidu.com/item/CRC32) 循环冗余校验。在数据存储和数据通讯领域，为了保证数据的正确，就不得不采用检错的手段。在诸多检错手段中，CRC是最著名的一种。
[CRC32Helper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Encryption/CRC32Helper.cs)
``` CSharp
/// <summary>
/// CRC32加密(8位小写)
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <returns>CRC32密文(8位小写)</returns>
public static string CRC32Encrypt_8Lower(string strPlaintext)
{
    try
    {
        Crc32 crc32Crypto = new Crc32();
        byte[] bytes_crc32_in = UTF8Encoding.Default.GetBytes(strPlaintext);
        byte[] bytes_crc32_out = crc32Crypto.ComputeHash(bytes_crc32_in);
        string str_crc32_out = BitConverter.ToString(bytes_crc32_out);
        str_crc32_out = str_crc32_out.Replace("-", "");
        str_crc32_out = str_crc32_out.ToLower();
        return str_crc32_out;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// CRC32加密(8位大写)
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <returns>CRC32密文(8位大写)</returns>
public static string CRC32Encrypt_8Upper(string strPlaintext)
{
    try
    {
        Crc32 crc32Crypto = new Crc32();
        byte[] bytes_crc32_in = UTF8Encoding.Default.GetBytes(strPlaintext);
        byte[] bytes_crc32_out = crc32Crypto.ComputeHash(bytes_crc32_in);
        string str_crc32_out = BitConverter.ToString(bytes_crc32_out);
        str_crc32_out = str_crc32_out.Replace("-", "");
        str_crc32_out = str_crc32_out.ToUpper();
        return str_crc32_out;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 获取文件CRC32值(8位小写)
/// </summary>
/// <param name="strFilePath">文件路径</param>
/// <returns>文件CRC32值(8位小写)</returns>
public static string FileCRC32Encrypt_8Lower(string strFilePath)
{
    try
    {
        String hashCRC32 = String.Empty;
        //检查文件是否存在,如果文件存在则进行计算,否则返回空值
        if (System.IO.File.Exists(strFilePath))
        {
            using (System.IO.FileStream fileStream = new System.IO.FileStream(strFilePath, System.IO.FileMode.Open, System.IO.FileAccess.Read))
            {
                //计算文件的CSC32值
                Crc32 calculator = new Crc32();
                Byte[] buffer = calculator.ComputeHash(fileStream);
                calculator.Clear();
                //将字节数组转换成十六进制的字符串形式
                StringBuilder stringBuilder = new StringBuilder();
                for (int i = 0; i < buffer.Length; i++)
                {
                    stringBuilder.Append(buffer[i].ToString("x2"));
                }
                hashCRC32 = stringBuilder.ToString();
            }
        }
        return hashCRC32;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 获取文件CRC32值(8位大写)
/// </summary>
/// <param name="strFilePath">文件路径</param>
/// <returns>文件CRC32值(8位大写)</returns>
public static string FileCRC32Encrypt_8Upper(string strFilePath)
{
    try
    {
        String hashCRC32 = String.Empty;
        //检查文件是否存在,如果文件存在则进行计算,否则返回空值
        if (System.IO.File.Exists(strFilePath))
        {
            using (System.IO.FileStream fileStream = new System.IO.FileStream(strFilePath, System.IO.FileMode.Open, System.IO.FileAccess.Read))
            {
                //计算文件的CSC32值
                Crc32 calculator = new Crc32();
                Byte[] buffer = calculator.ComputeHash(fileStream);
                calculator.Clear();
                //将字节数组转换成十六进制的字符串形式
                StringBuilder stringBuilder = new StringBuilder();
                for (int i = 0; i < buffer.Length; i++)
                {
                    stringBuilder.Append(buffer[i].ToString("X2"));
                }
                hashCRC32 = stringBuilder.ToString();
            }
        }
        return hashCRC32;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 提供 CRC32 算法的实现
/// </summary>
public class Crc32 : System.Security.Cryptography.HashAlgorithm
{
    /// <summary>
    /// Default Polynomial
    /// </summary>
    public const UInt32 DefaultPolynomial = 0xedb88320;
    /// <summary>
    /// Default Seed
    /// </summary>
    public const UInt32 DefaultSeed = 0xffffffff;
    private UInt32 hash;
    private UInt32 seed;
    private UInt32[] table;
    private static UInt32[] defaultTable;

    /// <summary>
    /// Crc32
    /// </summary>
    public Crc32()
    {
        table = InitializeTable(DefaultPolynomial);
        seed = DefaultSeed;
        Initialize();
    }

    /// <summary>
    /// Crc32
    /// </summary>
    /// <param name="polynomial"></param>
    /// <param name="seed"></param>
    public Crc32(UInt32 polynomial, UInt32 seed)
    {
        table = InitializeTable(polynomial);
        this.seed = seed;
        Initialize();
    }

    /// <summary>
    /// 初始化
    /// </summary>
    public override void Initialize()
    {
        hash = seed;
    }

    /// <summary>
    /// Hash Core
    /// </summary>
    /// <param name="buffer"></param>
    /// <param name="start"></param>
    /// <param name="length"></param>
    protected override void HashCore(byte[] buffer, int start, int length)
    {
        hash = CalculateHash(table, hash, buffer, start, length);
    }

    /// <summary>
    /// Hash Final
    /// </summary>
    /// <returns></returns>
    protected override byte[] HashFinal()
    {
        byte[] hashBuffer = UInt32ToBigEndianBytes(~hash);
        this.HashValue = hashBuffer;
        return hashBuffer;
    }

    /// <summary>
    /// Compute
    /// </summary>
    /// <param name="buffer"></param>
    /// <returns></returns>
    public static UInt32 Compute(byte[] buffer)
    {
        return ~CalculateHash(InitializeTable(DefaultPolynomial), DefaultSeed, buffer, 0, buffer.Length);
    }

    /// <summary>
    /// Compute
    /// </summary>
    /// <param name="seed"></param>
    /// <param name="buffer"></param>
    /// <returns></returns>
    public static UInt32 Compute(UInt32 seed, byte[] buffer)
    {
        return ~CalculateHash(InitializeTable(DefaultPolynomial), seed, buffer, 0, buffer.Length);
    }

    /// <summary>
    /// Compute
    /// </summary>
    /// <param name="polynomial"></param>
    /// <param name="seed"></param>
    /// <param name="buffer"></param>
    /// <returns></returns>
    public static UInt32 Compute(UInt32 polynomial, UInt32 seed, byte[] buffer)
    {
        return ~CalculateHash(InitializeTable(polynomial), seed, buffer, 0, buffer.Length);
    }

    private static UInt32[] InitializeTable(UInt32 polynomial)
    {
        if (polynomial == DefaultPolynomial && defaultTable != null)
        {
            return defaultTable;
        }
        UInt32[] createTable = new UInt32[256];
        for (int i = 0; i < 256; i++)
        {
            UInt32 entry = (UInt32)i;
            for (int j = 0; j < 8; j++)
            {
                if ((entry & 1) == 1)
                    entry = (entry >> 1) ^ polynomial;
                else
                    entry = entry >> 1;
            }
            createTable[i] = entry;
        }
        if (polynomial == DefaultPolynomial)
        {
            defaultTable = createTable;
        }
        return createTable;
    }
    private static UInt32 CalculateHash(UInt32[] table, UInt32 seed, byte[] buffer, int start, int size)
    {
        UInt32 crc = seed;
        for (int i = start; i < size; i++)
        {
            unchecked
            {
                crc = (crc >> 8) ^ table[buffer[i] ^ crc & 0xff];
            }
        }
        return crc;
    }
    private byte[] UInt32ToBigEndianBytes(UInt32 x)
    {
        return new byte[] { (byte)((x >> 24) & 0xff), (byte)((x >> 16) & 0xff), (byte)((x >> 8) & 0xff), (byte)(x & 0xff) };
    }
}
```
##### 其它方式（Base64）
###### Base64
[Base64](https://baike.baidu.com/item/base64) 网络上最常见的用于传输8Bit字节码的编码方式之一，准确的来说Base64不属于加密范围，仅是一种基于64个可打印字符来表示二进制数据的方法。多用于图片传输使用。
[Base64Helper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Encryption/Base64Helper.cs)
``` CSharp
/// <summary>
/// Base64加密
/// </summary>
/// <param name="strPlaintext">明文</param>
/// <returns>Base64密文</returns>
public static string Base64Encrypt(string strPlaintext)
{
    try
    {
        byte[] bytes = Encoding.UTF8.GetBytes(strPlaintext);
        return Convert.ToBase64String(bytes);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// Base64解密
/// </summary>
/// <param name="strCiphertext">Base64密文</param>
/// <returns>明文</returns>
public static string Base64Decrypt(string strCiphertext)
{
    try
    {
        byte[] bytes = Convert.FromBase64String(strCiphertext);
        return Encoding.UTF8.GetString(bytes);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 图片Base64加密
/// </summary>
/// <param name="strImagePath">图片路径</param>
/// <param name="imageFormat">指定图像格式</param>
/// <returns>Base64密文</returns>
public static string ImageBase64Encrypt(string strImagePath, ImageFormat imageFormat)
{
    try
    {
        MemoryStream memoryStream = new MemoryStream();
        Bitmap bitmap = new Bitmap(strImagePath);
        if (imageFormat == null)
        {
            imageFormat = GetImageFormatFromPath(strImagePath);
        }
        bitmap.Save(memoryStream, imageFormat);
        byte[] bytes = memoryStream.GetBuffer();
        return Convert.ToBase64String(bytes);
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return string.Empty;
    }
}

/// <summary>
/// 图片Base64解密
/// </summary>
/// <param name="strCiphertext">Base64密文</param>
/// <param name="strSaveFilePath">解密图片目录</param>
/// <param name="imageFormat">指定图像格式</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool ImageBase64Decrypt(string strCiphertext, string strSaveFilePath, ImageFormat imageFormat)
{
    try
    {
        byte[] bytes = Convert.FromBase64String(strCiphertext);
        MemoryStream memoryStream = new MemoryStream(bytes);
        Bitmap bitmap = new Bitmap(memoryStream);
        if (imageFormat == null)
        {
            imageFormat = GetImageFormatFromPath(strSaveFilePath);
        }
        bitmap.Save(strSaveFilePath, imageFormat);
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 根据图片路径获得图片格式(缺少MemoryBmp)
/// </summary>
/// <param name="strImagePath">图片路径</param>
/// <returns>图片格式</returns>
public static ImageFormat GetImageFormatFromPath(string strImagePath)
{
    try
    {
        string strImageExtension = Path.GetExtension(strImagePath).ToLower();
        if (string.IsNullOrEmpty(strImageExtension))
        {
            return null;
        }
        else
        {
            if (strImageExtension.Equals(".bmp") || strImageExtension.Equals(".rle") || strImageExtension.Equals(".dlb"))
            {
                return ImageFormat.Bmp;
            }
            else if (strImageExtension.Equals(".emf"))
            {
                return ImageFormat.Emf;
            }
            else if (strImageExtension.Equals(".exif"))
            {
                return ImageFormat.Exif;
            }
            else if (strImageExtension.Equals(".gif"))
            {
                return ImageFormat.Gif;
            }
            else if (strImageExtension.Equals(".ico"))
            {
                return ImageFormat.Icon;
            }
            else if (strImageExtension.Equals(".jpg") || strImageExtension.Equals(".jpeg") || strImageExtension.Equals(".jpe"))
            {
                return ImageFormat.Jpeg;
            }
            else if (strImageExtension.Equals(".png") || strImageExtension.Equals(".pns"))
            {
                return ImageFormat.Png;
            }
            else if (strImageExtension.Equals(".tif") || strImageExtension.Equals(".tiff"))
            {
                return ImageFormat.Tiff;
            }
            else if (strImageExtension.Equals(".wmf"))
            {
                return ImageFormat.Wmf;
            }
            else
            {
                return null;
            }
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}
```
##### 文件夹加密
起因是一个女性朋友提出的需求，大概是说不想自己电脑的文件夹被其他人打开，但是又不想加密文件，因为耗时会很久，只想加密文件夹。查过各种资料后没有找到更好的方法，暂且使用一种修改文件夹后缀名达到让电脑识别为控制面板或回收站等图标的方式，然后修改恢复文件夹的时候预设匹对一个设定好的密码文件，就可以达到加密解密文件夹的效果，但是理解原理的人是可以直接破解的，但是我相信理解这项技术的不多，并且不会对每一个系统图标虎视眈眈的。
[FolderHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/Utils.Helper/Encryption/FolderHelper.cs)
``` CSharp
/// <summary>
/// 加密文件
/// </summary>
public static string Lock = ".{2559a1f2-21d7-11d4-bdaf-00c04f60b9f0}";
/// <summary>
/// 控制面板
/// </summary>
public static string Control = ".{21EC2020-3AEA-1069-A2DD-08002B30309D}";
/// <summary>
/// RunIE
/// </summary>
public static string RunIE = ".{2559a1f4-21d7-11d4-bdaf-00c04f60b9f0}";
/// <summary>
/// 回收站
/// </summary>
public static string Recycle = ".{645FF040-5081-101B-9F08-00AA002F954E}";
/// <summary>
/// Help
/// </summary>
public static string Help = ".{2559a1f1-21d7-11d4-bdaf-00c04f60b9f0}";
/// <summary>
/// NetWork
/// </summary>
public static string NetWork = ".{7007ACC7-3202-11D1-AAD2-00805FC1270E}";

/// <summary>
/// 文件夹加密(可破解)
/// </summary>
/// <param name="strFolderPath">文件夹路径</param>
/// <param name="strClsid">Clsid类型</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool FolderEncrypt(string strFolderPath, string strClsid)
{
    try
    {
        DirectoryInfo directoryInfo = new DirectoryInfo(strFolderPath);
        directoryInfo.MoveTo(directoryInfo.Parent.FullName + "\\" + directoryInfo.Name + strClsid);
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 文件夹解密
/// 理论上可以解密所有该方法加密的文件夹
/// </summary>
/// <param name="strFolderPath">文件夹路径</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool FolderDecrypt(string strFolderPath)
{
    try
    {
        DirectoryInfo directoryInfo = new DirectoryInfo(strFolderPath);
        directoryInfo.MoveTo(strFolderPath.Substring(0, strFolderPath.LastIndexOf(".")));
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 文件夹加密(带密码)(可破解)
/// </summary>
/// <param name="strFolderPath">文件夹路径</param>
/// <param name="strClsid">Clsid类型</param>
/// <param name="strPassword">加密密码</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool FolderEncrypt(string strFolderPath, string strClsid, string strPassword)
{
    try
    {
        DirectoryInfo directoryInfo = new DirectoryInfo(strFolderPath);
        XmlDocument xmlDocument = new XmlDocument();
        XmlNode xmlNode = xmlDocument.CreateNode(XmlNodeType.XmlDeclaration, "", "");
        xmlDocument.AppendChild(xmlNode);
        XmlElement xmlElement = xmlDocument.CreateElement("", "ROOT", "");
        XmlText xmlText = xmlDocument.CreateTextNode(strPassword);
        xmlElement.AppendChild(xmlText);
        xmlDocument.AppendChild(xmlElement);
        xmlDocument.Save(strFolderPath + "\\Lock.xml");
        directoryInfo.MoveTo(directoryInfo.Parent.FullName + "\\" + directoryInfo.Name + strClsid);
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}

/// <summary>
/// 文件夹解密(带密码)
/// </summary>
/// <param name="strFolderPath">文件夹路径</param>
/// <param name="strPassword">加密密码</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool FolderDecrypt(string strFolderPath, string strPassword)
{
    try
    {
        DirectoryInfo directoryInfo = new DirectoryInfo(strFolderPath);
        bool bIsPassword = false;
        XmlTextReader xmlTextReader = new XmlTextReader(strFolderPath + "\\Lock.xml");
        while (xmlTextReader.Read())
        {
            if (xmlTextReader.NodeType == XmlNodeType.Text)
            {
                if (xmlTextReader.Value == strPassword)
                {
                    bIsPassword = true;
                    break;
                }
            }
        }
        xmlTextReader.Close();
        if (bIsPassword)
        {
            File.Delete(strFolderPath + "\\Lock.xml");
            directoryInfo.MoveTo(strFolderPath.Substring(0, strFolderPath.LastIndexOf(".")));
            return true;
        }
        else
        {
            return false;
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
```