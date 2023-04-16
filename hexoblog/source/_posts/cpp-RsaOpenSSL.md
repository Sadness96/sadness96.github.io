---
title: 使用 OpenSSL RSA 加密解密
date: 2023-03-21 19:27:56
tags: [c++,openssl,rsa,chatgpt]
categories: C++
---
### ChatGPT 生成 RSA 加密解密代码
<!-- more -->
### 简介
[RSA](https://baike.baidu.com/item/RSA%E7%AE%97%E6%B3%95?fromtitle=RSA&fromid=210678) 一种非对称加密算法。在公开密钥加密和电子商业中RSA被广泛使用。
c# 中的使用方法参考 [加密解密帮助类](https://sadness96.github.io/blog/2018/01/10/csharp-EncryptionHelper/)，c++ 中的普遍做法是调用 [OpenSSL](https://www.openssl.org/) 库，此文中代码都由 [ChatGPT](https://openai.com/) 生成，没有一行是我写的。ChatGPT 是当下主流的人工智能，尝试一下面向 ChatGPT 开发。

### 安装环境
[OpenSSL](https://www.openssl.org/) 官网只提供源代码，不提供编译好的二进制版本，所以使用第三方 [Shining Light Productions](https://slproweb.com/) 编译的版本 [Win64OpenSSL-1_1_1t](https://slproweb.com/download/Win64OpenSSL-1_1_1t.exe)。
配置好环境变量，使用 openssl 命令尝试生成 RSA 公钥与私钥
``` cmd
:: 生成一个 2048 位的 RSA 私钥，并将其保存到名为 "private.key" 的文件中。
openssl genrsa -out private.key 2048
:: 从私钥中提取公钥，并将其保存到名为 "public.key" 的文件中。
openssl rsa -in private.key -pubout -out public.key
```
### 创建项目
1. 创建 C++ 项目
1. 在包含目录中添加 "C:\Program Files\OpenSSL-Win64\include"
1. 在库目录中添加 "C:\Program Files\OpenSSL-Win64\lib"
1. 附加依赖项中添加 libcrypto.lib、libssl.lib

### 代码
``` cpp
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/bio.h>
#include <openssl/rand.h>
#include <openssl/err.h>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <iostream>
#include <string>

using namespace std;

// 加载随机数种子文件或使用系统 API 获取随机数据
void load_random_seed() {
#ifdef _WIN32
	// Windows 下使用 CryptGenRandom() 函数获取随机数据
	HCRYPTPROV hProv;
	BYTE pbData[32];
	DWORD dwDataLen = sizeof(pbData);
	if (CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
		CryptGenRandom(hProv, dwDataLen, pbData);
		CryptReleaseContext(hProv, 0);
	}
	RAND_seed(pbData, dwDataLen);
#else
	// Linux 下读取 /dev/urandom 文件获取随机数据
	FILE* urand = fopen("/dev/urandom", "rb");
	if (urand != NULL) {
		unsigned char buf[32];
		size_t n = fread(buf, sizeof(buf), 1, urand);
		fclose(urand);
		if (n == 1) {
			RAND_seed(buf, sizeof(buf));
		}
	}
#endif
}

// 生成 RSA PKCS#1
void RSA_generate_pkcs1(const int keyLength, string& publicPem, string& privatePem)
{
	// 生成 RSA 密钥对
	RSA* keypair = RSA_generate_key(keyLength, RSA_F4, nullptr, nullptr);

	// 获取 PKCS#1 格式公钥并输出到控制台
	BIO* bio_pubkey = BIO_new(BIO_s_mem());
	PEM_write_bio_RSAPublicKey(bio_pubkey, keypair);
	char* pubkey_str;
	long pubkey_len = BIO_get_mem_data(bio_pubkey, &pubkey_str);
	publicPem = string(pubkey_str, pubkey_len);

	// 获取 PKCS#1 格式私钥并输出到控制台
	BIO* bio_privkey = BIO_new(BIO_s_mem());
	PEM_write_bio_RSAPrivateKey(bio_privkey, keypair, nullptr, nullptr, 0, nullptr, nullptr);
	char* privkey_str;
	long privkey_len = BIO_get_mem_data(bio_privkey, &privkey_str);
	privatePem = string(privkey_str, privkey_len);

	// 释放资源
	RSA_free(keypair);
	BIO_free_all(bio_pubkey);
	BIO_free_all(bio_privkey);
}

// // 生成 RSA PKCS#8
void RSA_generate_pkcs8(const int keyLength, string& publicPem, string& privatePem)
{
	// 生成 RSA 密钥对
	RSA* keypair = RSA_generate_key(keyLength, RSA_F4, nullptr, nullptr);

	// 将 RSA 私钥转换为 EVP 私钥对象
	EVP_PKEY* pkey = EVP_PKEY_new();
	EVP_PKEY_set1_RSA(pkey, keypair);

	// 获取 PKCS#8 格式公钥并输出到控制台
	BIO* bio_pubkey = BIO_new(BIO_s_mem());
	PEM_write_bio_PUBKEY(bio_pubkey, pkey);
	char* pubkey_str;
	long pubkey_len = BIO_get_mem_data(bio_pubkey, &pubkey_str);
	publicPem = string(pubkey_str, pubkey_len);

	// 获取 PKCS#8 格式私钥并输出到控制台
	BIO* bio_privkey = BIO_new(BIO_s_mem());
	PEM_write_bio_PKCS8PrivateKey(bio_privkey, pkey, nullptr, nullptr, 0, nullptr, nullptr);
	char* privkey_str;
	long privkey_len = BIO_get_mem_data(bio_privkey, &privkey_str);
	privatePem = string(privkey_str, privkey_len);

	// 释放资源
	RSA_free(keypair);
	EVP_PKEY_free(pkey);
	BIO_free_all(bio_pubkey);
	BIO_free_all(bio_privkey);
}

// 打印错误信息
void print_openssl_error() {
	unsigned long err = ERR_get_error();
	char err_msg[120];
	ERR_error_string_n(err, err_msg, sizeof(err_msg));
	printf("Error: %s\n", err_msg);
}

// 加载公钥
RSA* load_public_key(const char* public_key_str) {
	RSA* rsa = NULL;
	BIO* bio = BIO_new_mem_buf(public_key_str, -1);
	if (bio != NULL) {
		rsa = PEM_read_bio_RSA_PUBKEY(bio, NULL, NULL, NULL);
		BIO_free(bio);
		if (rsa == NULL) {
			print_openssl_error();
		}
	}
	else {
		print_openssl_error();
	}
	return rsa;
}

// 加载私钥
RSA* load_private_key(const char* private_key_str) {
	RSA* rsa = NULL;
	BIO* bio = BIO_new_mem_buf(private_key_str, -1);
	if (bio != NULL) {
		rsa = PEM_read_bio_RSAPrivateKey(bio, NULL, NULL, NULL);
		BIO_free(bio);
		if (rsa == NULL) {
			print_openssl_error();
		}
	}
	else {
		print_openssl_error();
	}
	return rsa;
}

// Encrypt plaintext with RSA public key
std::string rsaEncrypt(const std::string& plaintext, RSA* rsa)
{
	std::string result;
	int rsaLen = RSA_size(rsa);
	char* rsaResult = new char[rsaLen];
	int encryptSize = RSA_public_encrypt(plaintext.length(), (const unsigned char*)plaintext.c_str(), (unsigned char*)rsaResult, rsa, RSA_PKCS1_PADDING);

	if (encryptSize == -1) {
		std::cerr << "Failed to encrypt data (Error code: " << ERR_get_error() << ")" << std::endl;
		delete[] rsaResult;
		return result;
	}

	result.assign(rsaResult, encryptSize);
	delete[] rsaResult;

	return result;
}

// Decrypt ciphertext with RSA private key
std::string rsaDecrypt(const std::string& ciphertext, RSA* rsa)
{
	std::string result;
	int rsaLen = RSA_size(rsa);
	char* rsaResult = new char[rsaLen];
	int decryptSize = RSA_private_decrypt(ciphertext.length(), (const unsigned char*)ciphertext.c_str(), (unsigned char*)rsaResult, rsa, RSA_PKCS1_PADDING);

	if (decryptSize == -1) {
		std::cerr << "Failed to decrypt data (Error code: " << ERR_get_error() << ")" << std::endl;
		delete[] rsaResult;
		return result;
	}

	result.assign(rsaResult, decryptSize);
	delete[] rsaResult;

	return result;
}

int main()
{
	// Initialize OpenSSL library
	OpenSSL_add_all_algorithms();
	ERR_load_BIO_strings();
	ERR_load_crypto_strings();
	load_random_seed();

	string publicPem;
	string privatePem;

	RSA_generate_pkcs1(2048, publicPem, privatePem);
	std::cout << "PKCS#1 Public Key:" << std::endl << publicPem << std::endl;
	std::cout << "PKCS#1 Private Key:" << std::endl << privatePem << std::endl;
	RSA_generate_pkcs8(2048, publicPem, privatePem);
	std::cout << "PKCS#8 Public Key:" << std::endl << publicPem << std::endl;
	std::cout << "PKCS#8 Private Key:" << std::endl << privatePem << std::endl;

	RSA* public_RSA = load_public_key(publicPem.c_str());
	RSA* private_RSA = load_private_key(privatePem.c_str());

	std::string plaintext = "Hello, world!";
	std::string ciphertext = rsaEncrypt(plaintext, public_RSA);
	std::string decryptedText = rsaDecrypt(ciphertext, private_RSA);

	std::cout << "Plaintext: " << plaintext << std::endl;
	std::cout << "Ciphertext: " << ciphertext << std::endl;
	std::cout << "Decrypted Text: " << decryptedText << std::endl;

	return 0;
}
```

### 注意事项
#### 报错：C4996 'RSA_new': Since OpenSSL 3.0
使用 OpenSSL 3.1.0 调用已弃用的 OpenSSL 1.1.1 函数
添加宏定义 #define OPENSSL_API_COMPAT 0x10100000L 使用与 OpenSSL 1.1.x 兼容的 API

#### PKCS#1 在公钥转 RSA 对象时会报错
使用命令提取公钥信息以验证公钥正确性，但是不清楚原因的使用 PKCS#1 在公钥转 RSA 对象时会报错，但是使用 PKCS#8 没问题。
``` CMD
:: 使用 OpenSSL 命令来提取公钥内容。
:: 检查输出结果以确保公钥信息正确。在输出结果中能够看到包含模数和指数的 Public-Key: (RSA) 行。
openssl rsa -in public.pem -pubin -text
```