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

#include <iostream>
#include <string>

// Generate RSA key pair
RSA* generateRSAKeyPair(const int keyLength)
{
	RSA* rsa = RSA_new();
	BIGNUM* bne = BN_new();
	unsigned long e = RSA_F4;

	if (!BN_set_word(bne, e)) {
		std::cerr << "Failed to set Big Number exponent" << std::endl;
		return nullptr;
	}

	// Generate the RSA key
	if (!RSA_generate_key_ex(rsa, keyLength, bne, nullptr)) {
		std::cerr << "Failed to generate RSA key" << std::endl;
		return nullptr;
	}

	// Free memory
	BN_free(bne);

	return rsa;
}

// Convert RSA public key to PEM format
std::string convertRSAPublicKeyToPEM(RSA* rsa)
{
	BIO* bio = BIO_new(BIO_s_mem());
	PEM_write_bio_RSAPublicKey(bio, rsa);
	BUF_MEM* bufferPtr;
	BIO_get_mem_ptr(bio, &bufferPtr);
	std::string publicKey(bufferPtr->data, bufferPtr->length);
	BIO_set_close(bio, BIO_CLOSE);
	BIO_free_all(bio);

	return publicKey;
}

// Convert RSA private key to PEM format
std::string convertRSAPrivateKeyToPEM(RSA* rsa)
{
	BIO* bio = BIO_new(BIO_s_mem());
	PEM_write_bio_RSAPrivateKey(bio, rsa, nullptr, nullptr, 0, nullptr, nullptr);
	BUF_MEM* bufferPtr;
	BIO_get_mem_ptr(bio, &bufferPtr);
	std::string privateKey(bufferPtr->data, bufferPtr->length);
	BIO_set_close(bio, BIO_CLOSE);
	BIO_free_all(bio);

	return privateKey;
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
	RAND_load_file("/dev/random", 32);

	const int KEY_LENGTH = 2048;
	RSA* rsa = generateRSAKeyPair(KEY_LENGTH);

	std::string publicKey = convertRSAPublicKeyToPEM(rsa);
	std::string privateKey = convertRSAPrivateKeyToPEM(rsa);

	std::cout << "Public Key:" << std::endl << publicKey << std::endl;
	std::cout << "Private Key:" << std::endl << privateKey << std::endl;

	std::string plaintext = "Hello, world!";
	std::string ciphertext = rsaEncrypt(plaintext, rsa);
	std::string decryptedText = rsaDecrypt(ciphertext, rsa);

	std::cout << "Plaintext: " << plaintext << std::endl;
	std::cout << "Ciphertext: " << ciphertext << std::endl;
	std::cout << "Decrypted Text: " << decryptedText << std::endl;

	// Clean up
	RSA_free(rsa);
	EVP_cleanup();
	CRYPTO_cleanup_all_ex_data();
	ERR_free_strings();

	return 0;
}
```