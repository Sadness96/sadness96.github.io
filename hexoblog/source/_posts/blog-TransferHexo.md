---
title: 迁移博客至 GitHub Pages + Hexo
date: 2019-06-18 12:40:45
tags: [hexo,python,cmd,git]
categories: Blog
---
<img src="https://raw.githubusercontent.com/Sadness96/sadness96.github.io/master/images/blog/blog-TransferHexo/BlogLogo.png"/>

<!-- more -->
### 简介
原来只有使用 [WordPress](https://cn.wordpress.org/) 记录一部分博客，以及平时有随手记录工作生活的习惯，这次迁移博客统一整理一遍。
### 技术简介
博客主体使用 [GitHub Pages](https://pages.github.com/) 作为静态网站托管平台，使用 [Hexo](https://hexo.io/zh-cn/) 作为静态博客生成框架，以及使用 [theme-next](http://theme-next.iissnan.com/) 主题。
### 开发环境
[VSCode](https://code.visualstudio.com/) + [Node.js](http://nodejs.cn/) + [Hexo](https://hexo.io/zh-cn/) + [Python](https://www.python.org/) + [GIT](https://git-scm.com/)
### 搭建
#### 创建 GitHub Pages
##### GitHub 仓库创建与用户名同名的 ".github.io" 库
要求：名称必须为小写字母或数字
例：Github 用户名为：Test；则创建库为：test.github.io
库地址即为：https://github.com/Test/test.github.io
git地址为：https://github.com/Test/test.github.io.git
Pages访问地址为：https://test.github.io
设置网站为[HTTPS](https://baike.baidu.com/item/https/285356?fr=aladdin)：库 Settings 中勾选：Enforce HTTPS
##### 域名配置
在[阿里云](https://www.aliyun.com/)中购买域名（例：test.com）后需实名认证
安全设置中设置[禁止转移锁](https://wanwang.aliyun.com/domain/transferlock/?spm=5176.100251.0.0.7dd54f15rzxrOx)、[禁止更新锁](https://wanwang.aliyun.com/domain/domainlock/?spm=5176.100251.0.0.7dd54f15rzxrOx)
设置域名解析：新增两条记录
1、记录类型：CNAME；主机记录：www；解析线路：默认；记录值：test.github.io；TTL：10分钟；
2、记录类型：CNAME；主机记录：@；解析线路：默认；记录值：test.github.io；TTL：10分钟；
即可跳转为 test.github.io 地址
设置打开网站为域名地址：
GitHub 库中新增 CNAME 文件 保存内容为域名地址（例：test.com）
Settings 中显示 "Your site is published at https://test.com/ " 即可
##### 目录结构
|目录文件夹或文件|作用及功能|
|:---|:---|
|blog|用于发布的静态博客|
|flash|网页中加载的Flash文件|
|hexoblog|Hexo 编译博客源码|
|images|网页中加载的图片文件|
|resume|简历页面|
|.nojekyll|关闭jekyll检查|
|404.html|网站404页面|
|CNAME|设置Pages解析域名地址|
|README.md|自述文件|
|index.html|博客主界面|
##### 修改主题样式
仅展示代码样式部分，主题配置文件请查阅官网 [theme-next](http://theme-next.iissnan.com/)
修改文件路径：hexoblog/themes/next/source/css/_custom/custom.styl
``` CSS
// Custom styles.
// 主页文章添加阴影效果
.post {
    margin-top: 0px;
    margin-bottom: 20px;
    padding: 10px;
    background: #FFF;
    -webkit-box-shadow: 0 0 5px rgba(202, 203, 203, .5);
    -moz-box-shadow: 0 0 5px rgba(202, 203, 204, .5);
}

.content-wrap {
    background: transparent;
}

.footer {
    padding: 0px;
    padding-bottom: 20px;
}

// 主页文章内间距
.content-wrap {
    padding: 0px;
}

.posts-expand {
    padding-top: 0px;
}

// 超链接颜色
.post-body p a {
    color: #0593d3;
    border-bottom: none;
    border-bottom: 1px solid #0593d3;
    &:hover {
    color: #fc6423;
    border-bottom: none;
    border-bottom: 1px solid #fc6423;
    }
}

// 主页文章块缩进
.posts-expand .post-eof {
    margin: 30px auto 10px;
    width: 0px;
    height: 0px;
}

.posts-expand .post-meta {
    margin: 3px 0 30px 0;
}

.posts-expand .post-body img {
    margin: 0px auto 0px;
}

.post-button {
    margin-top: 30px;
}

// 底部页码格式
.pagination {
    margin: 40px 0 30px;
    border-top: 0px;
}
```
##### 压缩Hexo博客生成空白行
解决Hexo博客系统生成导致大量空白行问题
自动遍历目录下所有html文件，创建临时文件，把非空格行拷贝，最后在重命名文件恢复
生成博客文章后执行命令：python compress.py 即可
``` Python
import re
import os

def minify_html(filename):
    with open(filename, 'r', encoding='utf-8') as p:
        with open(filename+'.tmp', 'w', encoding='utf-8') as t:
            while True:
                l = p.readline()
                if not l:
                    break
                else:
                    if re.search(r'\S', l):
                        t.write(l)
    os.remove(filename)
    os.rename(filename+'.tmp', filename)
    print('%s 已压缩！' % filename)

def yasuo(dir_path):
    if dir_path[len(dir_path)-1] == '/':
        dir_path = dir_path[:len(dir_path)-1]
    file_list = os.listdir(dir_path)
    for i in file_list:
        if i.find('html') > 0:
            minify_html(dir_path+'/'+i)
        elif os.path.isdir(dir_path+'/'+i) and not re.match(r'\.|\_', i):
            yasuo("%s/%s" % (dir_path, i))

# dir_path：压缩相对路径
yasuo('public')
```
##### 拷贝Hexo博客系统public目录
当前系统Hexo博客源码在 /hexoblog 目录下（该目录不被上传），将生成文件自动拷贝至上层目录 /blog 下
生成博客文章后执行命令：python deploy.py 即可
``` Python
import os
import shutil

def deploy(dir_path, dir_copy):
    dirpath = r'%s\%s' % (os.path.dirname(
        os.path.realpath(__file__)), dir_path)
    dircopy = r'%s\%s' % (os.path.abspath(
        os.path.dirname(os.getcwd())), dir_copy)
    print('dirpath：%s', dirpath)
    print('dircopy：%s', dircopy)
    # 删除文件夹
    if os.path.exists(dircopy):
        shutil.rmtree(dircopy)
        print('删除文件夹成功！')
    # 拷贝文件夹
    shutil.copytree(dirpath, dircopy)
    print('拷贝文件夹成功！')

# 生成博客相对路径
# 拷贝上级目录相对路径
deploy('public', 'blog')
```
##### 遇到的问题
###### 使用VSCode开发时生成静态页不加载CSS样式
插件：View In Browser 本地网页打开
插件：Live Server 启动一个服务打开网站
###### 上传Hexo博客源码后报错：Date is not a valid datetime
由于GitHub Pages默认使用jekyll作为代码检查，在上传Hexo源码之后存在编译不通过的情况，所以需要创建.nojekyll空文件在Repository的根目录下以关闭针对jekyll的检查。
Windows下创建以“.”开头文件夹和文件（执行命令）：
``` CMD
md .folder              //创建文件夹
echo >.file             //创建文件
```
###### Hexo生成静态博客存在大量空白行
请查看文章：[压缩Hexo博客生成空白行](/blog/2019/06/17/python-CompressHexo/)
##### 执行命令
``` CMD
cd hexoblog                 //进入博客源码文件夹
hexo new <title>            //创建新文章
hexo clean                  //清理博客生成文件
hexo generate               //生成博客静文件
hexo server                 //启动博客服务测试内容
python compress.py          //压缩Hexo生成空白行
python deploy.py            //拷贝至上级 "../blog" 发布目录
cd ..                       //回到上级 "../" 目录
git status                  //对比差异文件
git add .                   //添加修改至缓存区
git commit -m "Message"     //填写修改内容
git push                    //提交修改
```