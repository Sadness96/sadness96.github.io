---
title: 压缩Hexo博客生成空白行
date: 2019-06-17 13:32:19
tags: [python,hexo]
categories: Python
---
### 解决Hexo博客系统生成导致大量空白行问题，生成文件拷贝到外部目录问题
<!-- more -->
#### 删除Hexo博客系统生成文件大量空白行
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
#### 拷贝Hexo博客系统public目录
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