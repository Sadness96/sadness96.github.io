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
