# code:utf-8

import os
import numpy as np

# 获取本地文件图片
def list_pic(picpath):
    print("Get image files ... ", end='\n')

    files = os.listdir(picpath)
    print(files, end='\n')
    pic_files = []

    for f in files:
        if os.path.isdir(f):
            continue

        if get_file_ext(f).lower() == '.jpg':
            pic_files.append(f)

        if get_file_ext(f).lower() == '.jpeg':
            pic_files.append(f)

        if get_file_ext(f).lower() == '.png':
            pic_files.append(f)

    count_pic = len(pic_files)
    print("%s found" % count_pic)
    print(picpath, end='\n')
    return count_pic, pic_files


# 获取文件后缀
def get_file_ext(file_name):
    dot_pos = file_name.rfind('.')
    if dot_pos == -1:
        ext = ''
    else:
        ext = file_name[dot_pos:]

    return ext