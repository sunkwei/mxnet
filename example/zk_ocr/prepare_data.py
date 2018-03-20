#!/bin/env python
#coding: utf-8


''' 输入：路径名字，输出：prefix.lst

    输入目录中的文件结构：
        \--
            xxxx_<label>.jpg
            yyyy_<label>.jpg

        文件名字中，最后一个 '_' 之后到 '.jpg' 之前为该图片文件的 label 内容
'''


import sys
import os
import os.path as osp


def load_all(path):
    ''' 加载这个目录中的所有 xxx_<label>.jpg 格式的文件名，然后提取 label， 返回最长 label ...
    '''
    descrs = []
    path = osp.abspath(path)
    max_len = 0
    max_label = None

    for fname in os.listdir(path):
        pos = fname.rfind('_')
        if pos == -1:
            continue

        main,ext = osp.splitext(fname)
        if ext != '.jpg':
            continue

        label = main[:pos]
        if len(label) > max_len:
            max_len = len(label)
            max_label = label

        descrs.append((label, osp.sep.join((path, fname))))

    return descrs, max_label, max_len



if __name__ == '__main__':
    pass