#!/bin/env python
#coding: utf-8


''' 准备华师标注的图像集合，输出 train.lst, val.lst 

        root_path
            华师标注0108
                022--
                    022S0-2.jpg
                    022S0-2.xml
                    ...
                ...
            ...

'''


root_path = '/media/nas/华师标注'
train_set = ['华师标注0108', '华师标注0115']
val_set = ['华师标注0206']


import os.path as osp
import os
import sys
import subprocess
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))
from dataset.pascal_voc import PascalVoc
from dataset.mscoco import Coco
from dataset.concat_db import ConcatDB


def load_jpg_xml(path):
    rs = []
    for fname in os.listdir(path):
        basename,ext = osp.splitext(fname)
        if ext == '.jpg':
            rs.append(osp.sep.join((path, basename)))
    return rs


def load_subpath(path):
    ''' 返回前缀列表, path=root_path/华师标注0108 然后找这个目录下的所有子目录里面的 jpg, xml
    '''
    rs = []
    for name in os.listdir(path):
        name = osp.sep.join((path, name))
        if osp.isdir(name):
            rs.extend(load_jpg_xml(name))
    return rs

rs = []
for ts in train_set:
    rs.extend(load_subpath(osp.sep.join((root_path, ts))))

with open(curr_path+'/VOC2007/ImageSets/Main/trainval.txt', 'w') as f:
    for r in rs:
        f.write(r)
        f.write('\n')
print(curr_path+'/VOC2007/ImageSets/Main/trainval.txt' + '  saved!')

rs = []
for vs in val_set:
    rs.extend(load_subpath(osp.sep.join((root_path, vs))))

with open(curr_path+'/VOC2007/ImageSets/Main/test.txt', 'w') as f:
    for r in rs:
        f.write(r)
        f.write('\n')
print(curr_path+'/VOC2007/ImageSets/Main/test.txt' + '  saved!')


def load_pascal(image_set, year, devkit_path, shuffle=False):
    """
    wrapper function for loading pascal voc dataset

    Parameters:
    ----------
    image_set : str
        train, trainval...
    year : str
        2007, 2012 or combinations splitted by comma
    devkit_path : str
        root directory of dataset
    shuffle : bool
        whether to shuffle initial list

    Returns:
    ----------
    Imdb
    """
    image_set = [y.strip() for y in image_set.split(',')]
    assert image_set, "No image_set specified"
    year = [y.strip() for y in year.split(',')]
    assert year, "No year specified"

    # make sure (# sets == # years)
    if len(image_set) > 1 and len(year) == 1:
        year = year * len(image_set)
    if len(image_set) == 1 and len(year) > 1:
        image_set = image_set * len(year)
    assert len(image_set) == len(year), "Number of sets and year mismatch"

    imdbs = []
    for s, y in zip(image_set, year):
        imdbs.append(PascalVoc(s, y, devkit_path, shuffle, is_train=True, classes=[
            'stand up',
            'writing',
            'reading',
            'student_look book',
            'student_raise hand',
            'wait',
            'no listen',
            'discussion group',
            'bend',
            'point screen',
            'point blackboard',
            'point students',
            'writing blackboard',
            'demonstrate',
            'head',
        ]))
    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]


db = load_pascal('trainval', '2007', curr_path, True)
print("saving list to disk...")
db.save_imglist(curr_path + '/trainval.lst')

db = load_pascal('test', '2007', curr_path, False)
db.save_imglist(curr_path + '/test.lst')