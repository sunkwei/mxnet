#!/bin/env python
#coding: utf-8



root_path = '/home/sunkw/work/git/pcm_ssd'
train_set = ['train',]
val_set = ['val']
prefix = 'pcm-'

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


rs = []
for ts in train_set:
    rs.extend(load_jpg_xml(osp.sep.join((root_path, ts))))

with open(curr_path+'/VOC2007/ImageSets/Main/{}trainval.txt'.format(prefix), 'w') as f:
    for r in rs:
        f.write(r)
        f.write('\n')
print(curr_path+'/VOC2007/ImageSets/Main/{}trainval.txt'.format(prefix) + '  saved!')

rs = []
for vs in val_set:
    rs.extend(load_jpg_xml(osp.sep.join((root_path, vs))))

with open(curr_path+'/VOC2007/ImageSets/Main/{}test.txt'.format(prefix), 'w') as f:
    for r in rs:
        f.write(r)
        f.write('\n')
print(curr_path+'/VOC2007/ImageSets/Main/{}test.txt'.format(prefix) + '  saved!')


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
            '000', '001', '002', '003', '004', '005', '006']))

    if len(imdbs) > 1:
        return ConcatDB(imdbs, shuffle)
    else:
        return imdbs[0]


db = load_pascal('{}trainval'.format(prefix), '2007', curr_path, True)
print("saving list to disk...")
db.save_imglist(curr_path + '/../data/{}trainval.lst'.format(prefix))

db = load_pascal('{}test'.format(prefix), '2007', curr_path, False)
db.save_imglist(curr_path + '/../data/{}test.lst'.format(prefix))