#!/bin/env python3
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
import pickle
from collections import namedtuple


curr_path = osp.dirname(osp.abspath(__file__))
Descr = namedtuple('Descr', ('fname', 'label', 'img_width'))


def get_img_width(fname):
    import cv2
    img = cv2.imread(fname)
    return img.shape[1]


def load_all(path):
    ''' 加载这个目录中的所有 xxx_<label>.jpg 格式的文件名，然后提取 label， 返回最长 label ...
    '''
    descrs = []
    path = osp.abspath(path)
    max_len = 0
    max_label = None

    for i,fname in enumerate(os.listdir(path)):
        pos = fname.rfind('_')
        if pos == -1:
            continue

        main,ext = osp.splitext(fname)
        if ext != '.jpg':
            continue

        label = main[pos+1:]
        if len(label) > max_len:
            max_len = len(label)
            max_label = label

        fname = osp.sep.join((path, fname))
        descr = Descr(fname=fname, label=label, img_width=get_img_width(fname))
        descrs.append(descr)

        
        if i % 1000 == 999:
            print('load {} items'.format(i+1))

    # 根据 img_width 排序
    descrs = sorted(descrs, key=lambda x:x.img_width)

    return descrs, max_label, max_len


def build_vocab(labels, vocab=None, r_vocab=None):
    ''' 根据 labels 构造字典，vocab 为当前已知 ...

        返回的 vocab 是 list, r_vocab 是 dict
    '''
    if vocab is None:
        vocab = []

    if r_vocab is None:
        r_vocab = {}

    for label in labels:
        for c in label:
            if c not in r_vocab:
                r_vocab[c] = len(vocab)
                vocab.append(c)

    return vocab, r_vocab


def label2ids(label_str, r_vocab, max_label_len):
    ids = [ str(r_vocab[c]) for c in label_str ]
    return '\t'.join(ids)


def save_list(fname, label_fnames, r_vocab, max_label_len):
    ''' 需要补齐 max_label_len
        在 label 前后插入 <b>, <e>
    '''
    with open(fname, 'w') as f:
        for i,lf in enumerate(label_fnames):
            line = '\t'.join((str(i), '1', label2ids(lf.label, r_vocab, max_label_len), '2', lf.fname, '\n'))
            f.write(line)


if __name__ == '__main__':
    vocab = ['<pad>', '<b>', '<e>', '~']
    r_vocab = { 
        '<pad>': 0, 
        '<b>': 1,
        '<e>': 2,
        '~': 3,
    }

    test_sample_path = 'nas/ocr/OCR_samples/sundy_all_samples_可直接使用的/test'
    train_sample_path = 'nas/ocr/OCR_samples/sundy_all_samples_可直接使用的/train'

    print('loading: test ...')
    test_label_fnames, test_max_label, test_max_label_len = load_all(test_sample_path)
    test_labels = [ lf.label for lf in test_label_fnames ]
    vocab, r_vocab = build_vocab(test_labels, vocab, r_vocab)
    print('test: vocab size:{}'.format(len(vocab)))

    print('loading: train ...')
    train_label_fnames, train_max_label, train_max_label_len = load_all(train_sample_path)
    train_labels = [ lf.label for lf in train_label_fnames ]
    vocab, r_vocab = build_vocab(train_labels, vocab, r_vocab)
    print('train: vocab size:{}'.format(len(vocab)))

    max_label_len = max((test_max_label_len, train_max_label_len))

    print('max_label_len is {}'.format(max_label_len))

    save_list(curr_path + '/data/train.lst', train_label_fnames, r_vocab, max_label_len)
    save_list(curr_path + '/data/test.lst', test_label_fnames, r_vocab, max_label_len)

    print('train.lst, val.lst saved')

    v = {'vocab': vocab, 'r_vocab': r_vocab }
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(v, f, True)

    print('vocab.pkl saved')


