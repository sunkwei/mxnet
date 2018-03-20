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


curr_path = osp.dirname(osp.abspath(__file__))


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

        label = main[pos+1:]
        if len(label) > max_len:
            max_len = len(label)
            max_label = label

        descrs.append((label, osp.sep.join((path, fname))))

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
    '''
    with open(fname, 'w') as f:
        for i,lf in enumerate(label_fnames):
            line = '\t'.join((str(i), label2ids(lf[0], r_vocab, max_label_len), lf[1], '\n'))
            f.write(line)


if __name__ == '__main__':
    test_label_fnames, test_max_label, test_max_label_len = load_all('/media/nas/ocr/OCR_samples/sundy_all_samples_可直接使用的/test')
    test_labels = [ lf[0] for lf in test_label_fnames ]
    vocab, r_vocab = build_vocab(test_labels)

    train_label_fnames, train_max_label, train_max_label_len = load_all('/media/nas/ocr/OCR_samples/sundy_all_samples_可直接使用的/train')
    train_labels = [ lf[0] for lf in train_label_fnames ]
    vocab, r_vocab = build_vocab(train_labels, vocab, r_vocab)

    max_label_len = max((test_max_label_len, train_max_label_len))

    print('max_label_len is {}'.format(max_label_len))

    save_list(curr_path + '/train.lst', train_label_fnames, r_vocab, max_label_len)
    save_list(curr_path + '/test.lst', test_label_fnames, r_vocab, max_label_len)


