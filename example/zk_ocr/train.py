#!/bin/env python
#coding: utf-8


''' 使用 bucket model 训练
'''


import mxnet as mx
import numpy as np
import os
import os.path as osp
import sys
import cv2
from collections import namedtuple


Batch = namedtuple('Batch', ('data', 'label'))


class DataSource:
    def __init__ (self, path, batch_size=32):
        idx_fname = osp.splitext(path)[0] + '.idx'
        self.db_ = mx.recordio.MXIndexedRecordIO(idx_fname, path, 'r')
        self.db_.open()
        self.num_ = len(self.db_.keys)
        self.batch_size_ = batch_size

    def __del__(self):
        self.db_.close()

    @property
    def provide_data(self):
        pass

    @property
    def provide_label(self):
        pass

    def reset(self):
        pass

    def __iter__(self):
        ''' 因为图片本来已经打乱，所以按照顺序走吧 :)
        '''
        cnt = self.num_ / self.batch_size_
        for i in range(int(cnt)):
            imgs, labels = self.load_imgs_labels(i*self.batch_size_)
            yield Batch(data=imgs, label=labels)
        raise StopIteration

    def load_imgs_labels(self, base_idx):
        ''' 加载 base_idx 到 base_idx+batch_size 个 img，根据 shape[2] 做成统一的 ...
        '''
        imgs = []
        labels = []

        max_width = 0
        max_label_len = 0

        for i in range(base_idx, base_idx + self.batch_size_):
            raw_data = self.db_.read_idx(i)
            data = mx.recordio.unpack(raw_data)

            label = data[0].label # label: numpy, shape=(yyy,)
            if isinstance(label, float):
                label = np.array([label], dtype=np.int)
            else:
                label = label.astype(np.int)

            if max_label_len < len(label):
                max_label_len = len(label)

            img = cv2.imdecode(np.fromstring(data[1], dtype=np.uint8), 1)
            # img.shape = (60, xxx, 3)
            img = np.swapaxes(img, 0, 2)    # (3, xxx, 60)
            img = np.swapaxes(img, 1, 2)    # (3, 60, xxx)

            if img.shape[-1] > max_width:
                max_width = img.shape[-1]

            imgs.append(img)
            labels.append(label)

        # 将 img 转化为 (3, 60, max_weight)
        imgs2 = []
        for img in imgs:
            pad_n = max_width - img.shape[-1]
            if pad_n:
                img = np.pad(img, ((0,0),(0,0),(0,pad_n)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            img = img.reshape(3*60*max_width)
            imgs2.append(img)
            
        imgs = np.vstack(imgs2)
        imgs = imgs.reshape((self.batch_size_, 3, 60, max_width))

        labels2 = []
        for label in labels:
            pad_n = max_label_len - label.shape[0]
            if pad_n:
                label = np.pad(label, (0,pad_n), 'constant', constant_values=(0,-1))
            label = label.reshape((1,-1))
            labels2.append(label)

        labels = np.vstack(labels2)
        labels = labels.reshape((self.batch_size_, max_label_len))

        # 返回之前，转化为 mx.nd.array
        return [mx.nd.array(imgs)], [mx.nd.array(labels)]


def train():
    pass


if __name__ == '__main__':
    ds = DataSource(osp.dirname(osp.abspath(__file__))+'/t0.rec', batch_size=2)
    for batch in ds:
        print(type(batch.data[0]), batch.data[0].shape, batch.label[0].shape)