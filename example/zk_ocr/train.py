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


Batch = namedtuple('Batch', 'data', 'label')


class DataSource(mx.io.DataIter):
    def __init__ (self, path, batch_size=32):
        idx_fname = osp.splitext(path)[0] + '.idx'
        self.db = mx.recordio.MXIndexedRecordIO(idx_fname, path, 'r')
        self.db.open()
        self.num_ = len(db.keys)
        self.batch_size_ = batch_size

    def __del__(self):
        self.db.close()

    @property
    def provide_data(self):
        pass

    @property
    def provide_label(self):
        pass

    def reset(self):
        pass

    def iter_next(self):
        return self.get_batch()

    def next(self):
        batch = self.iter_next()
        if not batch:
            raise StopIteration
        return batch

    def load_imgs_labels(self, base_idx):
        ''' 加载 base_idx 到 base_idx+batch_size 个 img，根据 shape[2] 做成统一的 ...
        '''
        imgs = []
        labels = []

        for i in range(base_idx, base_idx + self.batch_size_):
            raw_data = self.db.read_idx(i)
            data = mx.recordio.unpack(raw_data)
            label = data[0].label # label: numpy, shape=(yyy,)
            img = cv2.imdecode(np.fromstring(data[1], dtype=np.uint8), 1)
            # img.shape = (60, xxx, 3)
            imgs.append(img)
            labels.append(label)

        # TODO: 返回之前，转化为 mx.nd.array
        return imgs, labels




    def get_batch(self):
        ''' 因为图片本来已经打乱，所以按照顺序走吧 :)
        '''
        cnt = self.num_ / self.batch_size_
        for i in range(cnt):
            imgs, labels = self.load_imgs_labels(i)
            yield Batch(data=imgs, label=labels)

        return None
