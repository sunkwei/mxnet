#!/bin/env python3
#coding: utf-8


''' 使用 bucket model 训练
'''


import mxnet as mx
import numpy as np
import os
import os.path as osp
import argparse
import sys
import cv2
from collections import namedtuple
from network import build_net
import math
import pickle
from config import Config
from stt_metric import STTMetric


ctx = mx.gpu(1)
curr_path = osp.dirname(osp.abspath(__file__))
prefix = 'ocr'


def load_vocab(fname=curr_path+'/data/vocab.pkl'):
    with open(fname, 'rb') as f:
        d = pickle.load(f)

    return d['vocab'], d['r_vocab']


Config.vocab, Config.r_vocab = load_vocab()
Config.vocab_size = len(Config.vocab)



class Batch:
    def __init__(self, data, label, bucket_key):
        self.data = data
        self.label = label
        self.bucket_key = bucket_key

    @property
    def provide_data(self):
        return [('data', self.data[0].shape)]

    @property
    def provide_label(self):
        return [('label', self.label[0].shape)]


class DataSource:
    def __init__ (self, path, batch_size=32):
        idx_fname = osp.splitext(path)[0] + '.idx'
        self.db_ = mx.recordio.MXIndexedRecordIO(idx_fname, path, 'r')
        self.db_.open()
        self.num_ = len(self.db_.keys)
        self.batch_size_ = batch_size

    def __del__(self):
        self.db_.close()

    def __iter__(self):
        ''' 因为图片本来已经打乱，所以按照顺序走吧 :)
        '''
        cnt = self.num_ / self.batch_size_
        for i in range(int(cnt)):
            imgs, labels, bucket_key = self.load_imgs_labels(i*self.batch_size_)
            yield Batch(imgs, labels, bucket_key)

        self.db_.reset()
        raise StopIteration
        

    @property
    def count(self):
        return self.num_

    def load_imgs_labels(self, base_idx):
        ''' 加载 base_idx 到 base_idx+batch_size 个 img，根据 shape[2] 做成统一的 ...
            
            XXX: 为了使用 bucketing 模式训练，应该设计一组"阶梯" ...
        '''
        imgs = []
        labels = []

        max_width = 0


        for i in range(base_idx, base_idx + self.batch_size_):
            raw_data = self.db_.read()
            data = mx.recordio.unpack(raw_data)

            label = data[0].label # label: numpy, shape=(yyy,)
            if isinstance(label, float):
                label = np.array([label], dtype=np.int)
            else:
                label = label.astype(np.int)

            img = cv2.imdecode(np.fromstring(data[1], dtype=np.uint8), 1)
            if img.shape[1] >= Config.max_img_width:
                img = img[:,:Config.max_img_width,:]

            # img.shape = (60, xxx, 3)
            img = np.swapaxes(img, 0, 2)    # (3, xxx, 60)
            img = np.swapaxes(img, 1, 2)    # (3, 60, xxx)

            img = img.astype(dtype=np.float32)

            img /= 255.0
            img -= 0.5

            if img.shape[-1] > max_width:
                max_width = img.shape[-1]

            imgs.append(img)
            labels.append(label)

        # 检查 max_width 落在那个 bucket 区间
        n = int(math.ceil(1.0 * Config.max_img_width / Config.bucket_num)) # 每个区间的 width 
        max_width = (max_width + n - 1) // n * n
        img_key = max_width // n - 1

        # 将 img 转化为 (3, 60, bucket_cc)
        imgs2 = []
        for img in imgs:
            pad_n = max_width - img.shape[-1]
            if pad_n:
                img = np.pad(img, ((0,0),(0,0),(0,pad_n)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            img = img.reshape(3*60*max_width)
            imgs2.append(img)
            
        imgs = np.vstack(imgs2)
        imgs = imgs.reshape((self.batch_size_, 3, 60, max_width))

        ll = int(math.ceil(1.0*Config.max_label_len/Config.bucket_num))
        label_len = ll * (img_key+1)    # 此处 img_key 相当于 bucket_key

        labels2 = []
        for label in labels:
            pad_n = label_len - label.shape[0]
            if pad_n < 0:
                print('bucket key={}, img width={}'.format(img_key, max_width))
                print('???? label_len={}, label.shape={}, {}'.format(label_len, label.shape, label))
                label = label[:label_len]
            elif pad_n:
                label = np.pad(label, (0,pad_n), 'constant', constant_values=(0, 0)) # 0: <pad>
            label = label.reshape((1,-1))
            labels2.append(label)

        labels = np.vstack(labels2)
        labels = labels.reshape((self.batch_size_, label_len))

        # 返回之前，转化为 mx.nd.array
        return [mx.nd.array(imgs)], [mx.nd.array(labels)], img_key


def save_checkpoint(mod, prefix, epoch):
    sym,_,_, = mod._sym_gen(Config.bucket_num-1)
    sym.save('{}-symbol.json'.format(prefix))
    mod.save_params(prefix + '-%04d.params' % epoch)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, help='resume epoch')
    args = parser.parse_args()
    
    batch_size = Config.batch_size
    epochs = Config.epoch_num

    loss_metric = STTMetric(batch_size=batch_size, is_epoch_end=False)
    eval_metric = STTMetric(batch_size=batch_size, is_epoch_end=True)

    ds_train = DataSource(curr_path+'/data/train.rec', batch_size)
    ds_test = DataSource(curr_path+'/data/test.rec', batch_size)

    mod = mx.mod.BucketingModule(build_net, default_bucket_key=Config.bucket_num - 1,
            context=[ctx])

    # bind 时，使用 default_bucket_key 计算
    ww = int(math.ceil(1.0 * Config.max_img_width / Config.bucket_num))

    data_shapes = [('data', (batch_size, 3, 60, ww * Config.bucket_num))]
    label_shapes = [('label', (batch_size, Config.max_label_len))]
    mod.bind(data_shapes=data_shapes, label_shapes=label_shapes, for_training=True)

    if args.resume:
        from_epoch = args.resume
        print('RESUME from {}'.format(from_epoch))
        sym,args,auxs = mx.model.load_checkpoint(prefix, from_epoch)
        mod.set_params(args, auxs)
    else:
        from_epoch = -1
        mod.init_params(mx.init.Xavier(factor_type='in', magnitude=2.34)) 
        
#    mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': Config.learning_rate, 'momentum': 0.5})
    mod.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': Config.learning_rate})
    
    # go
    for e in range(from_epoch+1, epochs):
        eval_metric.reset()
        for i,batch in enumerate(ds_test):
            mod.forward(batch, is_train=False)
            if i % Config.eval_show_step == Config.eval_show_step -1:
                mod.update_metric(eval_metric, batch.label)


        # train
        loss_metric.reset()

        for i,batch in enumerate(ds_train):
            mod.forward_backward(batch)
            mod.update()

            if i % Config.train_show_step == Config.train_show_step - 1:
                print('==> epoch:{}, batch:{}'.format(e, i))
                mod.update_metric(loss_metric, batch.label)

        # save checkpoint
        save_checkpoint(mod, 'ocr', e)

        # val
#        eval_metric.reset()

#        for i,batch in enumerate(ds_test):
#            mod.forward(batch, is_train=False)
#            if i % Config.eval_show_step == Config.eval_show_step -1:
#                mod.update_metric(eval_metric, batch.label)

        # 根据 e 调整 lr
#        if (e+1) % Config.lr_reduce_epoch:
#            pass


if __name__ == '__main__':
    def test_ds():
        ds = DataSource(osp.dirname(osp.abspath(__file__))+'/t0.rec', batch_size=2)
        for batch in ds:
            print(batch.provide_data, batch.provide_label, batch.bucket_key)

    train()
