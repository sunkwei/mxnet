#!/bin/env python3
#coding: utf-8

import mxnet as mx
from network import build_net
from config import Config
import cv2
import numpy as np
from train import Batch
import sys
import os.path as osp
from label_util import LabelUtil


''' 构造 Bucketing model 用于 ocr 识别,
    输入图像, 将保持比例拉伸为 (60, xxx), 放到对应的 bucket 中 ...
'''


curr_path = osp.dirname(osp.abspath(__file__))
epoch = 1
prefix = curr_path + '/ocr'


class OCRPredictor:
    def __init__(self, default_bucket_key=Config.bucket_num-1):
        self._mod = self.load_model(default_bucket_key)
        self._vocab = LabelUtil()

    def pred(self, img):
        batch = self.prepare_img_batch(img)
        self._mod.forward(batch, is_train=False)
        return self.get_result()

    def prepare_img_batch(self, img):
        ''' 将图片保持比例拉伸为 height=60, 然后填充到对应的 bucket 大小, 作为网络输入
        '''
        assert(img.shape[2] == 3) # BGR
        aspect = 1.0 * img.shape[0] / img.shape[1]
        x = int(60 * img.shape[1] / img.shape[0])
        size = (x, 60)
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 其实换不换无所谓 ...

        delta = int(Config.max_img_width / Config.bucket_num)
        seg = int(x // delta + 1)

        width = seg * delta
        if x < width:
            # 需要补齐
            pad = np.zeros((60, width-x, 3), dtype=np.uint8)
            img = np.hstack((img, pad))

        assert(img.shape == (60,width,3))
        img = np.swapaxes(img, 0, 2)    # (ch, c, r)
        img = np.swapaxes(img, 2, 1)    # (ch, r, c)

        img = img.astype(dtype=np.float32)
        img /= 255.0
        img -= 0.5

        # FIXME: 貌似需要提供一个 label ??
        label = mx.nd.array([0]*Config.max_label_len, dtype=np.int32).reshape((1,Config.max_label_len))

        img = img.reshape((1,3,60,width))
        return Batch(data=[mx.nd.array(img)], label=[label], bucket_key=seg-1)

    def load_model(self, key):
        len_per_seg = int(Config.max_img_width // Config.bucket_num)

        self._mod = mx.mod.BucketingModule(sym_gen=build_net, default_bucket_key=key)
        self._mod.bind(data_shapes=(('data',(1,3,60,(key+1)*len_per_seg)),), 
                label_shapes=(('label', (1,Config.max_label_len)),),
                for_training=False)
        
        # load checkpoint
        _,args,auxs = mx.model.load_checkpoint(prefix, epoch)
        self._mod.set_params(args, auxs)
        return self._mod

    def get_result(self):
        ''' 获取 ctc 的输出, 然后合并相邻相同的输出 '''
        out = self._mod.get_outputs()[0]
        # out.shape = (slice, vocab_size)
        idxs = np.argmax(out, axis=1)
        idxs = idxs.asnumpy().astype(dtype=np.int32).tolist()
        nums = self.merge(idxs)
        return self._vocab.convert_num_to_word(nums)

    def merge(self, nums):
        ''' 将 nums 中, 相邻并且相等的合并
        '''
        merged = []
        curr = -1
        for n in nums:
            if n != curr:
                merged.append(n)
                curr = n
        return merged


if __name__ == '__main__':
    pred = OCRPredictor()
    img = np.zeros((60, 123, 3), dtype=np.uint8)
    img[:,:,:] = 255
    cv2.putText(img, '1234', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,0), 2)
    cv2.imwrite(curr_path+'/test.jpg', img)
    txt = pred.pred(img)
    print(txt)