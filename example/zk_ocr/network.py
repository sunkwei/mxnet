#!/bin/env python
#coding: utf-8


import mxnet as mx
import math
from config import Config



ww = math.ceil(1.0 * Config.max_img_width / Config.bucket_num)




def build_net(bucket_key):
    ''' 构造网络，返回 (sym, data_names, label_names)
    '''

    # simg 输入图像 width，slabel 输入标签长度
    simg = (bucket_key+1) * ww

    data = mx.sym.var(name='data')      # data shape: (batch_size, 3, 60, ww)
    label = mx.sym.var(name='label')    # label shape: (batch_size, MAX_LABEL_LEN)

    ker = (3,3)
    stride = (1,4)
    data = mx.sym.Convolution(data, name='conv1', num_filter=64, kernel=ker, stride=stride)
    data = mx.sym.Activation(data, act_type='relu')
    axis3 = int(math.floor((simg - ker[1]) / stride[1] + 1))  # (simg - kernel) / stride + 1

    ker = (3,3)
    stride = (2,4)
    data = mx.sym.Convolution(data, name='conv2', num_filter=128, kernel=ker, stride=stride)
    data = mx.sym.Activation(data, act_type='relu')
    axis3 = int(math.floor((axis3 - ker[1]) / stride[1] + 1))

    # k_w = 3
    # s_w = 5
    # data = mx.sym.Convolution(data, name='conv3', num_filter=256, kernel=(3,k_w), stride=(1,s_w))
    # data = mx.sym.Activation(data, act_type='relu')
    # axis3 = int(math.floor((axis3 - k_w) / s_w + 1))

    # 将 data[3] 轴完全分割 ...
    slice_cnt = axis3
    data = mx.sym.split(data, num_outputs=slice_cnt, axis=3)

    data = [ d for d in data ]

    # rnn
    stack = mx.rnn.SequentialRNNCell()
    for i in range(Config.rnn_layers_num):
        cell = mx.rnn.GRUCell(Config.rnn_hidden_num, prefix='gru_{}'.format(i))
        stack.add(cell)

    outputs, states = stack.unroll(slice_cnt, data)

    # fc: 使用共享参数，将 outputs[n] 映射到 vocab 空间
    cls_weight = mx.sym.var(name='cls_weight')
    cls_bias = mx.sym.var(name='cls_bias')
    
    seq = []
    for i,o in enumerate(outputs):
        fc = mx.sym.FullyConnected(data=o, name='fco_{}'.format(i), 
                num_hidden=Config.vocab_size, weight=cls_weight, bias=cls_bias)
        seq.append(fc)
    
    data = mx.sym.concat(*seq, dim=0)

    # ctc 为啥要把 batch 合并呢？
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')

    print('build_net: img_width={}, slice_cnt={}, label_length={}'.format(simg, slice_cnt, Config.max_label_len))
    data = mx.sym.WarpCTC(data=data, label=label, input_length=slice_cnt, label_length=Config.max_label_len)

    return (data, ['data'], ['label'])



if __name__ == '__main__':
    net = build_net(9)
    print(net)
