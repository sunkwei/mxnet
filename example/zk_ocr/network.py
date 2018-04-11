#!/bin/env python
#coding: utf-8


import mxnet as mx
import math
from config import Config



ww = math.ceil(1.0 * Config.max_img_width / Config.bucket_num)
ll = math.ceil(1.0 * Config.max_label_len / Config.bucket_num)



def build_net(bucket_key):
    ''' 构造网络，返回 (sym, data_names, label_names)

        label 长度根据 bucket_key 决定：

    '''

    # simg 输入图像宽度
    simg = (bucket_key+1) * ww
    
    # label 长度
    slab = (bucket_key+1) * ll

    data = mx.sym.var(name='data')      # data shape: (batch_size, 3, 60, simg)
    label = mx.sym.var(name='label')    # label shape: (batch_size, slab)

    ker = (3,3)
    stride = (2,2)
    data = mx.sym.Convolution(data, name='conv1', num_filter=16, kernel=ker, stride=stride, pad=(1,1))
    data = mx.sym.Activation(data, act_type='relu')
    axis3 = int(math.floor((simg - ker[1] + 2) / stride[1] + 1))  # (simg - kernel) / stride + 1

    ker = (3,3)
    stride = (1,2)
    data = mx.sym.Convolution(data, name='conv2', num_filter=64, kernel=ker, stride=stride, pad=(1,1))
    data = mx.sym.Activation(data, act_type='relu')
    axis3 = int(math.floor((axis3 - ker[1] + 2) / stride[1] + 1))

    ker = (3,3)
    stride = (1,1)
    data = mx.sym.Convolution(data, name='conv3', num_filter=128, kernel=ker, stride=stride, pad=(1,1))
    data = mx.sym.Activation(data, act_type='relu')
    axis3 = int(math.floor((axis3 - ker[1] + 2) / stride[1] + 1))

#    ker = (3,3)
#    stride = (1,2)
#    data = mx.sym.Convolution(data, name='conv4', num_filter=256, kernel=ker, stride=stride, pad=(1,1))
#    data = mx.sym.Activation(data, act_type='relu')
#    axis3 = int(math.floor((axis3 - ker[1] + 2) / stride[1] + 1))


    # 将 data[3] 轴完全分割 ...
    slice_cnt = axis3
    data = mx.sym.split(data, num_outputs=slice_cnt, axis=3)

    data = [ d for d in data ]

    # rnn
    stack = mx.rnn.SequentialRNNCell()
    for i in range(Config.rnn_layers_num):
        cell = mx.rnn.GRUCell(Config.rnn_hidden_num, prefix='gru_{}'.format(i))
        stack.add(cell)
#        stack.add(mx.rnn.DropoutCell(0.3, prefix='drop_{}'.format(i)))

    outputs, states = stack.unroll(slice_cnt, data)

    # fc: 使用共享参数，将 outputs[n] 映射到 vocab 空间
    cls_weight = mx.sym.var(name='cls_weight')
    cls_bias = mx.sym.var(name='cls_bias')
    
    seq = []
    for i,o in enumerate(outputs):
        fc = mx.sym.FullyConnected(data=o, name='fco_{}'.format(i), 
                num_hidden=Config.vocab_size, weight=cls_weight, bias=cls_bias)
        fc = mx.sym.Dropout(fc, p=0.3)
        seq.append(fc)
    
    data = mx.sym.concat(*seq, dim=0)

    # ctc 为啥要把 batch 合并呢？
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data=label, dtype='int32')

    print('build_net: img_width={}, slice_cnt={}, label_length={}'.format(simg, slice_cnt, slab))
    data = mx.sym.WarpCTC(data=data, label=label, input_length=slice_cnt, label_length=slab)

    return (data, ['data'], ['label'])



if __name__ == '__main__':
    net = build_net(9)
    print(net)
