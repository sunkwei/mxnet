#!/bin/env python3
#coding: utf-8


# 存储动态配置信息

class Config:
    batch_size = 32
    epoch_num = 50
    
    rnn_layers_num = 3
    rnn_hidden_num = 256

    vocab_size = -1
    vocab = None
    r_vocab = None

    max_img_width = 1000    # 图像最大宽度
    max_label_len = 55     # 最大 label 长度
    
    bucket_num = 10        # 桶的数目，等分

    learning_rate = 0.0001

    train_show_step = 1000
    eval_show_step = 1
