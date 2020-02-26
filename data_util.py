#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = data_util.py
__author__ = 'guo_h'
__mtime__ = 2020/02/25 
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓     ┏┓
            ┏┛┻━━━━━┛┻━┓
            ┃    ☃    ┃
            ┃  ┳┛  ┗┳  ┃
            ┃    ┻     ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━┓
              ┃  神兽保佑 ┣┓
              ┃　永无BUG！┏┛
              ┗┓┓┏━━━┓┓┏━┛
               ┃┫┫   ┃┫┫
               ┗┻┛   ┗┻┛
"""
import os
import jieba

from getConfig import get_config


CONFIG = dict()
CONFIG = get_config()

conv_path = CONFIG['resource_data']
if not os.path.exists(conv_path):
    exit()


convs = list()
with open(conv_path, encoding='utf-8') as resource_file:
    one_conv = list()
    for line in resource_file:
        line = line.strip('\n').replace('/', '')
        if line == '':
            continue
        if line[0] == CONFIG['e']:
            if one_conv:
                convs.append(one_conv)
            one_conv = []
        elif line[0] == CONFIG['m']:
            one_conv.append(line.split(' ')[1])

seq = []

for conv in convs:
    if len(conv) == 1:
        continue
    if len(conv) % 2 != 0:
        conv = conv[:-1]
    for i in range(len(conv)):
        if i % 2 == 0:
            conv[i] = " ".join(jieba.cut(conv[i]))
            conv[i+1] = " ".join(jieba.cut(conv[i+1]))
            seq.append(conv[i] + '\t' + conv[i+1])

seq_train = open(CONFIG['seq_data'], 'w')
for i in range(len(seq)):
    seq_train.write(seq[i] + '\n')
    if 1 % 1000 == 0:
        print(len(range(len(seq))), "处理进度:", i)

seq_train.close()
