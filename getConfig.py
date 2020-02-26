#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = getConfig.py
__author__ = 'guo_h'
__mtime__ = 2020/02/24 
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
import configparser


def get_config(config_file='config.ini'):
    parser = configparser.ConfigParser()
    parser.read(config_file)
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    # _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_strings)
