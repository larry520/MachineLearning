#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author:dmsoft
@file: __init__.py.py
@time: 2020/03/16
"""

from.unpack import load_train_labels,load_train_images
from.unpack import load_test_images,load_test_labels

__all__ = ["load_train_labels"
            ,"load_train_images"
            ,"load_test_images"
            ,"load_test_labels"]