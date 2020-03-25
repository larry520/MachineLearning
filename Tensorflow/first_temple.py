#!/usr/bin/env python
# -*- encoding:utf-8 -*-
"""
@version: ??
@author:Administrator
@file: first_temple.py
@time:2020/3/23 23:01
"""

# https://codeload.github.com/dragen1860/Deep-Learning-with-TensorFlow-book/zip/master

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

message = tf.constant("welcom to the exciting world of Machine learning!")

with tf.Session() as sess:
    print(sess.run(message).decode())