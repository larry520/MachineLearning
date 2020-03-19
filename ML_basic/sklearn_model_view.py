#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author:dmsoft
@file: sklearn_model_view.py
@time: 2020/03/17

本函数用于显示sklearn分类模型预测边界
"""
import numpy as np
import matplotlib.pyplot as plt


def sklearn_model_view(sklean_model,x1,x2,y_train):
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)

    x1_min = x1.min(axis=0) - 0.5
    x2_min = x2.min(axis=0) - 0.5
    x1_max = x1.max(axis=0) + 0.5
    x2_max = x2.max(axis=0) + 0.5
    h = 0.1
    x1_values, x2_values = np.meshgrid(
        np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))

    t = (x1_values.reshape(-1,1),x2_values.reshape(-1,1))
    Z = sklean_model.predict(
        np.concatenate((x1_values.reshape(-1,1),x2_values.reshape(-1,1)),axis=1))

        # np.concatenate((x1_values.ravel(),x2_values.ravel()),axis=1))

    Z = Z.reshape(x1_values.shape)
    # plt.contourf 对区域进行填充
    plt.contourf(x1_values,x2_values,Z ,cmap=plt.cm.Spectral)
    plt.scatter(x1.ravel(),x2.ravel(), cmap=plt.cm.Spectral)

