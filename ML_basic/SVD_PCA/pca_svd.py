#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author:dmsoft
@file: pca_svd.py
@time: 2020/03/19
"""

"""
SVD 奇异值分解
PCA 主成分分析
"""
import numpy as np


def row(rows):
    return np.random.randint(1, 20, rows).reshape(-1, 1)

def sheet(rows, cols):
    tabel = row(rows)
    for i in range(1,cols):
        tabel = np.concatenate((tabel, row(rows)), axis=1)
    return tabel

def data_collection(rows, cols, layers):
    layer = []
    for i in range(layers):
        layer.append(sheet(rows, cols))
    return np.array(layer)

# # 数据集中
x = sheet(5, 5)

def dimension_decrease():
    """手动实现降维"""

    x_centered = x - x.mean(axis=0)
    # 奇异值分解获取主成分
    U, s, V = np.linalg.svd(x_centered)
    """提取前2个主成分"""
    c1 = V.T[:,0]
    c2 = V[:,1]
    """将训练集投影到主成分
    Xd-proj = X . Wd   （d:目标维数）
    """

    """ PCA逆转换，回到原始维度
    Xrecovered = Xd-prot . Wd.T"""

    W2 = V.T[:,:2]
    X2D = x_centered.dot(W2)
    X_back = X2D.dot(W2.T)
    return X2D

def PCA_sk():
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)  # 把维度降到二维
    X2D_sk = pca.fit_transform(x)
    """主成分"""
    print(pca.components_.T[:,0])
    """各主成分轴对整个数据集贡献度"""
    print(pca.explained_variance_, pca.explained_variance_ratio_)

    return X2D_sk

x2d = dimension_decrease()
x2d_sk = PCA_sk()