#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author:dmsoft
@file: random_tree.py
@time: 2020/03/16
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ML_basic import sklearn_model_view as skview
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
from ML_basic.minist.unpack import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



data = make_moons(n_samples=1000, noise=0.2)
data = np.concatenate((data[0],data[1].reshape(-1,1)), axis=1)
train_set,test_set = train_test_split(data,test_size=0.2)
x = train_set[:,:2]
y = train_set[:,2]
x_test = test_set[:,:2]
y_test = test_set[:,2]

def moondata_view():
    for i in range(len(data[:,0])):
        if data[i,2]==0:
            plt.scatter(data[i,0],data[i,1], c="r")
        else:
            plt.scatter(data[i,0],data[i,1], c="b")
    plt.show()

def vote(votetype="hard"):
    """
    集成学习 结合多个分类器结果并输出
    :param votetype: "hard" 硬投票 收集所有分类器投票数最高的作为输出
    "soft" 软投票 收集所有分类器对各结果的概率，取概率加权平均最高的作为输出
    :return:
    """
    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC(probability=True) # 设置为True 增加predict_proba()方法
    voting_clf = VotingClassifier(
        estimators=[('lf', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting=votetype
    )
    # voting_clf.fit(x,y)

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(x, y)
        y_pred = clf.predict(x_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

def bagging():
    """
    bagging:从总样本中采样训练后数据放回  bootstrap=True
    pasting:从总样本中采样训练后数据从总样本中剃除 bootstrap=False
    """
    # n_jobs 表示用几个CPU内核进行训练，-1表示所有核
    # n_estimators 采用集成模型个数
    # max_samples 每次随机从训练集中采样实例个数
    # oob_score 请求是否进行包外评估 True 是
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500
        ,max_samples=100, bootstrap=True, n_jobs=-1
        ,oob_score=True
    )
    bag_clf.fit(x,y)
    y_pred =  bag_clf.predict(x_test)
    print(bag_clf.oob_score_)     # 得到包外预测的正确率
    print(bag_clf.oob_decision_function_) # 得到训练集每个实例所属类别的概率
    print(accuracy_score(y_test,y_pred))

    # moondata_view()  # 显示原始数据
    # vote("hard")      # 集成学习方法 硬投票法
    # vote("soft")      # 集成学习方法，软投票法

    skview(bag_clf,x[:,0],x[:,1],y)

def random_forest():
    """
    除少数例外，随机森林拥有 BaggingClassifier 和 DecisionClassifier 所有的超参数
    越重要的特征越出现在靠近根节点的位置，估通过计算特征在森林中所有树上的平均深度可估算出一个特征的重要程度
    通过变量 feature_importance_ 可输出各特征的重要比例

    :ExtraTreesClassifier() 极端随机树 速度更快，节点分裂对特征使用随机阈值，而不是搜索最佳阈值
    :return:
    """
    rnd_clf = RandomForestClassifier(n_estimators=500
                                      # ,max_leaf_nodes=16
                                     ,n_jobs=-1,oob_score=True)
    # rnd_clf.fit(x,y)
    # skview(rnd_clf, x, x[:, 1], y)
    # print(rnd_clf.feature_importances_)

    iris = load_iris()
    rnd_clf.fit(iris["data"],iris["target"])# 只用重要特征做分类准确率不一定
    for name,score in zip(iris["feature_names"],rnd_clf.feature_importances_):
        print(name,score)
    # skview(rnd_clf,iris["data"][:,2],iris["data"][:,3],iris["target"])

    et_clf = ExtraTreesClassifier(n_estimators=100
                                  ,bootstrap=True
                                  ,n_jobs=-1,oob_score=True
                                  )
    et_clf.fit(iris["data"], iris["target"])
    print(rnd_clf.oob_score_, et_clf.oob_score_)

def boosting():
    """
    adaptive boosting：AdaBoost 自适应提升法 训练器串行，不利于拓展
    gradient boosting DecisionTree 梯度提升

    :return:
    """
    def ada_boosting():
        ada_clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1),n_estimators=200
            ,algorithm="SAMME.R", learning_rate=0.5
        )
        ada_clf.fit(x,y)
        y_pred = ada_clf.predict(x_test)
        print("ada_boosting accuracy: ",accuracy_score(y_test,y_pred))

    ada_boosting()

    def GBDT():
        """
        :n_estimators: 控制迭代次数
        :learning_rate: 学习率，设置较小时需要更多的迭代次数
        早期停止法： model.staged_predict() 所有逐步预测的结果迭代器
        :return:
        """

        x = np.linspace(0,10,100).reshape(-1,1)
        y = x**2 - 10* x + 2*np.random.randn(100,1)

        def gbrt_sk():

            gbdt_reg = GradientBoostingRegressor(max_depth=2,n_estimators=4
                                             ,learning_rate=1)
            gbdt_reg.fit(x,y)
            plt.figure("subplot")
            plt.subplot(221), plt.scatter(x,y),plt.plot(x,gbdt_reg.predict(x),c="r",label="gbrt_sk"),plt.legend()

            print("gbrt_sk mean squared error:", mean_squared_error(y,gbdt_reg.predict(x)))
        gbrt_sk()

        def realize():
            """手动实现"""
            tree_reg = []
            y_pred = []
            y_pred_sum =[]

            for i in range(4):
                if i == 0:
                    y_pred.append(y)
                tree_reg.append(DecisionTreeRegressor(max_depth=2))
                tree_reg[i].fit(x, y_pred[i])
                y_pred.append( y_pred[i] - tree_reg[i].predict(x).reshape(-1,1))

            for i in range(len(tree_reg)):
                y_pred_sum.append(sum(tree.predict(x) for tree in tree_reg[:i+1]))

            plt.figure("realize")
            plt.subplot(221), plt.scatter(x,y),plt.plot(x,y_pred_sum[0],c="r",label="1"),plt.legend()
            plt.subplot(222), plt.scatter(x,y),plt.plot(x,y_pred_sum[1],c="r",label="2"),plt.legend()
            plt.subplot(223), plt.scatter(x,y),plt.plot(x,y_pred_sum[2],c="r",label="3"),plt.legend()
            plt.subplot(224), plt.scatter(x,y),plt.plot(x,y_pred_sum[3],c="r",label="4"),plt.legend()

        def gbrt_01():
            """通过使用不同迭代次数后的模型进行预测，选择方差最小模型对应的迭代次数"""
            x_train,x_val,y_train,y_val = train_test_split(x,y)

            gbrt = GradientBoostingRegressor(max_depth=2,n_estimators=120)
            gbrt.fit(x_train,y_train.ravel())

            errors = [mean_squared_error(y_val,y_pred.reshape(-1,1))
                for y_pred in gbrt.staged_predict(x_val)]
            bst_n_estimators = np.argmin(errors)
            gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
            gbrt_best.fit(x_train,y_train.ravel())
            print("gbrt_01 mean squared error:", mean_squared_error(y_val,gbrt_best.predict(x_val))
                  ,"gbrt_best.n_estimators: ",gbrt_best.n_estimators)

            plt.figure("subplot")
            plt.subplot(222), plt.scatter(x, y), plt.plot(x, gbrt_best.predict(x), c="r", label="gbrt_01"), plt.legend()

        # realize()
        # gbrt_sk()
        gbrt_01()

        def gbrt_02():
            """逐渐增加迭代次数，当连续5次都没有改善后使用最小方差的迭代次数
            超参数 warm_start=True 会保留现有的树，从而实现增量训练"""
            x_train, x_val, y_train, y_val = train_test_split(x, y)
            gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)

            min_val_error = float("inf")
            error_going_up = 0
            for i in range(1,120):
                gbrt.n_estimators = i
                gbrt.fit(x_train,y_train.ravel())
                val_error = mean_squared_error(y_val,gbrt.predict(x_val))
                if val_error<min_val_error:
                    min_val_error = val_error
                    error_going_up = 0
                else:
                    error_going_up += 1
                    if error_going_up == 5:
                        gbrt.n_estimators = i-5
                        break
            print("gbrt_02 mean squared error: ", mean_squared_error(y_val,gbrt.predict(x_val)),
                  "gbrt.n_estimators: ",gbrt.n_estimators)

            plt.figure("subplot")
            plt.subplot(223), plt.scatter(x, y), plt.plot(x, gbrt.predict(x), c="r", label="gbrt_02"), plt.legend()

        gbrt_02()

    GBDT()

def summary():

    x_image = load_train_images()
    x_label = load_train_labels()
    x_train,x_val,y_train,y_val = train_test_split(x_image[:5000],x_label[:5000],train_size=0.8)

    x_train = x_train.reshape(-1,784)
    x_val = x_val.reshape(-1,784)

    log_clf = LogisticRegression(max_iter=200)
    rnd_clf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
    rbf_svm_clf = Pipeline([
        # ("scalar",StandardScaler()),
        ("rbf_svm", SVC(kernel="rbf", gamma="scale", C=1, probability=False))
    ])

    et_clf = ExtraTreesClassifier(n_estimators=100
                                  ,bootstrap=True
                                  ,n_jobs=-1,oob_score=True
                                  )
    gbdt_cl = GradientBoostingClassifier(n_estimators=200
                                         ,learning_rate=0.5
                                         ,max_depth=2
                                         )
    voting_clf = VotingClassifier(
        estimators=[('lf', log_clf), ('rf', rnd_clf), ('rbf_svc', rbf_svm_clf)
                    ,('et', et_clf),# ('gbdt', gbdt_cl)
                    ],
        voting="hard"
        )
    # (rnd_clf,rbf_svm_clf,et_clf,gbdt_cl,voting_clf)
    for clf in (log_clf,rnd_clf,et_clf,rbf_svm_clf,voting_clf):
        clf.fit(x_train,y_train.ravel())
        y_pred = clf.predict(x_val)
        print(clf.__class__.__name__,accuracy_score(y_val,y_pred))
