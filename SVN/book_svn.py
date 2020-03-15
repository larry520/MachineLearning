#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: dmsoft
@file: book_svn.py
@time: 2020/03/10
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV


import time

class svm_clf_gather(object):
    """
    硬间隔分类、软间隔分类  对特征缩放敏感
    """
    def __init__(self,show=False):
        self.iris = load_iris()
        self.data = self.iris["data"]
        if show:
            x = np.arange(1,151)
            for i in range(4):
                y = (self.data[:,i]).ravel()
                plt.scatter(x,y,label="feature "+str(i))
                plt.legend()
                plt.pause(0.01)
        self.x = self.iris["data"][:,(2,3)]  # petal length, petal width
        self.y = (self.iris["target"]==2).astype(np.float64) # Iris-Virginica

        svm_clf = Pipeline([
            # LinearSVC 会对偏置项正则化，需先减去平均值使训练集集中
            # StandardScaler 会对数据预处理
            ("scaler", StandardScaler()),
            ("linear_svc", LinearSVC(C=1, loss="hinge")),
            ])
        svm_clf.fit(self.x,self.y)
        print(svm_clf.predict([[2.5,1.7]]))

    def veriy(self,model,figure):
        """验证moons数据训练模型预测边界"""
        n_x1, n_x2 = np.meshgrid(np.linspace(-1,2,20),np.linspace(-0.6,1.5,20))
        n_x1 = n_x1.reshape(-1,1)
        n_x2 = n_x2.reshape(-1,1)
        n_x = np.concatenate((n_x1,n_x2),axis=1)
        y_pred = model.predict(n_x)
        plt.figure(figure)
        for i in range(len(n_x1)):
            if y_pred[i] == 0:
                plt.scatter(n_x1[i], n_x2[i], c="g")
            else:
                plt.scatter(n_x1[i], n_x2[i], c="#EE82EE")
        plt.pause(0.0001)

    def add_feature(self):
        """线性不可分数据集通过增加特征可能会导致其可分"""
        from sklearn.datasets import make_moons
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        from sklearn.svm import SVC
        from sklearn import clone

        svm_method = [ 1,2 ,3] # 测试方法类别
        """相比于LinearSVC, SVC 不需要真的添加新特征达到高阶多项式效果
        可有效果避免因特征增加导致数据爆炸风险"""
        x = make_moons(n_samples=100, shuffle=True, noise=0.1, random_state=None)
        x1, x2 = x[0][:,0],x[0][:,1]
        y = x[1]
        # plt.figure("Real")
        def origin_plot(self,figure):
            plt.figure(figure)
            for i in range(len(x1)):
                if y[i] == 0:
                    plt.scatter(x1[i],x2[i],c="r")
                else:
                    plt.scatter(x1[i],x2[i],c="b")

        if 1 in svm_method :
            """通过增加特征,增加线性可分可能性，此方法针对大量特征有可能导致数据爆炸"""
            polynomial_svm_clf = Pipeline([
                ("poly_features", PolynomialFeatures(degree=10)),
                ("scaler", StandardScaler()),
                ("svm_clf", LinearSVC(C=5, loss="hinge", max_iter=1000))
            ])
            t1 = time.time()
            polynomial_svm_clf.fit(x[0],y)
            t2 = time.time()
            print("runtime: ",(t2-t1),"s")
            origin_plot(self,"LinearSVC")
            self.veriy(polynomial_svm_clf,"LinearSVC")

        if 2 in svm_method:
            """多项式核 SVC 通过算式得到增加多特征效果"""
            poly_kernel_svm_clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel="poly", degree=10, coef0=1, C=5))
            ])
            t1 = time.time()
            poly_kernel_svm_clf.fit(x[0],y)
            t2 = time.time()
            print("runtime: ", (t2 - t1), "s")
            origin_plot(self,"poly_SVC")
            self.veriy(poly_kernel_svm_clf, "poly_SVC")

        if 3 in svm_method:
            """高斯RBF核"""
            rbf_kernel_svm_clf = Pipeline([
                ("scalar",StandardScaler()),
                ("svm_clf", SVC(kernel="rbf", gamma=10, C=0.001))
            ])
            t1 = time.time()
            rbf_kernel_svm_clf.fit(x[0],y)
            t2 = time.time()
            print("runtime: ", (t2 - t1), "s")
            origin_plot(self, "rbf_SVC")
            self.veriy(rbf_kernel_svm_clf,"rbf_SVC")
        if 666 in svm_method:
            """测试模型参数"""
            def featrues_T(self,gamma=10.0, C=0.001):
                rbf_kernel_svm_clf = Pipeline([
                    ("scalar", StandardScaler()),
                    ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
                ])
                rbf_kernel_svm_clf.fit(x[0], y)
                title ="gamma="+str(gamma)+"C="+str(C)
                origin_plot(self, title)
                self.veriy(rbf_kernel_svm_clf, title)

            featrues_T(self,0.1,0.001)
            featrues_T(self, 0.1, 1000)
            featrues_T(self, 5, 0.001)
            featrues_T(self, 5, 1000)

class svm_reg_gather(object):
    def __init__(self):
        m = 100
        self.x =2*np.random.rand(m,1)
        self.y = 5*self.x + 3 + np.random.rand(m,1)
    def svm_reg(self):
        """epsilon 间隔系数 成正比"""
        lin_svm_reg = LinearSVR(epsilon=1, max_iter=2000)
        lin_svm_reg.fit(self.x, self.y.ravel())
        print(lin_svm_reg.coef_,lin_svm_reg.intercept_)
        svm_poly_reg = SVR(kernel="poly", degree=2,
                           C=0.01, epsilon=0.1)
        svm_poly_reg.fit(self.x,self.y.ravel())
        # print(svm_poly_reg.coef_,svm_poly_reg.intercept_)
        lin_reg = LinearRegression()
        lin_reg.fit(self.x,self.y)

        plt.scatter(self.x,self.y)
        plt.scatter(self.x,lin_svm_reg.predict(self.x))
        plt.scatter(self.x, svm_poly_reg.predict(self.x))
        plt.scatter(self.x,lin_reg.predict(self.x))

from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data[:, 2:]  # petal(花瓣) length and width
y = iris.target
class decision_tree_method(object):


    def decision_tree_clf(self,x,y):
        """如果不限制最大深度，会直到做到基尼系数为0
            min_samples_split 分裂前节点必须有最小分类数
            min_samples_leaf 叶节点最小样本数量
            min_weight_fraction_leaf 同上，计量方式加权实例总数的占比
            max_leaf_nodes 最大叶节点数量
            max_features 最大特征数量

        """
        self.tree_clf = DecisionTreeClassifier(max_depth=5)
        self.tree_clf.fit(x,y)
        return self.tree_clf

    def decision_tree_reg(self,x,y):
        self.tree_reg = DecisionTreeRegressor(max_depth=2)
        self.tree_reg.fit(x,y)
        return self.tree_reg

    def tree_view(self,model,filename="iris.dot"):
        from sklearn.tree import export_graphviz
        data = export_graphviz(model,
            out_file=filename,
            feature_names=iris.feature_names[2:],
            class_names=iris.target_names,
            rounded=True,
            filled=True
        )
        return data

def test_decision_tree_method():
    clf = decision_tree_method()
    clf_model = clf.decision_tree_clf(x,y)
    dot_data = clf.tree_view(clf_model,"iris_clf.dot")

    reg_model = clf.decision_tree_reg(x,y)
    clf.tree_view(reg_model,"iris_reg.dot")

    """混淆矩阵"""
    # pred = model.predict(x)
    # conf_mx = confusion_matrix(y,pred)
    # row_sums = conf_mx.sum(axis=1,keepdims=True)  # 缩成1列，行求和
    # norm_conf_ms = conf_mx/row_sums
    # # 用 0 填充矩阵的对角值
    # np.fill_diagonal(norm_conf_ms,0)
    # plt.matshow(norm_conf_ms,cmap="gray")

# test_decision_tree_method()

# def tree_moons(figure):
from sklearn.model_selection import train_test_split
import pandas as pd
data =make_moons(n_samples=1000, noise=0.4)
data = np.concatenate((data[0],data[1].reshape(-1,1)),axis=1)
train_set,test_set = train_test_split(data,test_size=0.2,random_state=42)
x_train = train_set[:,:2]
y_train = train_set[:,2]
x_test = test_set[:,:2]
y_test = test_set[:,2]
# plt.figure("Real")
def origin_plot(figure, x=x_train,y=y_train):
    plt.figure(figure)
    for i in range(len(x)):
        if y[i] == 0:
            plt.scatter(x[i,0],x[i,1],c="r")
        else:
            plt.scatter(x[i,0],x[i,1],c="b")
    plt.xlabel("x1")
    plt.ylabel("x2")
# origin_plot(figure=1)

decision_tree_clf = DecisionTreeClassifier()
param_grid = [
    {"max_depth":[3,6,9,15,20],'max_leaf_nodes':[5,10,50,100],"min_samples_split":[2,5,10,20]}

]
grid_search = GridSearchCV(decision_tree_clf, param_grid, cv=3,
                           scoring='neg_mean_squared_error')
grid_search.fit(x_train,y_train)
print(grid_search.best_params_, grid_search.best_estimator_)

best_estimator_ = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=3, max_features=None, max_leaf_nodes=5,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')

best_estimator_.fit(x_train,y_train)
y_pred = best_estimator_.predict(x_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score
print(precision_score(y_test,y_pred))
cmx = confusion_matrix(y_test,y_pred)
row_sum = np.sum(cmx,axis=1)


def tree_view( model, filename="moon.dot"):
    from sklearn.tree import export_graphviz
    data = export_graphviz(model,
                           out_file=filename,
                           rounded=True,
                           filled=True
                           )
    return data
tree_view(best_estimator_)