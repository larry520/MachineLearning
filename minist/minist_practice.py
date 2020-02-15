# -*- encoding:UTF-8 -*-

from minist.unpack import *
import numpy as np


mnist_data = load_train_images()
mnist_label = load_train_labels()

shuffle_index = np.random.permutation(60000)
X_train = mnist_data[shuffle_index]
X_label = mnist_label[shuffle_index]

X_train = X_train.reshape(-1,784)
X_label_5 = (X_label ==5)


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, X_label_5)
# 训练完成，进行预测
res = sgd_clf.predict(X_train)

# 测试集数据
test_data = load_test_images()
test_label = load_test_labels()
y_data = test_data.reshape(-1, 784)
test_label_5 = (test_label ==5)

test_pre = sgd_clf.predict(y_data)
n_correct = sum(test_pre == test_label_5)
print("sgd_clf predict :", n_correct/len(test_label))

# ------ 交叉验证测量精度 ----------
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

# 折叠验证 利用训练集数据进行验证  评估模型
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for  train_index, test_index in skfolds.split(X_train,X_label_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    X_label_5_folds = X_label_5[train_index]
    Y_test_folds = X_train[test_index]
    Y_label_5_folds = X_label_5[test_index]

    clone_clf.fit(X_train_folds, X_label_5_folds)
    y_pred = clone_clf.predict(Y_test_folds)
    n_correct = sum(y_pred == Y_label_5_folds)
    print(n_correct/len(y_pred))

# cross_val_score() 交叉验证打分函数 功能同上
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, X_label_5, cv=3, scoring="accuracy"))

# 使用混淆矩阵可有效评估分类器的性能
from sklearn.model_selection import cross_val_predict



