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
    print("n_correct/len(y_pred): " ,n_correct/len(y_pred))

# cross_val_score() 交叉验证打分函数,得到准确率 功能同上
from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, X_label_5, cv=3, scoring="accuracy"))

# -----------使用混淆矩阵可有效评估分类器的性能-----------
from sklearn.model_selection import cross_val_predict
# 执行K-fold 交叉验证，返回每个折叠的预测
X_train_pred = cross_val_predict(sgd_clf, X_train,X_label_5, cv=3)

# confusion_matrix(A,B) A:行表示类别 B 列表示类别
from sklearn.metrics import confusion_matrix
res_array = confusion_matrix(X_label_5,X_train_pred)
print("res_array: ",res_array)
#-------*************************************-----------------
#                      预测
#                 负类      正类
#         负类    [[52627(TN)  1952 (FP)]
# 实际
#         正类    [  1017(FN)  4404(TP)]]
# precision_score 精度： TP/(TP + FP)
# recall_score 召回率： TP/(TP + FN)
# F1分数 2/(1/精度 + 1/召回率）= TP/(TP + (FN +FP)/2)
# 真正类率 = 召回率 = TP/(TP + FN)
# 假正类率 FPR = FP/(TN + FP)
# 真负类率/特异度 TNR = TN/(TN + FP)


from sklearn.metrics import precision_score, recall_score, f1_score
precisionScore = precision_score(X_label_5,X_train_pred)
recallScore = recall_score(X_label_5,X_train_pred)
f1Score = f1_score(X_label_5,X_train_pred)

# 对图片进行唐预测并返回打分
y_scores = sgd_clf.decision_function([X_train[4],X_train[5]])  # X_train[5] ：图片
# 调节决策阀值对精度/召回率权衡  阀值提高，精度提升，召回率下降
threshold = 0
# 返回预测结果 bool
y_some_digig_pred = (y_scores>threshold)

# -----**--获取所有实例的分数
y_scores = cross_val_predict(sgd_clf,X_train,X_label_5,cv=3,
                             method="decision_function")
#
from sklearn.metrics import precision_recall_curve
precisions, recalls, threshold = precision_recall_curve(X_label_5,y_scores)
def plot_precision_recall_vs_threshold(precisions,recalls, threshold):
    plt.plot(threshold,precisions[:-1],"b--",label="Precision")
    plt.plot(threshold,recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])  # 设置 y 参数范围

plot_precision_recall_vs_threshold(precisions, recalls, threshold)
plt.show()
# 根据曲线获取目标精度的阈值，得到特定精度的分类器。
X_train_pred_90 = (y_scores > 7000)
precisionScore_90 = precision_score(X_label_5,X_train_pred_90)
recallScore_90 = recall_score(X_label_5,X_train_pred_90)

from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(X_label_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1], "k--")
    plt.axis([0, 1, 0, 1])  # [ xmin, xmax, ymin, ymax ]
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

plot_roc_curve(fpr, tpr)
plt.show()

# ROC曲线 与 ROC AUC分数用于比较不同分类器
from sklearn.metrics import roc_auc_score
roc_auc_score(X_label_5, y_scores)

# -----------*****随机森林分类器
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
# method="decision_function" 返回分数
# "predict_proba"返回概率 每行为一个实例，每列为对应实例属于给定类别的概率
y_probas_forest = cross_val_predict(forest_clf,X_train,X_label_5,cv=3,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(X_label_5,y_scores_forest)
plt.plot(fpr,tpr,"b:",label="SGD")
plot_roc_curve(fpr_forest, tpr_forest,"Random Forest")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(X_label_5, y_scores_forest)
X_train_forest_pred = cross_val_predict(forest_clf, X_train,X_label_5, cv=3)
precision_score(X_label_5,X_train_forest_pred)

#-------***** 训练结果优化方法之一 输入进行缩放
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

cross_val_score(sgd_clf,X_train_scaled,X_label,cv=3,scoring="accuracy")
# 交叉验证训练后返回的预测值
X_train_pred = cross_val_predict(sgd_clf,X_train_scaled, X_label,cv=3)

# 矩阵图像显示 plt.matshow()  混淆矩阵 所有类
conf_mx = confusion_matrix(X_label, X_train_pred)
plt.matshow(conf_mx)
# keepdims 保留矩阵维度
row_sums = conf_mx.sum(axis=1,keepdims=True)
norm_conf_ms = conf_mx/row_sums
# 用 0 填充矩阵的对角值
np.fill_diagonal(norm_conf_ms,0)
plt.matshow(norm_conf_ms,cmap=plt.cm.get_cmap("gray"))

cl_a, cl_b = 3, 5
X_aa = X_train[(X_label==cl_a) & (X_train_pred==cl_a)]
X_ab = X_train[(X_label==cl_a) & (X_train_pred==cl_b)]
X_ba = X_train[(X_label==cl_b) & (X_train_pred==cl_a)]
X_bb = X_train[(X_label==cl_b) & (X_train_pred==cl_b)]


# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28 # minist 数据集中图片尺寸为28*28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances] #对每行数据还原为图片
    n_rows = (len(instances) - 1) // images_per_row + 1  # // 向下取整的除法
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=plt.cm.get_cmap("binary"), **options)
    plt.axis("off")

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# save_fig("error_analysis_digits_plot")
plt.show()

# 多标签分类 操作方法同单标签，只不过label有两类
from sklearn.neighbors import KNeighborsClassifier
X_label_large = (X_label>=7)   # 返回同类型数组，元素为 True False
X_label_odd = (X_label % 2 == 1)  # 返回同类型数组，元素为 True False
X_multilabel = np.c_[X_label_large,X_label_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, X_multilabel)

X_trian_knn_pred = cross_val_predict(knn_clf,X_train,X_label,cv=3)

