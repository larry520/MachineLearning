# -*- encoding:UTF-8 -*-

from ML_basic.minist.unpack import *
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix


train_data = load_train_images()
train_label = load_train_labels()
# test_data  = load_test_images()
# test_label = load_test_labels()

# X_train = train_data.reshape(-1,784)

shuffle_index = np.random.permutation(60000)
X_train = train_data[shuffle_index].reshape(-1,784)
X_label = train_label[shuffle_index]

X_train = X_train[:10000]
X_label = X_label[:10000]


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train,X_label)
X_trian_knn_score = cross_val_score(knn_clf,X_train,X_label,cv=3)

X_trian_knn_pred = cross_val_predict(knn_clf,X_train,X_label,cv=3)
conf_mx = confusion_matrix(X_label, X_trian_knn_pred)
plt.matshow(conf_mx)  # 混淆矩阵图，彩色的
# 计算错误率
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx/row_sums  # 每行除以该类别个数，得到相应错误识别的概率
np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx, cmap=plt.cm.get_cmap("gray"))

cl_a,cl_b = 1,7
X_aa = X_train[(X_label==cl_a) & (X_trian_knn_pred==cl_a)]
X_ab = X_train[(X_label==cl_a) & (X_trian_knn_pred==cl_b)]
X_bb = X_train[(X_label==cl_b) & (X_trian_knn_pred==cl_b)]
X_ba = X_train[(X_label==cl_b) & (X_trian_knn_pred==cl_a)]

def plot_digits(instances,imapges_per_row=5,**options):
    size = 28
    imapges_per_row = min(len(instances), imapges_per_row)
    # images = [instance.reshape(size, size) for instance in instances] #对每行数据还原为图片
    images = instances.reshape(-1,size,size)
    # rows = (len(images) -1)//imapges_per_row + 1
    rows = (images.shape[0] -1)//imapges_per_row + 1
    n_empty = rows*imapges_per_row - len(images)
    images = np.concatenate((images,np.zeros((n_empty,size,size))),axis=0)
    print(images.shape)
    row_images = []
    for row in range(rows):
        rimages = images[row*imapges_per_row:(row+1)*imapges_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    mat_images = np.concatenate(row_images,axis=0)
    plt.imshow(mat_images,cmap=plt.cm.get_cmap("binary"),**options)
    plt.axis("off")

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25],5)
plt.subplot(222); plot_digits(X_ab[:25],5)
plt.subplot(223); plot_digits(X_ba[:25],5)
plt.subplot(224); plot_digits(X_bb[:25],5)


