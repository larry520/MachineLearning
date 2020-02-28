# -*- encoding:UTF-8 -*-

from minist.unpack import *
import numpy as np

train_data = load_train_images()
train_label = load_train_labels()
test_data  = load_test_images()
test_label = load_test_labels()

# X_train = train_data.reshape(-1,784)

shuffle_index = np.random.permutation(60000)
X_train = train_data[shuffle_index]
X_label = train_label[shuffle_index]

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
