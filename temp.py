# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np

import hashlib
def test_set_check(identifier,test_ratio,hash):
    a = hash(np.int64(identifier)).digest()
    print(int(a[-1]))
    b = a[-1]<256*test_ratio
    return b
    # return hash(np.int64(identifier)).digest()[-1]<256*test_ratio

def split_train_test_by_id(data,test_ratio,id_column,hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_:test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set],data.loc[in_test_set]

np.random.seed(6)
a = np.random.permutation(10).reshape(5,2)
print(a)

b = pd.DataFrame(a,columns=["one","two"])
print(b)
b = b.reset_index()
c = b.iloc[[1,3,2]]
print("C:\n",c)
d = b.iloc[1]
print("D:\n",d)
train,test = split_train_test_by_id(b,0.1,"index")
print(test.info())