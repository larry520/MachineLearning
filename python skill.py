# -*- encoding:utf-8 -*-
import json
import numpy as np

import matplotlib

# 解决中文字体显示异常
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
# 解决负号’-‘显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False


# region  Json 序列化与反序列化
def json_serialize():
    d = '{"name":"小明","age":18,"skill":["开飞机","铁锅炖大鹅"]}'
    d_json = json.loads(d)  # 可处理复杂的字典嵌套
    d_eval = eval(d)  # 系统自带

    print(d_eval,d_json)

    class Student(object):
        def __init__(self, name, age, score):
            self.name = name
            self.age = age
            self.score = score

    s = Student('Bob', 20, 88)
    # default用于对复杂结构定义序列化时转化规则
    # 详情看 https://www.liaoxuefeng.com/wiki/1016959663602400/1017624706151424
    s_json = json.dumps(s, default=lambda obj:obj.__dict__)
    print(type(s_json),s_json)
    s_recall = json.loads(s_json)
    print(type(s_recall), s_recall)
# json_serialize()
# endregion

# region 二进制流 struck module
import struct

def demo1():
    # 使用bin_buf = struct.pack(fmt, buf)将buf为二进制数组bin_buf
    # 使用buf = struct.unpack(fmt, bin_buf)将bin_buf二进制数组反转换回buf
    # 默认小端存储（高有效位存储在高地址中）

    # int => binary stream
    buf1 = 256
    bin_buf1 = struct.pack('i',buf1)
    ret1 = struct.unpack('i',bin_buf1)
    print(bin_buf1,"<==>",ret1)

    # float => binary stream
    buf2 = 5.20
    bin_buf2 = struct.pack('d', buf2)
    ret2 = struct.unpack('d', bin_buf2)
    print(bin_buf2,"bin_buf2.type:",type(bin_buf2),"<==>",ret2, "ret2.type:",type(ret2))

    # string => binary stream
    t_bin = "雅典娜 hello world".encode('utf-8')
    t = t_bin.decode("utf-8")
    print(t_bin,"t_bin.type:",type(t_bin),"<==>",t,"t.type:",type(t))

# demo1()
# endregion

# region numpy array 合并、连接、降维

# np.concatenate 将列表进行连接，多个列表则首项连接，单个列表则降维连接
    """
    numpy.flatten()/numpy.ravel() 将多维数组降位一维 默认行优先，x.ravel("F") 则列优先
    两者的区别在于返回拷贝（copy）还是返回视图（view），
    numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵
    而numpy.ravel()返回的是视图（view），会影响（reflects）原始矩阵。
    https://blog.csdn.net/liuweiyuxiang/article/details/78220080
    """

import numpy as np
import pandas as pd
def array_con():
    a = np.zeros((3,3))
    b = np.ones((3,3))
    c = np.concatenate((a,b),axis=0)  # 行连接，列数不变 np.r_([a,b])
    d = np.concatenate((a,b),axis=1)  # 列连接，行数不变 np.c_([a,b])
    n_zero = np.zeros((5,2,3,3))
    n_one =np.ones((2,2,3,3))
    cmb = np.concatenate((n_one,n_zero),axis=0)
    cmb_one = np.concatenate(n_one)   # shape [2.2.3.3] --> [4,3,3]
    a_ = pd.DataFrame(a)
    b_ = pd.DataFrame(b)
    c_ = pd.concat((a_,b_),axis=0)  # 行连接，列数不变
    d_ = pd.concat((a_,b_),axis=1)  # 列连接，行数不变
    e_ = pd.merge(a_,b_,how="outer") # 上下联结
    e = e_[1].tolist()
array_con()

# endregion

# ----数据筛选
array_a = np.arange(1,20)
array_b = (array_a%2 == 0)
array_c = array_a[(array_a%2 == 0)]
array_d = array_b.astype(int)  # bool to int
array_e = array_a[[2,5,8,3,6,9]]  # 索引 array_a 中对应的数值

list_a = list(range(10))
list_b = list(range(20,30))

for i in range(len(list_b)):
    print(list_a[i],list_b[i])

n_x1, n_x2 = np.meshgrid(np.linspace(-1,2,30),np.linspace(-0.6,1,30))

n_x1 = n_x1.reshape(-1,1)
n_x2 = n_x2.reshape(-1,1)
n_x = np.concatenate((n_x1,n_x2),axis=1)
pass
print(n_x)




