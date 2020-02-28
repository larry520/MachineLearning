# -*- encoding:utf-8 -*-
import json

# region  Json 序列化与反序列化
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

demo1()
# endregion



