# -*- encoding:utf-8 -*-

"""
 Python中对字节流/二进制流的操作:struct模块简易使用教程
 在struct模块中，将一个整型数字、浮点型数字或字符流（字符数组）转换为字节流（字节数组）时，
 需要使用格式化字符串fmt告诉struct模块被转换的对象是什么类型，
 比如整型数字是'i'，浮点型数字是'f'，一个ascii码字符是's'。

"""

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
    buf3 = str("good")
    bin_buf3 = struct.pack('s', buf3)
    # ret3 = struct.unpack('11s', bin_buf3)
    # print(bin_buf3,"<==>",ret3)



demo1()

