# -*- encoding: utf-8 -*-

import pandas as pd
import os


path = "./csv_data/"

def get_csv(path):
    files = list(os.walk(path))
    file_names = files[0][2]
    data_list = []
    for i in file_names:
        """获取所有的csv文件"""
        if "csv" in i:
            data_list.append(i)
    data_list.sort()
    return data_list

data_list = get_csv(path)

"""
曲线图
1. 全国总体历史数据分析
    确诊病例，可疑人数，治愈人数，死亡人数 DataFrame
2. 全国各省历史数据分析
3. 对某个省、市进行数据分析
    确诊病例，可疑人数，治愈人数，死亡人数 DataFrame

"""

area = "湖北"

data = pd.DataFrame(columns=("date","area","confirmedCount", "suspectedCount", "curedCount", "deadCount"))

for i,file in enumerate(data_list):
    origin_data = pd.read_csv(path+file)
    row = origin_data[(origin_data["provinceShortName"]==area)].index.tolist()
    if len(row) !=0:
        t1 = origin_data["confirmedCount"][row[0]]
        t2 = file.replace(".csv",'')
        data = data.append([{"date":file.replace(".csv",''),
                          "area":area,
                          "confirmedCount":origin_data["confirmedCount"][row[0]],
                          "suspectedCount":origin_data["suspectedCount"][row[0]],
                          "curedCount":origin_data["curedCount"][row[0]],
                          "deadCount":origin_data["deadCount"][row[0]]}],ignore_index=True)


data.plot()

pass



#
#
# k = pd.read_csv(data_list[0])
# k.plot(x="Unnamed: 0", y="confirmedCount")
#
# """统计全国数据"""
# k["provinceShortName"].sum()


