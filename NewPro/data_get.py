# coding=utf-8
import pandas as pd
from urllib import request
from lxml import etree
import os
import ssl
import datetime
import time

from NewPro import page_crab


import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

def html_to_csv(file,filepath,savepath):
    """从HTML中提取信息并存储为CSV"""
    file_name = file.replace(".html",'')
    with open(filepath+file , "r", encoding="utf-8") as f:

        page = f.read()
        tree = etree.HTML(page)

        web_textChina = tree.xpath("//script[@id='getAreaStat']/text()")
        web_textWorld = tree.xpath("//script[@id='getListByCountryTypeService2true']/text()")

        if len(web_textChina) == 0:
            print("解析错误！")
        content = web_textChina[0].replace("try { window.getAreaStat =", '').replace("}catch(e){}", '')
        # content = json.loads(content)  # 可处理复杂的字典嵌套
        content = eval(content)

        if len(web_textWorld) == 0:
            print("解析错误！")
        content2 = web_textWorld[0].replace("try { window.getListByCountryTypeService2true =", '').replace("}catch(e){}", '')
        # content = json.loads(content)  # 可处理复杂的字典嵌套
        content_DF2 = pd.DataFrame(eval(content2))
        content_DF2.to_csv(savepath+ file_name+"_Nation.csv")

        content_DF = pd.DataFrame(content)
        content_DF.to_csv(savepath + file_name + ".csv")
        print("获取 " + file + " 数据成功!")

if __name__ == "__main__":

    page_crab.data_crab()

    filepath = "/Users/dmsoft/MachineLearning/NewPro/origin_data/"
    files = list(os.walk(filepath))
    file_names = files[0][2]
    data_list = []
    for i in file_names:
        """获取所有的html文件"""
        if "html" in i:
            data_list.append(i)

    savepath = "/Users/dmsoft/MachineLearning/NewPro/csv_data/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for file in data_list:
        if not os.path.exists(savepath + file.replace(".html",'.csv')):
            html_to_csv(file,filepath,savepath)
    # while True:
    #     a = 0
    #     if a ==5:
    #         print("go on ...")
    #         a = 0
    #     time.sleep(1)


