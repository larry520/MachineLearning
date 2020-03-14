# coding=utf-8
# 网页爬虫
import pandas as pd
from urllib import request
from lxml import etree
import os
import datetime
import json

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

header_dict = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36",
}


def get_http(load_url, header):
    """爬取原始html"""
    res = ""
    try:
        req = request.Request(url=load_url, headers=header)  # 创建请求对象
        coonect = request.urlopen(req)  # 打开该请求
        byte_res = coonect.read()  # 读取所有数据，很暴力
        try:
            res = byte_res.decode(encoding='utf-8')
        except:
            res = byte_res.decode(encoding='gbk')
    except Exception as e:
        print(e)
    return res

def data_crab():
    # 丁香园数据
    url = "https://ncov.dxy.cn/ncovh5/view/pneumonia_peopleapp?from=timeline&isappinstalled=0"

    time = str(datetime.datetime.now()).replace(" ",'_').replace(":","-").split(".")[0]

    path = "/Users/dmsoft/MachineLearning/NewPro/origin_data/"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+time+".html", "w", encoding="utf-8") as f:
        page = get_http(url, header_dict)
        f.write(page)
        print("html saved success!")


    page = get_http(url, header_dict)
    tree = etree.HTML(page)
    web_text = tree.xpath("//script[@id='getAreaStat']/text()")
    if len(web_text)==0 :
        print("解析错误！")
    content = web_text[0].replace("try { window.getAreaStat =",'').replace("}catch(e){}", '')
    # content = json.loads(content)  # 可处理复杂的字典嵌套
    content = eval(content)
    content_DF = pd.DataFrame(content)
    content_DF.to_csv(path+time+".csv")
    print("page_crab 获取 " + str(datetime.datetime.now()) + " 数据成功!")





