# -*- coding: UTF-8 -*-

import re
import sys
import json
import requests
import os
import multiprocessing
from bs4 import BeautifulSoup


def getNewsUrl(*params):
    url_param = {
        "channel": params[0],
        "page": str(params[1]),
        "show_all": "1",
        "show_num": "5000",
        "tag": "1",
        "format": "json",
    }

    # 构造api的url
    api_url = "http://api.roll.news.sina.com.cn/zt_list?"
    for key, value in url_param.items():
        api_url += key + "=" + value + "&"
    api_url = api_url[:-1]

    # 请求api数据
    response = requests.get(api_url)
    data = json.loads(response.content)
    data = data["result"]["data"]

    # 提取新闻的url
    for term in data:
        url = term["url"]
        yield url


def loadNews(*params):
    url = params[0]
    clsf = params[1]

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # 提取新闻id
    news_id = "-".join(url.split("/")[-2:])
    news_id = re.sub(r"\..*", "", news_id)

    # 提取新闻内容
    news_content = [p.text for p in soup.select("p") if not p.findChildren()]
    news_content = "\n".join(news_content)

    # 确认目录是否已存在
    dir_path = "data/" + clsf
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 存储新闻文本
    if len(news_content) >= 30:
        path = "data/" + clsf + "/" + news_id + ".txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(news_content)


def runSpider(clsf, max_count=1000):
    pool = multiprocessing.Pool(20)
    count = 0
    page = 1
    while count < max_count:
        urls = getNewsUrl(clsf, page)
        print(f"Page {page} - Crawling...")
        for url in urls:
            loadNews(url, clsf)
            pool.apply_async(loadNews, (url, clsf))
            count += 1
            if count >= max_count:  # 达到设定数量，停止抓取
                break
        page += 1
        if count >= max_count:  # 如果已经抓取了1千条数据，停止
            break
    pool.close()
    pool.join()
    print(f"Finished crawling {count} news articles.")


if __name__ == "__main__":
    # sports tech finance edu ent games fashion mil(news)
    runSpider("mil")
