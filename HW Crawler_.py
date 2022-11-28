#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 02:48:15 2022

@author: silvestradesi
"""
import urllib.request as req
url="https://www.ptt.cc/bbs/HatePolitics/index.html"
# Create a Request, attach the Request Headers information.
request=req.Request(url, headers={
    "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
})
with req.urlopen(request) as response:
    data=response.read().decode("utf-8")
print(data) 

import requests
from bs4 import BeautifulSoup
#import jieba
from string import punctuation, ascii_letters, digits
import nltk

def ptt_scraping(url):
    articles = []
    r = requests.get(url=URL, cookies={"over18":"1"})
    soup = BeautifulSoup(r.text, "lxml")
    tag_divs = soup.find_all("div", class_="r-ent")
    for tag in tag_divs:
        if tag.find("a"):
            href = tag.find("a")["href"]
            title = tag.find("a").text
            
            r2=requests.get(url="http://ptt.cc"+href, cookies={"over18":"1"})
            soup2 = BeautifulSoup(r2.text, "lxml")
            articles.append({"title":title, "href":href, "text":soup2.text})
    return articles

#URL = "http://ptt.cc/bbs/HatePolitics"
URL = "https://www.ptt.cc/bbs/Tech_Job"
print(URL)

articles = ptt_scraping(url=URL)
for i in range(5):
    article = articles[i]
    filename = article["href"].split("/")[-1]
    with open(file=filename+".txt", mode="w", encoding="utf8") as file1:
        file1.write(article["title"] + "\n")
        file1.write(article["text"])
    print("full-href", URL[:13] + article["href"])
    print("title", article["title"])

