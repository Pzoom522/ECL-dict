import requests
from bs4 import BeautifulSoup
import codecs
from tqdm import tqdm
import json
import os
import re
import urllib

# 4、根据文件中各个语言的href列表遍历可能存在的xml文件，并将其下载存储到result文件夹中, 按照语言进行文件夹划分
with codecs.open("language_url_dict.json",'r','utf-8') as jf:
    language_url_dict = json.load(jf)

for language, data in language_url_dict.items():
    file_dir = 'result/' + language + '/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    href_list = data['herf_url']
    for href in tqdm(href_list):
        response = requests.get(href)        #Get方式获取网页数据
        soup = BeautifulSoup(response.text, 'lxml')
        xml_list = soup.findAll(name='a',text=re.compile(r".*xml"))
        for xml_f in xml_list:
            urllib.request.urlretrieve(xml_f.get('href'), file_dir + xml_f.get('href').split('/')[-1]) # 将符合要求的xml文件进行下载保存
            # print(xml_f.get('href'))
        