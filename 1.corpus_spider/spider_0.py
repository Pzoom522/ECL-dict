
import requests
from bs4 import BeautifulSoup
import codecs
from tqdm import tqdm
import json

base_url = "http://www.language-archives.org"

entrance_url = "http://www.language-archives.org/country/CN" # 所有语言的主页面, 入口页面
# entrance_url = "http://www.language-archives.org/language/jya" # 其中一个语言

# 1、从主页面遍历语言
response = requests.get(entrance_url)        #Get方式获取网页数据
soup = BeautifulSoup(response.text, 'lxml')
language_url_list = []
language_url_dict = {} # 将符合筛选条件(内部href个数大于100)的语言进行打印存储
for li in soup.find_all('li'):
    language_url_list.append(base_url + li.find('a').get('href')) # 将获取的语言子页面链接与base_url拼接


# 2、根据每个语言遍历内部的href, href数量低于100的舍弃, 并将语言和其对应的urf、href列表存储到字典language_url_dict中
for language_url in tqdm(language_url_list):
    language_response = requests.get(language_url) 
    language_soup = BeautifulSoup(language_response.text, 'lxml')
    language_data = language_soup.select('body > ol:nth-child(9) > li > a') # 选择所有的数据, 选择器教程: http://c.biancheng.net/view/2011.html 
    
    language_href_list = []
    for item in language_data:
        language_href_list.append(base_url + item.get('href'))

    if len(language_href_list) > 100:
        language_name = language_url.split('/')[-1]
        language_url_dict[language_name] = {
            'language_url': language_url,
            'herf_url': language_href_list
        }

# 3、将筛选结果存储到文件中
with codecs.open("language_url_dict.json", "w", 'utf-8') as wf:
    json.dump(language_url_dict, wf, indent=4)



# 4、根据文件中各个语言的href列表遍历可能存在的xml文件，并将其下载存储到result文件夹中, 按照语言进行文件夹划分

# spider_2.py中
