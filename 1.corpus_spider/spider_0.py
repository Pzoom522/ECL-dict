
import requests
from bs4 import BeautifulSoup
import codecs
from tqdm import tqdm
import json

base_url = "http://www.language-archives.org"

entrance_url = "http://www.language-archives.org/country/CN" # all langs
# entrance_url = "http://www.language-archives.org/language/jya" # single lang


response = requests.get(entrance_url)       
soup = BeautifulSoup(response.text, 'lxml')
language_url_list = []
language_url_dict = {} 
for li in soup.find_all('li'):
    language_url_list.append(base_url + li.find('a').get('href')) 


for language_url in tqdm(language_url_list):
    language_response = requests.get(language_url) 
    language_soup = BeautifulSoup(language_response.text, 'lxml')
    language_data = language_soup.select('body > ol:nth-child(9) > li > a')
    
    language_href_list = []
    for item in language_data:
        language_href_list.append(base_url + item.get('href'))

    if len(language_href_list) > 100:
        language_name = language_url.split('/')[-1]
        language_url_dict[language_name] = {
            'language_url': language_url,
            'herf_url': language_href_list
        }


with codecs.open("language_url_dict.json", "w", 'utf-8') as wf:
    json.dump(language_url_dict, wf, indent=4)

