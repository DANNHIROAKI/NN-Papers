import pandas as pd
import requests
from bs4 import BeautifulSoup


article_url = []
 
# 폼 데이터 양식을 이용하여 url 주소를 추출
# lines = 폼데이터
lines = '''pageInfo:bksMain
login_chk:null
LOGIN_SN:null
LOGIN_NAME:null
indexName:news
keyword:
byLine:
searchScope:1
searchFtr:1
startDate:1990-01-01
endDate:2020-05-29
sortMethod:date
contentLength:639
providerCode:01100801
categoryCode:
incidentCode:7, 24
dateCode:
highlighting:true
sessionUSID:
sessionUUID:test
listMode:
categoryTab:
newsId:
filterProviderCode:
filterCategoryCode:
filterIncidentCode:
filterDateCode:
filterAnalysisCode:
startNo:1
resultNumber:100
topmenuoff:
resultState:
keywordJson:
keywordFilterJson:
realKeyword:
totalCount:
interval:
quotationKeyword1:
quotationKeyword2:
quotationKeyword3:
searchFromUseYN:N
mainTodayPersonYn:
period:3month'''.splitlines()

# 각 문서의 번호 저장
data = {}
for line in lines:
    key, value = line.split(':', 1)
    if value == 'null':
        value = None
    data[key] = value

# 빅카인즈 url에서 데이터 크롤링
result_url = 'https://www.bigkinds.or.kr/news/newsResult.do'
response = requests.post(result_url, data=data)
html = response.text
soup = BeautifulSoup(html, 'html.parser')

for tag in soup.select('.resultList li h3'):
    doc_id = tag['id'].replace('news_', '')
    doc_url = 'https://www.bigkinds.or.kr/news/detailView.do?docId={}&returnCnt=1'.format(doc_id)
    article_url.append(doc_url)

# 각 문서의 url 저장
pd.DataFrame(article_url).to_csv('뇌물수수url.csv')