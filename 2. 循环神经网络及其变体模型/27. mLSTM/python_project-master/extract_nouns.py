import pandas as pd
from konlpy.tag import Okt
from tqdm import tqdm

# 데이터 불러오기
news = pd.read_excel('normalize_뇌물수수.xlsx', index_col=0)
# 내용이 없는 데이터 행 삭제
news = news[news.CONTENT.notnull()]
news = news.reset_index(drop=True)

# Okt 이용
okt = Okt()
news_nouns = []

# 전체 데이터에 대하여 명사만 추출
for text in tqdm(news.CONTENT) :
    temp = ' '.join(okt.nouns(text))
    news_nouns.append(temp)

# 명사 추출 결과를 데이터 'nouns'열로 저장
news['nouns'] = news_nouns
news.to_excel('normalize_뇌물수수_add_nouns.xlsx')