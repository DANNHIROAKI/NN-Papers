import pandas as pd
from normalize import normalize

# 본문에 있는 영어, 숫자, 특수문자 제거 후 저장
data = pd.read_excel('뇌물수수.xlsx')
data['TITLE'] = data['TITLE'].apply(lambda x: x.replace(x, normalize(x)))
data['CONTENT'] = data['CONTENT'].apply(lambda x: x.replace(x, normalize(x)))
data.to_excel('normalize_뇌물수수.xlsx')