import pandas as pd
import re
import requests

# 해당 url 접속 후 문서 내용 크롤링
def insert_df(num):
    cnt_fx = num
    next_num = cnt_fx
    try:
        for i in range(cnt_fx,len(data)):
            url = df.iloc[:,-1][i]
            response = requests.get(url)
            test = response.text
            test = test.replace('false','"false"')
            dic = eval(test)
            tmp = eval(str(dic['detail']))
            data.loc[i] = [tmp['DATE'], tmp['CATEGORY_MAIN'], 
                           tmp['TMS_RAW_STREAM'], tmp['TITLE'],
                           tmp['CONTENT']]
            print(str(cnt_fx)+"번 완료")
            cnt_fx += 1
            next_num = cnt_fx+1
    except SyntaxError:
        print(next_num)
        insert_df(next_num)
        
# url주소 데이터 불러오기
df = pd.read_csv("29일_사건사고.csv")

# 불러온 url주소의 길이만큼 DataFrame 생성
data = pd.DataFrame(index=range(df.shape[0]), 
                 columns=['DATE','CATEGORY_MAIN', 'TMS_RAW_STREAM',
                          'TITLE','CONTENT'])

# 해당 url 접속 후 문서 내용 크롤링 수행
insert_df(0)

# 빈 데이터인 행 제거 후 저장
data = data[data['TITLE'].notnull()]
data = data[data['CONTENT'].notnull()]
data.CATEGORY_MAIN = '뇌물수수'
data.to_excel("뇌물수수.xlsx")