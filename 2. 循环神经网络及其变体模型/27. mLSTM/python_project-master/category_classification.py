import numpy as np
from konlpy.tag import Okt
import pickle
import os

def category_classification(content) :
    # data : .xlsx
    with open('vocab_list.pkl', 'wb') as f :
        vocab_list = pickle.load(f)
        
    file_list = os.listdir('./TFIDF/')
    category = [file_list[i][:-5] for i in range(len(file_list))]
    
    okt = Okt()

    cnt_num = np.zeros(len(file_list))
    noun_list = okt.nouns(content)
    for noun in noun_list :
        for j in range(3) :
            if noun in vocab_list[j] :
                cnt_num[j] += 1
    return category[np.argmax(cnt_num)]
