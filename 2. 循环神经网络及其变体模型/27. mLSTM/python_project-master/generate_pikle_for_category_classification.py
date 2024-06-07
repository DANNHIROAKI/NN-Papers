import pandas as pd
import os
import pickle

path = './TFIDF/'
file_list = os.listdir(path)

vocab_list = []
for k in range(len(file_list)) :
    tmp = pd.read_excel(path+file_list[k], index_col=0)
    tmp_vocab = [s for i in range(tmp.shape[1]) for s in tmp[i].to_list()]
    vocab_list.append(list(set(tmp_vocab)))
    
words = list(set([s for i in range(len(file_list)) for s in vocab_list[i]]))
word_table = pd.DataFrame(words)
category = [file_list[i][:-5] for i in range(len(file_list))]

with open('vocab_list.pkl', 'wb') as f :
    pickle.dump(vocab_list, f, pickle.HIGHEST_PROTOCOL)