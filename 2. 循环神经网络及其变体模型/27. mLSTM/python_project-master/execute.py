import pandas as pd
from generator_test_model import model_create
from word_to_index_and_padding import *
from test import *
from category_classification import *

data = pd.read_excel('test/29일_날씨.xlsx')
data = data[data.notnull()]
data = data[data['TITLE'].notnull()]

index = 0
category = category_classification(data.CONTENT[index])
content = category_data['CONTENT'][index].to_list()
title = category_data['TITLE'][index].to_list()

with open('word_dict/content_ix_to_word_' + category + '.pkl', 'rb') as f:
    content_ix_to_word = pickle.load(f)
with open('word_dict/title_ix_to_word_' + category + '.pkl', 'rb') as f:
    title_ix_to_word = pickle.load(f)
with open('word_dict/title_word_to_ix_' + category + '.pkl', 'rb') as f:
    title_word_to_ix = pickle.load(f)

content_len_dict = {'날씨':[976, 12], '사건_사고':[1134, 13], '뇌물수수':[1417, 17]}
content_len, title_len = content_len_dict[category]

pad_num = 0
index_title, index_content = word_to_index(category, content, title)
input_idx = seq_padding(index_content, content_len, pad_num, 0)
target_idx = seq_padding(index_title, title_len, pad_num, 1)

temp = pd.DataFrame(input_idx).to_numpy()
temp = np.array([s[:-1] for s in temp])
input_data = temp
temp = pd.DataFrame(target_idx).to_numpy()
temp = np.array([s[:-1] for s in temp])
target_data = temp


embedding_dim = 128

if (category == '날씨') or (category == '뇌물수수') : 
    hidden_size = 256
else : 
    hidden_size = 128

model, output = model_create(category, embedding_dim, hidden_size)
model.load_weights(category+'_'+str(embedding_dim)+'_'+str(hidden_size)+'/checkpoints')

encoder_model, decoder_model = test(model, output, hidden_size, content_len)

print("원문 : ", seq2text(input_data, content_ix_to_word))
print("실제 요약문 :", seq2summary(target_data, title_word_to_ix, title_ix_to_word))
print("예측 요약문 :", decode_sequence(input_data.reshape(1, content_len),
                                  encoder_model, decoder_model,
                                  title_word_to_ix, title_ix_to_word, content_len))
