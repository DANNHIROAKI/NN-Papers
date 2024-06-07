import pandas as pd
import numpy as np
from normalize import *
import operator
import pickle
from konlpy.tag import Komoran

def loading_data_by_category(data_path, eng=True, num=True, punc=False):
    # data example : 'title', 'content', 'category', 'nouns'
    # data format : xlsx
    corpus = pd.read_excel(data_path)
    corpus = np.array(corpus)
    title = []
    contents = []
    nouns = set()
    for doc in corpus:
        if type(doc[3]) is not str or type(doc[4]) is not str:
            continue
        if len(doc[3]) > 0 and len(doc[4]) > 0:
            tmptitle = normalize(doc[3], english=eng, number=num, punctuation=punc)
            title.append(tmptitle)

            tmpcontents = normalize(doc[4], english=eng, number=num, punctuation=punc)
            contents.append(tmpcontents)

            for noun in doc[5].split() :
                nouns.add(noun)
    return title, contents, nouns


def make_dict_all_cut(contents, nouns, min_length, max_length, title = False):
    dictionary = {}
    komoran = Komoran()
    if not title :
        for noun in sorted(nouns, reverse=False):
            if len(noun) > min_length:
                pos_result = komoran.pos(noun)
                tag_list = [tag for _, tag in pos_result]
                if ('NA' not in tag_list) and (
                        'NR' not in tag_list) and (
                        tag_list[0] != 'NNB') and (
                        'IC' not in tag_list):
                    # 'NA':분석불능범주, 'NR':수사, 'NNB':의존명사으로 시작, 'IC':감탄사 인 단어 제외
                    if (tag_list[0] == 'VV' or tag_list[0] == 'VA') and (
                            'NNG' not in tag_list and 'NNP' not in tag_list):
                        # 동사, 형용사 stemming
                        verb = pos_result[0][0] + '다'
                        if verb in dictionary:
                            if noun not in dictionary[verb]:
                                dictionary[verb].append(noun)
                        else:
                            dictionary[verb] = [verb]
                            dictionary[verb].append(noun)

                    else:
                        # 동사나 형용사에 해당되지 않는 단어
                        if noun[:max_length] in dictionary:
                            if noun not in dictionary[noun[:max_length]]:
                                dictionary[noun[:max_length]].append(noun)
                        elif noun[:max_length - 1] in dictionary:
                            if noun not in dictionary[noun[:max_length - 1]]:
                                dictionary[noun[:max_length - 1]].append(noun)
                        else:
                            dictionary[noun[:max_length]] = [noun]

    for doc in contents:
        for word in doc.split():
            tmp = []
            for char in word:
                if ord(char) < 12593 or ord(char) > 12643:
                    # 자음 or 모음만 있는 글자 제외
                    tmp.append(char)
            tmp = ''.join(tmp)
            if len(tmp) > min_length:
                pos_result = komoran.pos(tmp)
                tag_list = [tag for _, tag in pos_result]
                if ('NA' not in tag_list) and (
                        'NR' not in tag_list) and (
                        tag_list[0] != 'NNB') and (
                        'IC' not in tag_list):
                    if (tag_list[0] == 'VV' or tag_list[0] == 'VA') and (
                            'NNG' not in tag_list and 'NNP' not in tag_list):
                        verb = pos_result[0][0] + '다'
                        if verb in dictionary:
                            if tmp not in dictionary[verb]:
                                dictionary[verb].append(tmp)
                        else:
                            dictionary[verb] = [verb]
                            dictionary[verb].append(tmp)

                    else:
                        if tmp[:max_length] in dictionary:
                            if tmp not in dictionary[tmp[:max_length]]:
                                dictionary[tmp[:max_length]].append(tmp)
                        elif tmp[:max_length - 1] in dictionary:
                            if tmp not in dictionary[tmp[:max_length - 1]]:
                                dictionary[tmp[:max_length - 1]].append(tmp)
                        else:
                            dictionary[tmp[:max_length]] = [tmp]

    dictionary = sorted(dictionary.items(), key=operator.itemgetter(0))[1:]
    words = []
    for i in range(len(dictionary)):
        word = list(dictionary[i][1])
        words.append(word)

    words.insert(0, ['<PAD>']) # 패딩
    words.insert(1, ['<UNK>']) # unkown 단어
    if title :
        words.insert(2, ['<S>']) # start
        words.insert(3, ['<E>']) # end


    # word_to_index, index_to_word 생성
    index_to_word = {i: ch[0] for i, ch in enumerate(words)}
    word_to_index = {}
    for idx, word_list in enumerate(words):
        for word in word_list:
            word_to_index[word] = idx
    print('컨텐츠 갯수 : %s, 단어 갯수 : %s' % (len(contents), len(index_to_word)))

    return word_to_index, index_to_word

# pickle 저장함수
def save_obj(obj, name, path='') :
    with open(path + name + '.pkl', 'wb') as f :
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# 해당 분야 데이터 불러오기
title, contents, nouns = loading_data_by_category('normalize_뇌물수수_add_nouns.xlsx', 
                                                  eng=False, num=False, punc=False)


# word_to_ix, ix_to_word 만들기
contents_word_to_ix, contents_ix_to_word = make_dict_all_cut(contents, nouns, 
                                                             min_length=1, max_length=3)
title_word_to_ix, title_ix_to_word = make_dict_all_cut(title, None, 
                                                       min_length=1, max_length=3, title=True)

save_obj(contents_word_to_ix, 'content_word_to_ix_뇌물수수', 'word_dict/')
save_obj(contents_ix_to_word, 'content_ix_to_word_뇌물수수', 'word_dict/')

save_obj(title_word_to_ix, 'title_word_to_ix_뇌물수수', 'word_dict/')
save_obj(title_ix_to_word, 'title_ix_to_word_뇌물수수', 'word_dict/')