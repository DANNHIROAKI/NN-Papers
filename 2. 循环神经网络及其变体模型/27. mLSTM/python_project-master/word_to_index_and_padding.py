import pickle

def change_word_to_index(seq, dic, oov):
    sequence=[]
    tmp=[]
    for line in seq:
        for word in line.split():
            try:
                tmp.append(dic[word])
            except KeyError:
                tmp.append(oov)
        sequence.append(tmp)
        tmp=[]
    return sequence

def seq_padding(seq, num, pad, tmp=0, start_num=1, end_num=2):
    text = []
    tmp_list = []
    cnt = 0
    if tmp == 0:  # content일 경우
        for line in seq:
            for i in line:
                tmp_list.append(i)
                cnt += 1
            while cnt < num:
                tmp_list.append(pad)
                cnt += 1
            text.append(tmp_list)
            tmp_list = []
            cnt = 0

    if tmp == 1:  # title 경우
        for line in seq:
            tmp_list.append(start_num)
            for i in line:
                tmp_list.append(i)
                cnt += 1
            while cnt < num:
                tmp_list.append(pad)
                cnt += 1
            tmp_list.append(end_num)
            text.append(tmp_list)
            tmp_list = []
            cnt = 0

    return text


def word_to_index(category, content, title) :
    with open('word_dict/content_word_to_ix_'+category+'.pkl', 'rb') as f :
        content_word_to_ix = pickle.load(f)
    with open('word_dict/title_word_to_ix_'+category+'.pkl', 'rb') as f :
        title_word_to_ix = pickle.load(f)

    oov_num = 1

    index_title = change_word_to_index(title, title_word_to_ix, oov_num)
    index_content = change_word_to_index(content, content_word_to_ix, oov_num)

    return index_title, index_content

