from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Concatenate, Input

def model_create(category, embedding_dim, hidden_size):

    word_len_dict = {'날씨': [696, 4100], '사건_사고': [2999, 11623], '뇌물수수':[2668, 16511]}
    tar_vocab, src_vocab = word_len_dict[category]

    content_len_dict = {'날씨': [976, 12], '사건_사고': [1134, 13], '뇌물수수': [1417, 17]}
    content_len, title_len = content_len_dict[category]

    ###### 모델 설계
    # 인코더
    encoder_inputs = Input(shape=(content_len,))

    # 인코더의 임베딩 층
    enc_emb = Embedding(src_vocab, embedding_dim)(encoder_inputs)

    # 인코더의 LSTM 1
    encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True,
                         dropout=0.4, recurrent_dropout=0.4)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    # 인코더의 LSTM 2
    encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True,
                         dropout=0.4, recurrent_dropout=0.4)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)
    
    # 인코더의 LSTM 3
    encoder_lstm3 = LSTM(hidden_size, return_state=True, return_sequences=True,
                         dropout=0.4, recurrent_dropout=0.4)
    
    if category == '사건_사고' :
        encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)
    else :
        encoder_outpus3, state_h3, state_c3 = encoder_lstm3(endcoder_output2)
        
        # 인코더의 LSTM 4
        encoder_lstm4 = LSTM(hidden_size, return_state=True, return_sequences=True,
                         dropout=0.4, recurrent_dropout=0.4)
        encoder_outputs, state_h, state_c = encoder_lstm4(encoder_output3)
        

    # 디코더
    decoder_inputs = Input(shape=(None,))

    # 디코더의 임베딩 층
    dec_emb_layer = Embedding(src_vocab, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)

    # 디코더의 LSTM
    decoder_lstm = LSTM(hidden_size, return_sequences=True,
                        return_state=True, dropout=0.4, recurrent_dropout=0.2)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h,
                                                                 state_c])

    # 디코더의 출력층
    decoder_softmax_layer = Dense(tar_vocab, activation='softmax')
    decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs)

    # 모델 정의
    model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)

    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/thushv89/attention_keras/master/layers/attention.py",
                               filename="attention.py")
    from attention import AttentionLayer

    # 어텐션 층(어텐션 함수)
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

    # 어텐션의 결과와 디코더의 hidden state들을 연결
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')(
        [decoder_outputs, attn_out])

    # 디코더의 출력층
    decoder_softmax_layer = Dense(tar_vocab, activation='softmax')
    decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

    # 모델 정의
    model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
    
    if category == '날씨' : 
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    elif category == '사건_사고' :
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    else :
        from tensorflow.keras.optimizers import *

        adam = optimizers.Adam(lr=0.01, beta_1=0.8, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)
        model.compile(optimizer=adam, loss='sparse_categorical_crossentropy')


    outputs = [encoder_inputs, encoder_outputs,
               state_h, state_c, decoder_inputs,
               dec_emb_layer, decoder_lstm, decoder_softmax_layer]

    return model, outputs

