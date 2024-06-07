import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Input
import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/thushv89/attention_keras/master/layers/attention.py",
                           filename="attention.py")
from attention import AttentionLayer

def test(model, output, hidden_size, content_len) :
    encoder_inputs = output[0]
    encoder_outputs = output[1]
    state_h = output[2]
    state_c = output[3]
    decoder_inputs = output[4]
    dec_emb_layer = output[5]
    decoder_lstm = output[6]
    decoder_softmax_layer = output[7]

    # 인코더 설계
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

    # 이전 시점의 상태들을 저장하는 텐서
    decoder_state_input_h = Input(shape=(hidden_size,))
    decoder_state_input_c = Input(shape=(hidden_size,))

    dec_emb2 = dec_emb_layer(decoder_inputs)
    # 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용. 이는 뒤의 함수 decode_sequence()에 구현
    # 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태인 state_h와 state_c를 버리지 않음.
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2,
                                                        initial_state=[decoder_state_input_h, decoder_state_input_c])

    # 어텐션 함수
    decoder_hidden_state_input = Input(shape=(content_len, hidden_size))
    attn_layer = AttentionLayer(name='attention_layer')
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # 디코더의 출력층
    decoder_outputs2 = decoder_softmax_layer(decoder_inf_concat)

    # 최종 디코더 모델
    decoder_model = Model(
        [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs2] + [state_h2, state_c2])

    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, title_word_to_ix, title_ix_to_word, content_len):
    # 입력으로부터 인코더의 상태를 얻음
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    # <SOS>에 해당하는 원-핫 벡터 생성
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = title_word_to_ix['<S>']

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:  # stop_condition이 True가 될 때까지 루프 반복
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        print(sampled_token_index)
        sampled_char = title_ix_to_word[sampled_token_index]

        if (sampled_token_index != '<E>'):
            decoded_sentence += ' ' + sampled_char

        # <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_char == '<E>' or len(decoded_sentence) > content_len):
            stop_condition = True

        # 길이가 1인 타겟 시퀀스를 업데이트 합니다.
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # 상태를 업데이트 합니다.
        e_h, e_c = h, c

    return decoded_sentence


# 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2text(input_seq, content_ix_to_word):
    temp = ''
    for i in input_seq:
        if (i != 0):
            temp = temp + content_ix_to_word[i] + ' '
    temp = temp.replace('<UNK>', '')
    temp = temp.replace('<PAD>', '')
    temp = temp.replace('  ', ' ')
    return temp


# 요약문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2summary(input_seq, title_word_to_ix, title_ix_to_word):
    temp = ''
    for i in input_seq:
        if ((i != 0 and i != title_word_to_ix['<S>']) and (i != title_word_to_ix['<E>'])):
            temp = temp + title_ix_to_word[i] + ' '

    temp = temp.replace('<UNK>', '')
    temp = temp.replace('<PAD>', '')
    temp = temp.replace('  ', ' ')
    return temp


