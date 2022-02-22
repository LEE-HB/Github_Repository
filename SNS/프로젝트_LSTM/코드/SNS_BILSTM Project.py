# 코드 : 김광호, 박소라, 이한봄
import pandas as pd
import numpy as np
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer # 토큰화 및 워드 백터, TF IDF만들기
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dropout
import re


# 데이터 가져 오기 및 모듈 만들기
# Chater1 ++++++++++++++++++++++++++++++++++++++++++++++++++++
okt = Okt()
train_file = r'/Users/bombom/PycharmProjects/Python_Project/COVID_Preprocessing(rev-1.0).csv'
train_data = pd.read_csv(train_file, usecols=['극성', '내용'], encoding='utf-8')
# test_data = ['스터 샷 맞은 정도 항체 올라가서 확률 줄어들겠지 개월 지나간 항체 떨어지면 다시 확 진자 늘겠지 그땐 차 맞은 하는',
#              '백신 차로 끝 아닌 문 제이지 백신 맞은 마다 부작용 나타나면 복불복 하는 마음 은로 백신 맞기가 두렵다는 입니다',
#              '백신 효과 없는 기우제 마음 맞은 두는 게 좋은 게 언론 입장',
#              '백신 접종 면역 하제 확실하다고 생각 되는',
#              '팩트 백신 부스터 샷 접종 동일 감염 상황 시 미접 종자 접종 걸림 초기 백신 접종 시 감염 확률 적다고 하는 이제 와서 말 바꾸고 감염 시 백신 접종 중증 갈 확률 말 주 장일 사실 아닌 게 팩트',
#              '맞는 안 맞는 자유 강요 해서 안 됨',
#              '백신 확실한 효과',
#              '백신 제발 맞은 피해 주지 말고']
test_file = r'/Users/bombom/PycharmProjects/Python_Project/SNS/TOTAL_pre.csv'
test_data = pd.read_csv(test_file, usecols=['내용'], encoding='utf-8')
test_df = pd.read_csv(test_file, usecols=['내용'], encoding='utf-8')
test_data = test_data['내용'].values


# list 안에 문자열로 구성되어 있음 len(x_train) OK len(x_train[0]) NG / test_data 형식으로 변환됨
x_train_ = train_data['내용'].values
y_train = []
for polarity in train_data['극성'].values:
    sub_lst = [0, 0, 0]
    sub_lst[polarity] = float(1) # softmax를 돌리기 위해 float형식으로 변경해야함??
    y_train.append(sub_lst)
y_train = np.array(y_train) # Deeprunnig 모델을 학습 시킬 때 Numpy array 타입만 가능함
print(x_train_[:3], y_train[:3])
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 토큰화를 선 진행하여 단어의 개수를 표현할 백터의 차원을 구하기_단어 개수 = 입력 백터 차원
# Chater2 ++++++++++++++++++++++++++++++++++++++++++++++++++++
token_test = Tokenizer()
token_test.fit_on_texts(x_train_)

limit_iter = 2
word_cnt = len(token_test.word_index)
under_limit = 0
for key, val in token_test.word_counts.items():
    if val <= limit_iter:
        under_limit += 1

print('단어 개수: ', word_cnt)
print('기준 개수, 기준 미만 단어 수: ', limit_iter-1, under_limit)
print('기준 단어 미만 개수 비율: ', under_limit / word_cnt * 100)
voca_size = word_cnt - under_limit # 단어의 출현 빈도 횟수가 3이하면 취급하지 않겠다는 의미
print(voca_size)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 문장을 TF-IDF로 변환
# Chapter3 +++++++++++++++++++++++++++++++++++++++
token = Tokenizer(num_words=voca_size, oov_token='OOV')
token.fit_on_texts(x_train_) # 훈련 텍스트를 기준으로 훈련을 진행 함
x_train = token.texts_to_matrix(x_train_, mode='tfidf').round(3)
x_test = token.texts_to_matrix(test_data, mode='count').round(3)


x_train_count = token.texts_to_matrix(x_train_, mode='count')
x_test_count = token.texts_to_matrix(test_data, mode='count')
def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (count / len(nested_list)) * 100))

max_len = 80
# below_threshold_len(max_len, x_train_)
x_train_count = pad_sequences(x_train_count, maxlen=max_len)
x_test_count = pad_sequences(x_test_count, maxlen=max_len)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential

em_vector_dim = 100
hidden_dim = 16
batch_size = 100

BI_LSTM = Sequential()
BI_LSTM.add(Embedding(voca_size, em_vector_dim))
BI_LSTM.add(LSTM(hidden_dim))
BI_LSTM.add(Dropout(0.5))
BI_LSTM.add(Dense(32, activation='relu'))
BI_LSTM.add(Dropout(0.5))
BI_LSTM.add(Dense(3, activation='softmax'))

BI_LSTM.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
BI_LSTM.fit(x_train, y_train, epochs=5, batch_size=batch_size, validation_split=0.05)
BI_LSTM.summary()

predict_val1 = BI_LSTM.predict(x_test)

result_lst = []

positive_word_lst = []
negative_word_lst = []
middle_word_lst = []
for st, val in zip(test_df['내용'], predict_val1):
    index = val.argmax()
    result = '부정' if index == 0 else '긍정' if index == 1 else '중립'
    # print()
    if result == '부정': negative_word_lst.append(str(st))
    elif result == '긍정': positive_word_lst.append(str(st))
    else: middle_word_lst.append(str(st))
    result_lst.append('{0}의 확률로 {1}리뷰 입니다.'. format((np.max(val)*100).round(2), result))
print(negative_word_lst, positive_word_lst, middle_word_lst)


#============================================================

# 데이터 엑셀로 추출
# test_df["결과"] = result_lst
#
# print(test_df)
# test_df.to_csv(r"/Users/bombom/PycharmProjects/Python_Project/SNS/댓글감성분석 결과.csv", mode='w', index=False, encoding = 'utf-8-sig')

# wordcloud 시각화 하기


#============================================================
'''
딥 러닝을 진행 할 때 주의 사항!!
- 이진 분류 일때는 0,1로만 표기하고 활성함수를 Sigmoid or tanh을 사용해되 되지만 
  다중분류를 진행할 시에는 [1,0,0](3가지), [0,0,1,0](4가지)로 표현하고 활성함수를 softmax를 사용해야 한다.
  또한 Model 컴파일 시에 loss의 경우 이진은 binary_crossentropy를 다중(3개 이상)은 categorical_crossentropy를 지정해야 함
- 입력 데이터를 numpy으 데이터로 받아야함
- tensorflow의 토큰화를 진행 시에는 list 안에 바로 문자열이 있어야됨_list안의 list가 있으면 안됨
- 위의 Sequential의 모델의 경우 매우 선형으로 연결된 간단한 경우에 사용 되는 것이고 히든레이어가 병렬로 연결되거나 모듈로서 사용될 수도 있다.
- 
'''
