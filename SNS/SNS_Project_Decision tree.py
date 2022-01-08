#-*- coding: utf-8 -*-
import nltk
# nltk.download('punkt')
import pandas as pd
import numpy as np
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from tensorflow.keras.preprocessing.text import Tokenizer # 토큰화 및 워드 백터, TF IDF만들기
import matplotlib.pyplot as plt
from matplotlib import rc # 그래프에 한글을 가능하게 함

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

okt = Okt()
train_file = r'/Users/bombom/PycharmProjects/Python_Project/COVID_Preprocessing(rev-1.0).csv'
train_data = pd.read_csv(train_file, usecols=['극성', '내용'], encoding='utf-8')
test_data = ['스터 샷 맞은 정도 항체 올라가서 확률 줄어들겠지 개월 지나간 항체 떨어지면 다시 확 진자 늘겠지 그땐 차 맞은 하는',
             '백신 차로 끝 아닌 문 제이지 백신 맞은 마다 부작용 나타나면 복불복 하는 마음 은로 백신 맞기가 두렵다는 입니다',
             '백신 효과 없는 기우제 마음 맞은 두는 게 좋은 게 언론 입장',
             '백신 접종 면역 하제 확실하다고 생각 되는',
             '팩트 백신 부스터 샷 접종 동일 감염 상황 시 미접 종자 접종 걸림 초기 백신 접종 시 감염 확률 적다고 하는 이제 와서 말 바꾸고 감염 시 백신 접종 중증 갈 확률 말 주 장일 사실 아닌 게 팩트',
             '맞는 안 맞는 자유 강요 해서 안 됨',
             '백신 확실한 효과']
# test_data = [re.sub(r'[^가-힣 ]', ' ', st) for st in test_data]
# list 안에 문자열로 구성되어 있음 len(x_train) OK len(x_train[0]) NG / test_data 형식으로 변환됨
x_train_ = train_data['내용'].values
y_train = train_data['극성'].values
# for polarity in train_data['극성'].values:
#     sub_lst = [0, 0, 0]
#     sub_lst[polarity] = float(1) # softmax를 돌리기 위해 float형식으로 변경해야함??
#     y_train.append(sub_lst)
y_train = np.array(y_train) # Deeprunnig 모델을 학습 시킬 때 Numpy array 타입만 가능함

''' 토크나이저 하기'''
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
print(voca_size)# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 문장을 TF-IDF로 변환
# Chapter3 +++++++++++++++++++++++++++++++++++++++
token = Tokenizer(num_words=voca_size, oov_token='OOV')
token.fit_on_texts(x_train_) # 훈련 텍스트를 기준으로 훈련을 진행 함
x_train = token.texts_to_matrix(x_train_, mode='tfidf').round(3)
x_test = token.texts_to_matrix(test_data, mode='count').round(3)


''' 머신 러닝'''
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# print(x_test, np.shape(x_test), type(x_test[0]))
# print(type(x), type(y), np.shape(x), np.shape(y), len(x), len(y), type(x[0]), type(y[0]))
# x,y,의 타입을 Numpy.ndarray로 설정해야 되고 내부의 Data도 Numpy형식이어야 됨 / dytpe는 numpy의 데이터 타입이라는 의미임
commnets_forest = RandomForestClassifier(n_estimators=5, random_state=2)
# commnets_forest.fit(x_train, y_train)
# y_test = commnets_forest.predict(x_test)

# for comment, result in zip(test_f, y_test):
#     print(comment, result)
gbrt11 = GradientBoostingClassifier(random_state=0)
gbrt1 = GradientBoostingClassifier(random_state=0, max_depth=3)
gbrt01 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt11.fit(x_train, y_train)
gbrt1.fit(x_train, y_train)
gbrt01.fit(x_train, y_train)
y_test1 = gbrt11.predict(x_test)
y_test2 = gbrt11.predict(x_test)
y_test3 = gbrt11.predict(x_test)
i = 0
print()
for comment, r1, r2, r3 in zip(test_data, y_test1, y_test2, y_test3):
    r1 = '긍정 리뷰입니다.' if r1 == 1 else '부정 리뷰입니다.' if r1 == 0 else '중립 리뷰입니다.'
    r2 = '긍정 리뷰입니다.' if r2 == 1 else '부정 리뷰입니다.' if r2 == 0 else '중립 리뷰입니다.'
    r3 = '긍정 리뷰입니다.' if r3 == 1 else '부정 리뷰입니다.' if r3 == 0 else '중립 리뷰입니다.'
    print(r2, '\n', comment, '\n')
    i += 1
    if i == 10: break

