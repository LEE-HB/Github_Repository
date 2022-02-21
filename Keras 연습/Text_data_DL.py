import numpy as np
import string
'''3가지 방법으로 단어 vectorizer 하기'''
# 원-핫-백터
def one_hot_encoding(strs): # word
    token_dic = {}

    for str in strs:
        for word in str.split():
            if word not in token_dic:
                token_dic[word] = len(token_dic) + 1
    # print(token_dic, len(token_dic))
    max_length = len(token_dic)
    result = np.zeros(shape = (len(strs), max_length, max(token_dic.values())+1))
    # print(result, result.shape)

    for i, str in enumerate(strs):
        for j, word in enumerate(str.split()):
            index = token_dic.get(word) # value 값을 가져옴
            print(index)
            result[i,j,index] = 1.
    # print(result, result.shape)
    return result
# 문장 백터
def strs_encoding(strs):
    characters = string.printable # 정규표현식에 있는 모든 문자를 의미함
    token_index = dict(zip(characters, range(1, len(characters)+1)))
    # print(characters, token_index)
    max_length = 50
    result = np.zeros(shape = (len(strs), max_length, max(token_index.values())+1))
    for i, str in enumerate(strs):
        for j, word in enumerate(str.split()):
            index = token_index.get(word) # value 값을 가져옴
            print(index)
            result[i,j,index] = 1.
    # print(result, result.shape)
    return result
# 해싱 기법 사용_ 메모리가 절약 되나 해시 충돌 발생 시 머신 러닝이 단어의 차이를 인식하지 못 함
# 이것은 '해싱 공간 차원' > '고유 토큰 수' 훨씬 크게 하면 충돌 가능성 감소
def one_hot_hasing(samples):
    dimensionality = 10000
    max_length = 10
    result = np.zeros(shape = (len(samples), max_length, dimensionality))
    for i, str in enumerate(samples):
        for j, word in enumerate(str.split()):
            index = abs(hash(word)) % dimensionality
            print(index, hash(word))
            result[i, j, index] = 1.
    print(result, result.shape)
    return result


samples = ['The cat sat on the mat.', 'The dot ate my homework.']
one_hot_encoding(samples)
strs_encoding(samples)
one_hot_hasing(samples)

# keras Tokenizer 사용
from keras.preprocessing.text import Tokenizer

tokenize = Tokenizer(num_words=1000) # 가장 빈도 높은 1,000개의 단어만 선택함
tokenize.fit_on_texts(samples) # 단어 인덱스 구축
sequences = tokenize.texts_to_sequences(samples)

one_hot_result = tokenize.texts_to_matrix(samples, mode='binary')
word_index = tokenize.word_index
print(tokenize.word_index) # vectoriz한 dic를 출력
print(tokenize.word_counts) # 각 단어의 빈도 수를 출력
print(tokenize.document_count) # 샘플 수 출력

print('%s개의 고유한 토큰을 찾았습니다.' %len(word_index))

'''단어 Emvedding층을 사용하여 학습하기!!'''
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

embedding_layer = Embedding(1000, 64) #`1000개의 단어 인덱스, 임베딩 차원(64 차원 = 64개 특성?)

max_feature = 1000 # 특성 = 단어의 개수
max_len = 20 # 사용할 텍스트의 길이

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=max_feature) #이러면 가장 비번한 1000개의 단어만 나오는 건가?
print(len(x_train[0]), len(x_test), x_train.shape, x_train[0])
x_train = pad_sequences(x_train, maxlen= max_len)
x_test = pad_sequences(x_test, maxlen= max_len)
print(len(x_train[0]), len(x_test), x_train.shape,  x_train[0])

model = Sequential()
model.add(Embedding(10000, 8, input_length=(max_len))) # 만 개 샘플, max_len 길이, 8의 임베딩 차원

model.add(Flatten()) # 3D Tensor -> 2D Tensor로 변경 (samples, max_len * 8)

model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss = 'binary_crossentropy',
              metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs= 10,
                    batch_size = 32,
                    validation_split=0.2)

'''원본 텍스트 부터 단어 임베딩까지 진행하기'''
import os
base_dir = '/Users/bombom/PycharmProjects/Python_Project/Keras 연습/IMDB 원본 텍스트 데이터'
train_dir = '/Users/bombom/PycharmProjects/Python_Project/Keras 연습/IMDB 원본 텍스트 데이터/train'

labels = []
text = []

for type in ['neg','pos']:
    now_dir = os.path.join(train_dir, type)
    for fname in os.listdir(now_dir):
        if fname[-4:] == '.txt':
            f = open(os.path.join(now_dir, fname), encoding='utf8')
            text.append(f.read())
            f.close()
        if type == 'neg':
            labels.append(0)
        else: labels.append(1)

print(len(labels), len(text))
# print(text)

