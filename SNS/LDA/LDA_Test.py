# News 데이터 사용

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords') # 불용어 관련
stop_word = stopwords.words('English')

#%% 1. Data 전처리 및 훈련
'''Data 전처리 진행하기'''
#===================================================================
data = fetch_20newsgroups(shuffle=True, random_state= 1, remove=['headers', 'footers', 'quotes']).data

data_df = pd.DataFrame(data)
data_df.columns = ['documents']

data_df['documents'] = data_df['documents'].apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x)) # 영어 제외 글자 날리기
data_df['documents'] = data_df['documents'].apply(lambda x: ' '.join(z for z in x.split()
                                                                     if len(z) > 3 or z not in stop_word) ) # 3글자 미만 & 스탑워드 날리기
data_df['documents'] = data_df['documents'].apply(lambda x: x.lower()) # 소문자로 만들기

# print(data_df[:5])
#===================================================================

''' LDA 훈련하기 '''
import gensim # LDA 하기
from gensim import corpora # 단어 목록 백터 만들기
from sklearn.feature_extraction.text import TfidfVectorizer
#===================================================================
tokens = data_df['documents'].apply(lambda x: x.split())
# tfidf = TfidfVectorizer(stop_words='english', max_features= 1000, max_df=0.5, smooth_idf=True)

term_dic = corpora.Dictionary(tokens) # 단어 목록
# print(term_dic) # TF-IDF랑 다른 것인가....? 토큰화의 차이??
dic_termidxNcount = [term_dic.doc2bow(t) for t in tokens] # list 형식의 pair(idx, n)
# print(dic_termidxNcount[:5]) # (idx,n)으로 출력 되는 idx는 단어의 인덱스, n은 단어 출현 횟수
Topic_nums = 20
# data, 토픽수, 단어 목록, 알고리즘 수행 횟수: passes,
train_model = gensim.models.ldamodel.LdaModel(
    dic_termidxNcount, num_topics = Topic_nums, id2word = term_dic, passes = 15)

topics = train_model.print_topics(num_topics = 4)

#%% 2. LDA 시각화
#===================================================================
'''LDA 시각화 하기 '''
import pyLDAvis
import pyLDAvis.gensim_models as gsimv

# pyLDAvis.enable_notbook()
vis = gsimv.prepare(train_model, dic_termidxNcount, term_dic) #
pyLDAvis.display(vis)
pyldavis_html_path = r'C:\Users\sattl\PycharmProjects\pythonProject_Anconda\LDA시각화_topic20개_15회 반복.html'
pyLDAvis.save_html(vis, pyldavis_html_path)

#===================================================================
''' 숫자 확률 추출'''
for i,  x in enumerate(train_model[dic_termidxNcount]):
    if i == 100:
        break
    print(i+1, '번째 문서 topic비율: ', x)



