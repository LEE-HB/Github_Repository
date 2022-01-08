from konlpy.tag import Okt # 한국어 형태소 나누기 # 이거 동작 안됨... ㅠ
import tweepy
import re
import numpy as np
from textblob import TextBlob
from nltk import pos_tag # 품사 추출하기
from nltk.stem.wordnet import WordNetLemmatizer # 표제어 찾기
from nltk import ne_chunk # NER 개체명 찾기 위한 list -> nltk.tree type 변경
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer #Term Frequency, TF-IDF
import seaborn as sns # 시각화 하기 위한 툴
from matplotlib import font_manager, rc # 그래프에 한글을 가능하게 함
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import os
from os import path
import nltk
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')



# font_location ='C:/Windows/Fonts/malgun.ttf' # 폰트 경로 지정
# fn = font_manager.FontProperties(fname=font_location).get_name() # 폰트를 가져옴
# rc('font',family=fn) # 한글이 가능한 폰트 등록
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


st = '목숨을 내놓고 맞을 정도면 백신에 문제 있는 거 아닌가? 안 맞은 사람 뭐라 하지 말자 백신 맞고 잘못되면 본인 탓인데  미접종을 왜 뭐라 하는지? 본인을 위해서 맞아야 하는 백신 아니냐?선택은 본인이 자율적으로 하도록 하는데 맞다고 본다  자율 이라면서 강제하지 말고  백신 맞아도 돌파감염 높다면서 왜 자꾸 강제 접종하려는지?, 나는 아이들 혼자 키우고 있어서 못 맞는다  백신 맞았다가 무슨 일 생기면 애들은 누가 책임지나  내가 무슨 일 있으면 폭력  여자  도박  술 모두 다 해서 겨우 이혼하고 헤어진 애들 아빠한테 애들 친권이 간다  백신의 안전성은 생각도 안 하고 무조건 목표치 채워가며 백신패스 불이익을 주는 게 더 불공평한 것 아닌가  백신 맞아도 감염되는 상황인데    독감 주사 안 맞았다고 뭐라하나    백신이 안전하고 맞아도 괜찮다면 누군들 안 맞고 싶겠나  목숨 걸고 맞아야되는 상황이 웃프다, 백신 안 맞는 게 왜 논란 거리임? 개인의 선택인데  백신 맞는다고 해서 코로나 안 걸리는 것도 아냐  전파력이 낮아지는 것도 아냐  맞는다고 해서 적어도 몇 년 면역력이 생기는 걸로 기대할 수 있는 것도 아냐  단순히 중증으로 가는 것을  어느 정도  예방하는 것 외엔 기대할 수 있는 게 아닌데 안 맞는 게 남들에게 피해를 준다고 말할 수는 없는 거지 '
st = re.sub('[\s]+', ' ', st)
okt_ = Okt()
okt_token = okt_.morphs(st)
# print(okt_token)
# print(tweepy.__version__)

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
# icon = Image.open('./injection.png')
# mask = np.array(Image.open(path.join(d, "./ddd.png")))
covid_mask = np.array(Image.open('./D1.png'))
# color_mask = ImageColorGenerator(covid_mask)
# print(mask)
wc = WordCloud(font_path= "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
               background_color="white",
               max_words=100,
               contour_width=3,
               # max_font_size=30,
               mask=covid_mask)
wc_data = wc.generate(st)
# wc.recolor(color_func=color_mask)
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(wc_data, interpolation='bilinear')
plt.show()

# wc.to_file('wordcloud_test.png')


def Make_BoW(token_st):
    dic = {}
    for char in token_st:
        if char not in dic:
            dic[char] = len(dic)
    return dic
'''표제어_추출'''

st_lst =[
    'Data Science is an overlap between Arts and Science',
    'Generally, Arts graduates are right-brained and Science graduates are left-brained',
    'Excelling in both Arts and Science at a time becomes difficult',
    'Natural Language Processing is a part of Data Science'
]
st_token1 = TextBlob(st_lst[0])
st_token2 = TextBlob(st_lst[1])
st_token3 = TextBlob(st_lst[2])
st_token4 = TextBlob(st_lst[3])
st_token1_bow = Make_BoW(st_token1.words)
print(st_token1_bow)
stem_lemmat = WordNetLemmatizer()
st_lemmat1 = [stem_lemmat.lemmatize(x) for x in st_token1.words]
st_lemmat2 = [stem_lemmat.lemmatize(x) for x in st_token2.words]
st_lemmat3 = [stem_lemmat.lemmatize(x) for x in st_token3.words]
st_lemmat4 = [stem_lemmat.lemmatize(x) for x in st_token4.words]
print('첫 번째 문장 표제어: {0} '
      '\n두 번째 문장 표제어: {1} '
      '\n세 번째 문장 표제어: {2} '
      '\n네 번째 문장 표제어: {3}'
      .format(st_lemmat1, st_lemmat2, st_lemmat3, st_lemmat4))

''' 객채명 찾기_NER'''
# 개체명 추출하기_POS Tagging을 진행해야됨
st_pos1 = pos_tag(st_token1.words)
st_pos2 = pos_tag(st_token2.words)
st_pos3 = pos_tag(st_token3.words)
st_pos4 = pos_tag(st_token4.words)

st_entity_name1 = [x for x in ne_chunk(st_pos1,binary=True) if type(x) != tuple]
st_entity_name2 = [x for x in ne_chunk(st_pos2,binary=True) if type(x) != tuple]
st_entity_name3 = [x for x in ne_chunk(st_pos3,binary=True) if type(x) != tuple]
st_entity_name4 = [x for x in ne_chunk(st_pos4,binary=True) if type(x) != tuple]

print('\n첫 번째 문장 NER: {0} '
      '\n두 번째 문장 NER: {1} '
      '\n세 번째 문장 NER: {2} '
      '\n네 번째 문장 NER: {3}'
      .format(st_entity_name1, st_entity_name2,st_entity_name3,st_entity_name4))
# =================================================================================
'''Model 만들기'''
tf = CountVectorizer(binary=True)
tf_idf = TfidfVectorizer()

'''TF 백터, Array'''
tf_st_vector = tf.fit_transform(st_lst)
tf_st_arr = tf_st_vector.toarray()
voca = tf.get_feature_names()
# print('TF 값: ', tf_st_vector, tf_st_arr)

'''TF-IDF 백터, dense'''
tf_idf_st_vector = tf_idf.fit_transform(st_lst)
tf_idf_st_arr = tf_idf_st_vector.todense() # todense와 toarray 차이?

print('TF-IDF 값: ', tf_idf_st_vector, tf_idf_st_arr)
# =================================================================================
'''시각화 하기'''
fig, axes = plt.subplots(1,2, figsize=(9,6))
fig.suptitle('TF, TF-IDF HeatMap')
# print(fig, axes)

sns.heatmap(tf_st_arr, annot=True, cbar = False,
            xticklabels = voca,
            yticklabels = ['s1_ㄴ','s2','s3','s4'], ax = axes[0])
sns.heatmap(tf_idf_st_arr, annot=True, cbar = False,
            xticklabels = voca,
            yticklabels = ['s1_ㅇ','s2','s3','s4'], ax = axes[1])
axes[0].set_title('TF HeatMap_히트맵')
axes[1].set_title('TF-IDF HeatMap_히트맵')
# plt.show()