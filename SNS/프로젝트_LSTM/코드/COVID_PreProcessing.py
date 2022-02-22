# 코드 : 성현준, 이한봄

#-*- coding: utf-8 -*-
# nltk.download('punkt') # 첫 실행 후 필요 없음
import pandas as pd
import numpy as np
from konlpy.tag import Okt
import re
from soylemma import Lemmatizer
from pykospacing import Spacing
from hanspell import spell_checker
np.set_printoptions(threshold=np.inf, precision = 7) # (threshold = 무한으로 array 출력, rpecision = TF-IDF 때 float 자릿수 조정) 
#_____________________________________________________________________________________________
# 1. Crawling 된 csv 파일 생성 ↓↓
f = pd.read_csv(r"C:\Users\82102\Desktop\YTN.csv", encoding='utf-8-sig' )
# 2. np array로 변경 (TF-IDF 때 이용하기 위함) ↓↓
np_array = np.array(f)

fname3 = r"C:\Users\82102\Desktop\표제어 사전.csv"
lemma_dic_before = list(pd.read_csv(fname3,usecols=['before'], encoding='utf-8').iloc[:,0])
lemma_dic_after = list(pd.read_csv(fname3,usecols=['after'], encoding='utf-8').iloc[:,0])

#_____________________________________________________________________________________________
# 불용어 설정 (하기 주소 = 해당 사이트에 등록된 불용어들, stop_words2 = 우리가 등록한 불용어들) ↓↓
stopwords_ex = pd.read_csv("https://raw.githubusercontent.com/yoonkt200/FastCampusDataset/master/korean_stopwords.txt").values.tolist()
stop_words2 = ['강예원','적', '자식', '질좀', '이', '을', '들', '은', '에', '가', '도', '는', '의', '를', '으로', '보다', '것', '정부', '중', '와', '하고', '서', '로', '에게', '다', '더', '처럼', '제발', '그건', '임창정', '강예빈', '니', '하', '뭐', '무슨', '무식하다', '어쩌라고', '건가', '거지', '연예인', '력', '라도', '라면', '등', '과학자', '기레기', '공산', '공산주의', '온', '식', '수없이', '에만', '곳', '건가', '뭔', '부터', '뿐', '이렇게', '맙시', '대', '대한', '씨', '년', '놈', '죠', '하', '해', '돋', '때', '까지', '몇', '년뒤', '나르다', '위해', '데이터', '쌓다', '북한', '몰래', '빼', '돌리다', '계속', '들이다', '보다', '치다', '동기간', '꼴', '겁내', '어쨋', '든', '중국산', '가나', '을지', '살이', '구', '요', '마도', '볼수', '술처', '이고', '저', '달', '몇', '하다', '한', '부부', '남편', '이나'] 
# stop_words 에 우리가 등록할 불용어 추가 ↓↓
stop_words = [x[0] for x in stopwords_ex]
stop_words.extend(stop_words2)

#_____________________________________________________________________________________________
# 3. Okt 전처리
# 3-1. 객체 생성 ↓↓
okt = Okt() 
lemma = Lemmatizer()
spacing = Spacing()
# 3-2. 문장 전처리 함수 (댓글 한 개씩 들어옴!!!) ↓↓
def text_cleaning(a): 
    clean_text = []
    text = re.sub('[^가-힣]+',' ', str(a)) # (한글만 남기고 나머지 다 삭제)
    # 3-2-1. PyKoSpacing - (띄어쓰기 오류 수정)
    test_f = spacing(re.sub('[\s]+', '', text))
    # 3-2-2. 정규화(normalize) - (어지럽힌 문장을 깔끔하게 만듬 -> "안녕하세욬ㅋ" => "안녕하세요ㅋ",  "샤릉해" => "사랑해") ↓↓
    normal = okt.normalize(test_f)
    norminal = spell_checker.check(normal)
    # 3-2-3. 형태소분석(morphs) - (형태소를 분석 -> "오래간만이네요" => '오래간만', '이네요' ↓↓
    morphs = okt.morphs(norminal.checked)
    # 3-2-4. 불용어 제거 - (앞서 등록했던 불용어들을 제거) 
    [clean_text.append(j) for j in morphs if j not in stop_words]
    # 3-2-5. 형태소분석(Pos Tagging) - (품사 추출을 위해 한번 더 형태소 분석)
    pos_text = okt.pos(' '.join(clean_text))
    # 3-2-6. 전처리된 댓글(clean_text) 반환 
    return pos_text

# 3-4. 단어 전처리 함수 (댓글 한 개씩 들어옴!!!) 
def word_cleaning(a):
    clean_word = []
    # 3-4-1. 형용사, 동사, 명사만 사용할 것 ↓↓
    word = ['Adjective', 'Verb', 'Noun']
    # 3-4-2. 단어,품사 = x,y - (y(품사)가 word 안의 단어와 동일하면 clean_word 에 x(단어) 추가) ↓↓
    [clean_word.append(x) for x,y in a if y in word]
    # 3-4-3. 표제어 사전 적용
    lemma_removed_st = []
    for st in clean_word:
        if st in lemma_dic_before:
            ref_index = lemma_dic_before.index(st)
            lemma_removed_st.append(lemma_dic_after[ref_index])
        else:
            lemma_removed_st.append(st)
    return lemma_removed_st

datalist = [] 
df = pd.read_csv(r"C:\Users\82102\Desktop\YTN_pre.csv", encoding='cp949' )
# 3.5-2. 진행
for i in range(0, len(np_array)):
    text_clean = text_cleaning(np_array[i][2]) #(정규화, 형태소분석, 불용어 제거)
    word_clean = word_cleaning(text_clean) #(형용사, 동사, 명사 구분)
    datalist.append(' '.join(word_clean))

df["내용"] = datalist
df.to_csv(r"C:\Users\82102\Desktop\YTN_pre.csv", mode='w', index=False, encoding = 'utf-8-sig')    
#_____________________________________________________________________________________________
# 정제 끝








#_____________________________________________________________________________________________
# from tensorflow.keras.preprocessing.text import Tokenizer
# # 4. TF-IDF 진행
# tokenizer = Tokenizer() # 빈도수가 많은 순으로 단어 보존함
# tokenizer.fit_on_texts(sublist) # 문자 데이터를 리스트로 셋업 함
# X_train = tokenizer.texts_to_matrix(datalist, mode= 'tfidf').round(3) # Text를 숫자로 변환


from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf_vec = TfidfVectorizer()
tfidf = tfidf_vec.fit_transform(datalist).toarray()
# 4-4. Columns 기준 생성
dff = pd.DataFrame(tfidf, columns=tfidf_vec.get_feature_names())
# 4-5. TF-IDF csv 파일에 작성
dff.to_csv("TF-IDF.csv", mode='w', index=False, encoding = 'utf-8-sig')    

#_________________________________________________________________
# 4. TF-IDF 진행
# from sklearn.feature_extraction.text import TfidfVectorizer 
# # 4-1. 객체 생성
# tfidf_vec = TfidfVectorizer()
# sublist = [] 
# # 4-2. 
# for j in datalist:
#     sublist.append(' '.join(j))
# # 4-3. tfidf 를 array로 변경
# tfidf = tfidf_vec.fit_transform(sublist).toarray()
# # 4-4. Columns 기준 생성
# df = pd.DataFrame(tfidf, columns=tfidf_vec.get_feature_names())
# # 4-5. TF-IDF csv 파일에 작성
# df.to_csv("TF-IDF.csv", mode='w', index=False, encoding = 'utf-8-sig')    
# print(df)

# 3-3.    
# def word_lemmatizer(a):
#     lemmatizer_word = [lemmatizer.lemmatize(x) for x in a]
#     for y in range(0,len(lemmatizer_word)):
#         if  lemmatizer_word[y] == []:
#             lemmatizer_word[y] = a[y]
#         else :
#             lemmatizer_word[y] = lemmatizer_word[y][0][0]
#     return lemmatizer_word  


