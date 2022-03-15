import numpy as np
import numpy.linalg as nla

''' orthgonal Test '''
#====================================================================
M = np.array([[1,2],[4,3]])
val,vec = nla.eig(M)

# print(M)
# print(val)
# print(vec) # 왜 백터값이 [1,2]^T, [1, -1]^T로 안나오는가...
vect = vec.transpose()
val_M = np.eye(2) * val
make_M = np.dot(vec,val_M)
make_M1 = np.dot(make_M, vect) # ??
make_M2 = np.dot(make_M, nla.inv(vec))
print(vect, nla.inv(vec))
print('\n', make_M1,'\n', make_M2)

svd_val = nla.svd(M)
a,b,c = nla.svd(M)
print(svd_val)
print(val,'\n', vec, '\n')
print(a)
print(b)
print(c)
# print('=====================================')
#====================================================================

''' Transpose, dot, SVD, Eigen value & vector'''
# =====================================================================
A = np.array([[3,6],
             [2,3],
             [0,0],
             [0,0]])
# [[1,2],[4,3]]
# [3,6],
#              [2,3],
#              [0,0],
#              [0,0]

At = A.transpose() # 전치 행렬
AAt = np.dot(A, At) # 4x4
AtA = np.dot(At, A) # 2x2
AAt_eig_val, AAt_eig_vec = nla.eig(AAt)
AtA_eig_val, AtA_eig_vec = nla.eig(AtA)
AAt_svd = nla.svd(AAt)
AtA_svd = nla.svd(AtA)
A_svd = nla.svd(A)

sigma = np.eye(4,2) *  A_svd[1]
# print(A_svd[0], sigma)
test_A = np.dot(A_svd[0], sigma)
test_A = np.dot(test_A, A_svd[2])

# test_A = A_svd[0] * (np.eye(4,2) *  A_svd[1]) *  A_svd[2]

print('Test_A: ', test_A)
print('행렬 A:\n ', A)
print('AA^t:\n {0} \n A^tA:\n {1}'.format(AAt, AtA))
print('AA^t_val:\n {0} \n AA^t_vec:\n {1}'.format(AAt_eig_val, AtA_eig_vec))
print('A^tA_val:\n {0} \n A^tA_vec:\n {1}'.format(AtA_eig_val, AtA_eig_vec))

print('AAt\n U', AAt_svd[0], '\n\n S', AAt_svd[1], '\n\n V', AAt_svd[2])
print('AtA\n U', AtA_svd[0], '\n\n S', AtA_svd[1], '\n\n V', AtA_svd[2])
print('A\n U',A_svd[0], '\n\n S',A_svd[1],'\n\n V',A_svd[2])
print('A_inv & -1\n',nla.inv(A_svd[0]),A_svd[0].transpose() )

# Data 전처리
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords') # 불용어 관련
stop_word = stopwords.words('English')

news = fetch_20newsgroups(shuffle=True, random_state= 1, remove=['headers', 'footers', 'quotes'])
news_data = news.data
# print(news_data)
# print(news.target_names)
news_DF = pd.DataFrame(news_data)
news_DF.columns = ['doc']
# print(news_DF)
# news_DF['doc'] = news_DF['doc'].str.replace('[^a-zA-Z]+', ' ')
# print('정규 표현식, 3글자 이하 불용어 단어 제거 전: ', news_DF['doc'][1])
news_DF['doc'] = news_DF['doc'].apply(lambda x: re.sub('[^a-zA-Z]+',' ',x)) # 영어만 남기기
# print('정규 표현식 후, 3글자 이하 불용어 단어 제거 전: ', news_DF['doc'][1])
news_DF['doc'] = news_DF['doc'].apply(lambda x: ' '.join(z for z in x.split() if len(z) > 3 or z not in stop_word)) # 스탑워드 3글자 미만 단어 날리기
# print('정규 표현식, 3글자 이하 불용어 단어 제거 후: ', news_DF['doc'][1])
# print(news_DF)

# TF-IDF 작업 후 Truncated SVD 작업
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

tf_idf = TfidfVectorizer(stop_words='english', max_features= 1000, max_df=0.5, smooth_idf=True)
# term = tf_idf.get_feature_names()
trun_svd = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)

news_tfidf = tf_idf.fit_transform(news_DF['doc'])
news_tfidf_dense = news_tfidf.toarray()
news_trunsvd = trun_svd.fit(news_tfidf)

# print(news_tfidf,news_tfidf[0], news_tfidf[1])
# print(news_trunsvd, np.shape(trun_svd.components_),trun_svd.components_) # 20 x 1000

term = tf_idf.get_feature_names()
# term_dd = news_tfidf.get_feature_names()
# print(term)
def get_topic(comonents,f_name, n = 10):
    for i, topic_val in enumerate(comonents): # 1 x 1000 씩
        print('Topic {0}: {1}'.format(i + 1, [
            (f_name[i1], topic_val[i1].round(5)) for i1 in topic_val.argsort()[:-n - 1:-1]]))
        # for j in topic_val.argsort()[:-n - 1:-1]:
        #     print(j, len(topic_val))

# print(news_tfidf_dense)
# news_tfidf_dense = sorted(news_tfidf_dense, key = lambda D:D[1])
# print(news_tfidf_dense)

get_topic(trun_svd.components_, term)
# get_tfidf(news_tfidf_dense, term)


# =====================================================================