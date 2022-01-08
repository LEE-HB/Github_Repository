import matplotlib.pyplot as plt
from matplotlib import rc # 그래프에 한글을 가능하게 함
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import os
from os import path
import numpy as np
import pandas as pd

rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

file_word = r'/Users/bombom/PycharmProjects/Python_Project/SNS/댓글감성분석 결과___.csv'
datatable = pd.read_csv(file_word,encoding='utf-8')

positive_word_lst = []
negative_word_lst = []
middle_word_lst = []

for st, result in zip(datatable['내용'],datatable['결과']):
    if '긍정' in result:
        positive_word_lst.append(st)
    elif '부정' in result:
        negative_word_lst.append(st)
    else: middle_word_lst.append(st)


d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
# icon = Image.open('./injection.png')
# mask = np.array(Image.open(path.join(d, "./ddd.png")))
covid_mask = np.array(Image.open('./D1.png'))
# color_mask = ImageColorGenerator(covid_mask)
# print(mask)
wc_positive = WordCloud(font_path= "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
               background_color="white",
               max_words=100,
               contour_width=5,
               contour_color= 'blue',
               # max_font_size=30,
               mask=covid_mask)

wc_negative = WordCloud(font_path= "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
               background_color="white",
               max_words=100,
               contour_width=5,
               contour_color= 'red',
               # max_font_size=30,
               mask=covid_mask)

wc_middle = WordCloud(font_path= "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
               background_color="white",
               max_words=100,
               contour_width=5,
               contour_color= 'black',
               # max_font_size=30,
               mask=covid_mask)

wc_pos = wc_positive.generate(' '.join(positive_word_lst))
wc_neg = wc_negative.generate(' '.join(negative_word_lst))
wc_mid = wc_middle.generate(' '.join(middle_word_lst))


# plt.figure(figsize=(11.7))
plt.axis('off')
plt.title('긍정 댓글 단어 분포')
plt.imshow(wc_pos, interpolation='bilinear')
plt.show()

# plt.figure(figsize=(11.7))
plt.axis('off')
plt.title('부정 댓글 단어 분포')
plt.imshow(wc_neg, interpolation='bilinear')
plt.show()

# plt.figure(figsize=(11.7))
plt.axis('off')
plt.title('중립 댓글 단어 분포')
plt.imshow(wc_mid, interpolation='bilinear')
plt.show()
