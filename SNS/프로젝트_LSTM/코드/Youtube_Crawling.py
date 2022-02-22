#코드 : 김광호, 한륜결

from selenium import webdriver
from bs4 import BeautifulSoup
import time
import re
from selenium.webdriver.common.keys import Keys
import pandas as pd

start = time.time()

delay = 3
path = "C:\Crawling\chromedriver.exe"
start_url = 'https://www.youtube.com/'
browser = webdriver.Chrome(path)
browser.implicitly_wait(delay)
browser.get(start_url)
browser.maximize_window()

browser.find_element_by_xpath('//*[@id="search-form"]').click() #검색창영역클릭
browser.find_element_by_xpath('//*[@id="search-form"]/div/div/div/div[2]/input').send_keys('KBS News')#검색창 영역에 원하는 youtuber입력
browser.find_element_by_xpath('//*[@id="search-form"]/div/div/div/div[2]/input').send_keys(Keys.RETURN)#엔터
browser.find_element_by_xpath('//*[@id="info-section"]/a').click() #채널 클릭
browser.find_element_by_xpath('//*[@id="tabsContent"]/ytd-expandable-tab-renderer[7]/yt-icon-button/button/yt-icon').click() #채널 내 검색 클릭
browser.find_element_by_xpath('//*[@id="input-1"]/input').send_keys('백신 미접종') #채널 내 검색에 원하는 것 입력
browser.find_element_by_xpath('//*[@id="input-1"]/input').send_keys(Keys.RETURN) #엔터
browser.find_element_by_xpath('//*[@id="contents"]/ytd-item-section-renderer[8]/div[3]/ytd-video-renderer/div/ytd-thumbnail').click() #영상 클릭
time.sleep(6)
browser.find_element_by_xpath('//*[@id="skip-button:5"]/span/button').click() #광고 넘기기
time.sleep(1)
browser.find_element_by_xpath('//*[@id="movie_player"]/div[1]/video').click() #영상 일시정지
time.sleep(1)

#페이지 끝까지 스크롤
browser.execute_script("window.scrollTo(0, 900)")
time.sleep(1.5)
last_height = browser.execute_script("return document.documentElement.scrollHeight")
time.sleep(1.5)
while True:
    browser.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    new_height = browser.execute_script("return document.documentElement.scrollHeight")
    time.sleep(1.5)
    if new_height == last_height:
        break
    last_height = new_height

browser.execute_script("window.scrollTo(0, 900)")
time.sleep(1)

#팝업창 닫기
try:
    browser.find_element_by_css_selector("#dismiss-button > a").click()
except:
    pass

#대댓글 모두 열기
rereple = browser.find_elements_by_css_selector("#more-replies > a")
for re1 in rereple:
    browser.execute_script("arguments[0].click()", re1)
    time.sleep(1)

#스크롤
body = browser.find_element_by_css_selector('body')
for i in range(50):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(1.5)

browser.execute_script("window.scrollTo(0, 900)")
time.sleep(1)

# 정보추출, 파싱
source = browser.page_source
soup = BeautifulSoup(source, 'lxml')

# 데이터 저장
ids = [] #id
com = [] # 댓글
N = 1

comments = soup.find_all("ytd-comment-thread-renderer", class_="style-scope ytd-item-section-renderer")
replies = soup.find_all("ytd-comment-renderer", class_="style-scope ytd-comment-replies-renderer")
print(len(comments))
print(len(replies))
re_ids = [] # 답글 id
re_coms = [] # 답글 댓글

# 댓글별 ID,내용,좋아요,대댓글 저장
for comment in comments:
    # 댓글 내용
    com_temp = str(comment.find("yt-formatted-string", id="content-text").text)
    com_temp = com_temp.replace('\n', '')
    com_temp = re.sub('[^가-힣?]', '', com_temp)
    com.append(com_temp)
    # 대댓글
    replies = comment.find_all("ytd-comment-renderer", class_="style-scope ytd-comment-replies-renderer")

    for replie in replies:
        # 대댓글 내용
        re_com_temp = str(replie.find("yt-formatted-string", id="content-text").text)
        re_com_temp = re_com_temp.replace('\n', '')
        re_com_temp = re.sub('[^가-힣?]', '', re_com_temp)
        re_com = (N, re_com_temp)
        re_coms.append(re_com)

    N = N + 1

re_df = pd.DataFrame({"REPLY" : com})
rere_id = pd.DataFrame(re_ids)
rere_com = pd.DataFrame(re_coms)
rere_df = pd.DataFrame({"REPLY" : rere_com[1]})
commets_df = pd.concat([re_df, rere_df], axis=0)
print(commets_df, len(commets_df))
commets_df.to_csv("KBS_5.csv", mode='w', index=False, encoding='utf-8-sig')

browser.close()