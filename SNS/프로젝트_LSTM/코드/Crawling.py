# 코드 : 성현준, 박소라

#-*- coding: utf-8 -*-
# selenium, By, Keys 라이브러리 import 
from selenium import webdriver 
from selenium.webdriver.common.by import By 
from selenium.webdriver.common.keys import Keys
# 타이머 라이브러리
import time 
# 정규표현식 수정 관련 라이브러리 (이거 사용안하고 .replace 사용해도 됨)
import re  
# csv 파일 라이브러리
import csv 
#____________________________________________________________________________
# 1. 크롤링할 Webdriver = 크롬
driver = webdriver.Chrome('C:\Python\chromedriver.exe') 
# 2. 뉴스 기사 링크주소
driver.get(r'https://everytime.kr/377391/all/%EB%B0%B1%EC%8B%A0') 
# 2-1. 크롤링한 csv 생성
f = open('C:\Python\COVID1.csv','w', newline='') 
wr = csv.writer(f) 
# 3. 첫 댓글 더보기 클릭 (사이트에 따라 진행할 필요없을 수도 있음 - 네이버 기사의 경우 해당)
element = driver.find_element_by_xpath('//*[@id="container"]/div[2]/div[2]/a')
driver.execute_script("arguments[0].click()", element)
time.sleep(1)
# 4. range 범위만큼 더보기 진행 (사이트에 따라 진행할 필요없을 수도 있음 - 네이버 기사의 경우 해당)
for i in range(0,20) :
    element2 = driver.find_element_by_xpath('//*[@id="cbox_module"]/div[2]/div[9]/a/span/span/span[1]') 
    driver.execute_script("arguments[0].click()", element2)
    time.sleep(1)
# 5. rnage 범위만큼 댓글 크롤링
textlist = [] 
for x in range(1,69): 
    t = driver.find_element(By.XPATH, '//*[@id="cbox_module_wai_u_cbox_content_wrap_tabpanel"]/ul/li[%d]/div[1]/div/div[2]/span[1]'%x) 
    # 5-1. xpath 자료들은 selenium 형식임, 정규표현식 적용하려면 .text로 텍스트화
    text = t.text 
    # 5-2. 댓글들 csv에 추가 (댓글 한개씩)
    wr.writerow([x, text]) 
    time.sleep(1)



