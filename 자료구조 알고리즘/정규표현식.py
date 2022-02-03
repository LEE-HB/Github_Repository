from typing import List, Tuple, Dict

ex1 = "100-200*300-500+20" # 60420
ex2 = "50*6-3*2" # 300

# format 없이 간단하게 출력이 가능함
print(f'이건 f 포멧팅 {ex1}, zzz')

# 에너 그램 만들기
ex = ['eat', 'repaid', 'paired', 'tea', 'bat']
dic = {}
for x in ex:
    temp = ''.join(sorted(x)) # sort 시에 리스트로 변환되는 것 잊지 않기!!
    if temp in dic:
        dic[temp].append(x)
    else: dic[temp] = [x] # dic value를 리스트로 하면 위의 줄 처럼 리스트 형식의 함수 사용가능
print(dic)

import re
'''정규 표현식 규칙'''
def is_phone_number(number: List[str]) -> bool:
    # test = re.findall(phone_number_pattern, number)
    test = re.search(phone_number_pattern, number)
    return True if test != None else False
# 폰 번호 서치 하기
phone_number_pattern = re.compile('^\(?\d{3}\)?\ ?\-?\ ?\d{3,4}\ ?\-?\ ?\d{3,4}$')
ex11 = ['010-8030-1667', '010-123-4567', '010 1234 5678','1234512451346']
for x in ex11:
    dd = is_phone_number(x)
    # print(dd)
# IPv4, IPv6 서치 하기
def IPv_checker(IPv: str) -> bool:
    IPv4_pattern = re.compile('^([1-9]{0,2}\d{1}\.){3}[1-9]{0,2}\d{1}$')
    # IPv4_pattern = re.compile('(\d{1,3}\.){3}\d{1,3}')
    # [a-f0-9]
    IPv6_pattern = re.compile(r'^([a-f0-9]{1,4}[:]){6}[a-f0-9]{1,4}$', re.IGNORECASE)
    check4 = re.search(IPv4_pattern, IPv)
    check6 = re.search(IPv6_pattern, IPv)
    print(IPv)
    print(check4, check6)

    return True if check4 != None or check6 != None else False

ex12_1 = '172.2.10.1' # pass
ex12_2 = '172.2.10.01' # fail
ex12_3 = '172.2.10.001' # fail
ex12_4 = '172.258.160.32' # pass

ex13_1 = '2020:0bc3:0000:0000:853e:0777:1234' # pass
ex13_2 = '2020:0BC3:0:0:853e:0777:1234' # pass
ex13_3 = '2020:0BC3:::853e:0777:1234' # fail

print(IPv_checker(ex12_1))
print(IPv_checker(ex12_2))
print(IPv_checker(ex12_3))
print(IPv_checker(ex12_4))

print(IPv_checker(ex13_1))
print(IPv_checker(ex13_2))
print(IPv_checker(ex13_3))