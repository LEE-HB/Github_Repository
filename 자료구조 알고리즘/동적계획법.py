from typing import List
import math

''' 피보나치 수열의 동적 계획법 문제!!'''
# def fibo(n: int, memo:List[int]) -> int:
#     print(memo, n)
#     print()
#     if memo[n] == -1:
#         print(memo, n)
#         memo[n] = fibo(n-1, memo) + fibo(n-2, memo)
#     return memo[n]
#
# n = 5
# lst = [-1] * (n + 1)
# lst[0], lst[1] = 0, 1
# print(lst)
# val = fibo(n, lst)
# print(val)

'''세포 배양 문제_동적 계획법'''
ex1 = [5,3,4] # 13
ex2 = [3,4,3] # 11
ex3 = [7,4,2] # 14


def dp(n: int, lst: List[int], time_memo: List[int]) -> int: # bottom-up
    # print(time_memo, time_memo[n], n)
    if time_memo[n] == -1:
        val1 = lst[1] * n
        if n % 2 == 0:
            val2 = dp(n // 2, lst, time_memo) + lst[2] # 절반 나눈 친구에서 2배
        else:
            val2 = dp(n // 2, lst, time_memo) + lst[2] + lst[1] # 절반 나눈 친구에서 2배에서 하나 빼기
            val3 = dp(n // 2 + 1, lst, time_memo) + lst[2] + lst[1]  # 절반 나눈 친구에서 2배에서 하나 빼기
            val2 = min(val2, val3)
        time_memo[n] = min(val1, val2)
    print(time_memo)
    return time_memo[n]

def iter_(n: int, lst: List[int], time_memo:List[int]) -> int:
    for i in range(1, lst[0]):
        # if time_memo[i] == -1:
        #     val1 = lst[1] * (i + 1)
        #     if i % 2 == 0:
        #         val2 = (lst[1] + lst[2])
        #     else:
        #         val2 = (lst[1] + lst[2]) + lst[1]
        #     print(val1, val2)
        #     time_memo[i] = min(val1, val2)
        val1 = time_memo[i-1] + lst[1]
        if i % 2 == 0:
            val2 = time_memo[i-1] + lst[2]
        else:
            val2 = time_memo[i-1] + lst[2] + lst[1]

        time_memo[i] = min(val1, val2)
    print(time_memo)
    return time_memo[lst[0]-1]


time_lst = [-1] * (ex3[0] + 1)
time_lst[0], time_lst[1]= 0, ex3[1]
print(dp(ex3[0], ex3, time_lst))

