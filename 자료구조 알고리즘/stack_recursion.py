# stack과 deque는 동일한 기능을 함
from collections import deque
stack = []
deckque = deque()

stack.append(5)
stack.append(7)
stack.append(8)

deckque.append(8)
deckque.append(10)
deckque.append(11)
print(stack, deckque)
print('22', deckque[2])

print(stack.pop())
print(deckque.pop())

print(stack.pop())
print(deckque.pop())
print(stack, deckque)

'''
- list를 이용한 stack과 deque는 표면상 동작은 동일하나 stack은 연속된 공간에 할당되는 list의 형태이고
  deque는 이중 연결리스트를 기반으로 만들어졌다. 
- 공간의 할당의 차이가 있으며 인덱스 찾기에서 list가 deque에 비해 약간 빠르다. list는 바로 찾을 수 있지만 deque는
  head에서 차근차근 접근을 들어가야함_검색 방법은 동일하지만 내부 동작이 다르다는 의미임 
'''

def recursion_test(nums):
    if len(nums) == 1:
        return nums[0]
    return nums[0] + recursion_test(nums[1:])

lst = [1,2,3,4,5]
print(recursion_test(lst))

def isValid(st: str) -> bool:
    dic = {
        '(': ')',
        '[': ']',
        '{': '}'
    }

    stack = []
    for x in st:
        if x in dic.keys():
            stack.append(x)
        elif x in dic.values():
            # not stack이랑 len(stack) == 0이랑 같은 의미
            if not stack or x != dic[stack.pop()] :
                return False

    return True if not stack else False
ex1 = '(({}[]))'
ex2 = '('
ex3 = ')'
print(isValid(ex2))
# page 227문제

def Stairs_find(n: int, curr: int) -> int:
    up = [1,2]
    print(n, curr)
    if n <= curr:
        if n == curr: return 1
        else: return 0
    return Stairs_find(n, curr + up[0]) + Stairs_find(n, curr + up[1])

print('방법 수:',Stairs_find(3, 0))
