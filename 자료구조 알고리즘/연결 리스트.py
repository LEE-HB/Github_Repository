from typing import Any, List, Tuple, Dict

# 연결 리스트는 트리나 그래프의 주요 내용임
class Node:
    def __init__(self, data: Any):
        self.previous = None
        self.data = data
        self.next = None

class Linked_Node:
    def __init__(self, node_data: Any):
        self.head = node_data

    def tracer(self):
        node = self.head
        i = 0
        while node:
            print(node.data, end=' ')
            node = node.next
            i += 1
            if node == self.head:
                break
        print(f'\n현재 연결 리스트 길이는 {i}')

    def push_back(self, data: Any):
        new_node = Node(data)
        node = self.head

        while node:
            if node.next == None:
                node.next = new_node
                return
            node = node.next

    def push_front(self, data: Any):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def node_del(self, data: Any):
        curr_node = self.head
        pre_node = None
        if curr_node.data == data:  # 헤드랑 일치
            self.head = curr_node.next
            return

        while curr_node:
            if curr_node.data == data:
                pre_node.next = curr_node.next

            pre_node = curr_node
            curr_node = curr_node.next
            # print(f'이전{pre_node.data}, 이후{curr_node.data}')

    def node_reverse(self):
        node = self.head

        stack = []
        while node:
            stack.append(node.data)
            node = node.next
        print(stack)
        new_node = Linked_Node(Node(stack.pop()))
        self.head = new_node

        while stack:
            new_node.push_back(stack.pop())

        return self.head

    def check_cycle(self):
        fast_point = self.head
        slow_point = self.head
        while fast_point != None and fast_point.next != None:
            fast_point = fast_point.next.next
            slow_point = slow_point.next
            if fast_point == slow_point:
                return print('순환 하는 그래프 이다!!')
        return print('순환하지 않음')
# ================================================================================
node1 = Node(3)
node2 = Node(7)
node3 = Node(8)
node4 = Node(2)

node1.next = node2
node2.next = node3
node2.previous = node1

linked_nodes = Linked_Node(node1)
linked_nodes.push_back(3)

linked_nodes.tracer()
linked_nodes.check_cycle()
# ================================================================================
# ================================================================================
linked_node = Linked_Node(Node(5))
linked_node.push_back(7)
linked_node.push_back(9)
linked_node.push_back(6)
linked_node.tracer()

linked_node.push_front(1)
linked_node.tracer()

linked_node.node_del(6)
linked_node.tracer()

linked_node = linked_node.node_reverse()
linked_node.tracer()
# ================================================================================
def add_twoint(chain_lst1: Linked_Node, chain_lst2: Linked_Node) -> Linked_Node:
    st1 = ''
    st2 = ''
    node1 = chain_lst1.head
    node2 = chain_lst2.head
    while node1:
        st1 += str(node1.data)
        st2 += str(node2.data)
        node1 = node1.next
        node2 = node2.next
    sum_ = str(int(st1) + int(st2))
    # print(sum_)
    chain_lst = Linked_Node(Node(int(sum_[0])))
    for i, x in enumerate(sum_):
        if i == 0: continue
        chain_lst.push_back(int(x))
    return chain_lst
add_twoint(linked_node,linked_nodes).tracer()

import re
from itertools import permutations
import math

#==================================================================
def Check(num):
    if num == 0 or num == 1: return False

    iter = int(math.sqrt(num)) + 1
    print(iter)
    for ch in range(2, iter):
        if num % ch == 0:
            return False
    return True


def solution(numbers):
    lst_num = re.findall('[0-9]', numbers)
    # print('시작단계: ', lst_num)

    lst = []
    for i in range(1, len(lst_num) + 1):
        arr = list(permutations(lst_num, i))
        print('\nlst_num 길이: ',len(arr))# 2
        print('arr 값: ', arr)
        for j in range(len(arr)):
            num = int(''.join(map(str, arr[j])))
            # print("num: ", num)
            if Check(num) == True:
                lst.append(num)
        # print(set(lst))
    answer = len(set(lst))
    return answer
numbers1 = '17'
numbers2 = '011'
solution(numbers1)