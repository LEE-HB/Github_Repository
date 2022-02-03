from typing import List, Tuple, Dict

class Node:
    def __init__(self, data: any):
        self.left = None
        self.right = None
        self.data = data
    def __repr__(self):
        return str(self.data)

class Tree:
    def __init__(self):
        self.root = None

    def insert_(self, node, data: any):
        if not node.left and node.data >= data:
            node.left = Node(data)
        elif node.right == None and node.data < data:
            node.right = Node(data)

        else:
            if node.data >= data:
                self.insert_(node.left, data)
            elif node.data < data:
                self.insert_(node.right, data)


    def inorder_(self):
        r = []

        self.inorder_tracer(self.root, r)
        return  r

    def inorder_tracer(self, node, r):
        if node == None:
            return
        # print(node.data) # pre-order
        self.inorder_tracer(node.left, r)
        # print(node.data)
        r.append(node.data) # in-order
        self.inorder_tracer(node.right, r)
        # print(node.data) # post-order

    def element_find(self,  val: any) -> bool:
        node = self.root
        return self.element_find_rec(node, val)

    def element_find_rec(self,node , val: any):
        if node == None:
            return False
        elif node.data == val:
            return True

        elif node.data >= val:
            return self.element_find_rec(node.left, val)
        elif node.data < val:
            return self.element_find_rec(node.right, val)



root_node = Node(11)
tree = Tree()

tree.root = root_node

tree.insert_(tree.root, 10)
tree.insert_(tree.root, 1)
tree.insert_(tree.root, 8)
tree.insert_(tree.root, 15)
tree.insert_(tree.root, 17)
tree.insert_(tree.root, 13)


print(tree.inorder_())
print(tree.element_find(2))

print('test')

