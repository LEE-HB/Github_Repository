from keras.datasets import mnist
from keras import models, layers
import matplotlib.pyplot as plt
from typing import List
import numpy as np
from tensorflow.keras.utils import to_categorical

(train_image, train_class), (test_image, test_class) = mnist.load_data()

print('<변경 전>')
print('이미지 크기: ',train_image.shape,
      '\n이미지 차원: ', train_image.ndim,
      '\n 이미지 데이터 타입: ', train_image.dtype)
print('이미지 라벨: ', train_class, len(train_class))

print('\n이미지 크기: ',test_image.shape,
      '\n이미지 차원: ', test_image.ndim,
      '\n 이미지 데이터 타입: ', test_image.dtype)
print('이미지 라벨: ', test_class, len(test_class))

print(type(test_image[0]))
digit = test_image[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

train_image = train_image.reshape(60000,28 * 28)
train_image = train_image.astype('float32')

test_image = test_image.reshape(10000,28 * 28)
test_image = test_image.astype('float32')

train_class = to_categorical(train_class) #soft Max를 위한 변환
test_class = to_categorical(test_class) #soft Max를 위한 변환

print('\n<변경 후>')
print('이미지 크기: ',train_image.shape,
      '\n이미지 차원: ', train_image.ndim,
      '\n 이미지 데이터 타입: ', train_image.dtype)
print('이미지 라벨: ', train_class, train_class.shape, len(train_class), train_class[0])

print('\n이미지 크기: ',test_image.shape,
      '\n이미지 차원: ', test_image.ndim,
      '\n 이미지 데이터 타입: ', test_image.dtype)
print('이미지 라벨: ', test_class, test_class.shape, len(test_class), test_class[0])



minist_ = models.Sequential() # 딥러닝 모델 구축
minist_.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
minist_.add(layers.Dense(10, activation='softmax'))
minist_.compile(optimizer='rmsprop',
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])
minist_.fit(train_image, train_class, epochs=5, batch_size=128)




'''
아래는 numpy 실습
'''
#
# lst = [[1,2],
#        [3,4],
#        [5,6]]
# xx = [np.array(x) for x in lst]
# x = np.array(12)
# x1 = np.array([1,2,3,4,5])
# x2 = np.array(xx)
#
#
# print(x,'차원, 축의 수 : ', x.ndim)
# print(x1,'차원, 축의 수 : ', x1.ndim)
# print(x2,'차원, 축의 수 : ', x2.ndim)
# print(lst)

#
# def naive_relu(arr: np):
#     assert len(arr.shape) == 2,'x는 2차원 배열을 받아야 합니다.'
#     mat = arr.copy()
#     # print(mat.shape)
#     for i in range(mat.shape[0]):
#         for j in range(mat.shape[1]):
#             mat[i,j] = max(mat[i,j], 0)
#     return mat
#
# def vector_dot(vec1:np, vec2:np) -> int:
#     assert len(vec1.shape) == 1
#     assert len(vec2.shape) == 1
#     assert vec1.shape[0] == vec2.shape[0]
#
#     z = 0
#     for i in range(vec1.shape[0]):
#         z += vec1[i] * vec2[i]
#     return z
#
# def matrix_vector_dot(arr: np,vec:np) -> np:
#     assert len(arr.shape) == 2
#     assert len(vec.shape) == 1
#     assert arr.shape[1] == vec.shape[0]
#
#     z = np.zeros(arr.shape[0])
#     for i in range(arr.shape[0]):
#         for j in range(arr.shape[1]):
#             z[i] += arr[i,j] * vec[j]
#     return z
#
# def matrix_dot(arr1: np, arr2:np) -> np:
#     assert len(arr1.shape) == 2
#     assert len(arr2.shape) == 2
#     assert arr1.shape[1] == arr2.shape[0]
#
#     z = np.zeros((arr1.shape[0], arr2.shape[1]))
#     for i in range(arr1.shape[0]):
#         for j in range(arr2.shape[1]):
#             row = arr1[i, :]
#             col = arr2[:, j]
#             z[i,j] = vector_dot(row, col)
#     return z
#
#
#
# arr1 = np.array([[1,2,3],
#                  [10,11,12]])
# arr2 = np.array([[-1,2,-3],
#                  [10,-11,12]])
# arr3 = np.array([[[-1,2,-3],
#                  [10,-11,12]],
#                  [[11,23,10],
#                  [12,20,11]]])
# arr4 = np.array([[-1,2,-3],
#                  [10,-11,12],
#                  [7,8,9]])
# vec = np.array([3,4,5])
#
# ex1_1 = np.array([[1,2],
#                 [4,3]])
# ex1_2 = np.array([[5,5],
#                 [2,4]])
#
# print(naive_relu(arr1))
# print(naive_relu(arr2))
# # print(arr1.ndim, arr2.ndim, arr3.ndim)
# print(vector_dot(vec, vec))
# print(matrix_vector_dot(arr4, vec))
# print(matrix_dot(ex1_1, ex1_2))
# print(np.dot(ex1_1,ex1_2))
