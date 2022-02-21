
''' 직접 모델 구성해서 진행 해 보기 '''
# import os, shutil
#
# base_dir = "/Users/bombom/PycharmProjects/Python_Project/Keras 연습/Cat_dog 분류/base_dir"
# train_dir = os.path.join(base_dir, 'train')
# test_dir = os.path.join(base_dir, 'test')
# validation_dir = os.path.join(base_dir, 'validation')
#
#
# # 폴더 만들기_훈련, 검증, 테스트
# os.mkdir(train_dir)
# os.mkdir(test_dir)
# os.mkdir(validation_dir)
#
# # 폴더 만들기_고양이, 개
# train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)
#
# train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)
#
# test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)
#
# test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)
#
# validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)
#
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)
#
# # 파일 복사 하기 고양이
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)] # 훈련 데이터
# for fname in fnames:
#     src = os.path.join(base_dir, fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)] # 검증 데이터
# for fname in fnames:
#     src = os.path.join(base_dir, fname)
#     dst = os.path.join(validation_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)] # 테스트 데이터
# for fname in fnames:
#     src = os.path.join(base_dir, fname)
#     dst = os.path.join(test_cats_dir, fname)
#     shutil.copyfile(src, dst)
# # 파일 복사 하기 개
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)] # 훈련 데이터
# for fname in fnames:
#     src = os.path.join(base_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)] # 검증 데이터
# for fname in fnames:
#     src = os.path.join(base_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)] # 테스트 데이터
# for fname in fnames:
#     src = os.path.join(base_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# print('훈련용 고양이 개수: ', len(os.listdir(train_cats_dir)))
#
# '''CNN 시작'''
# # 모델 만들기
# from keras import layers, models
#
# CNN = models.Sequential()
# CNN.add(layers.Conv2D(32,(3,3), activation= 'relu',
#                       input_shape=(150,150,3)))
# CNN.add(layers.MaxPooling2D((2,2)))
# CNN.add(layers.Conv2D(64, (3,3), activation='relu'))
# CNN.add(layers.MaxPooling2D((2,2)))
# CNN.add(layers.Conv2D(128, (3,3), activation='relu'))
# CNN.add(layers.MaxPooling2D((2,2)))
# CNN.add(layers.Conv2D(128, (3,3), activation='relu'))
# CNN.add(layers.MaxPooling2D((2,2)))
#
# CNN.add(layers.Flatten())
# CNN.add(layers.Dropout(0.5)) # validation acc72%로 과대 적합을 개선하기 위함 50%의 유닛 날림
# CNN.add(layers.Dense(512, activation='relu'))
# CNN.add(layers.Dense(1, activation='sigmoid'))
# print(CNN.summary())
#
# # 모델 컴파일
# from keras import optimizers
# CNN.compile(loss='binary_crossentropy',
#             optimizer='rmsprop',
#             # optimizer=optimizers.RMSprop(lr=1e-4),
#             metrics = ['acc'])
# '''데이터 전처리_사진 전처리 '''
# from keras.preprocessing.image import ImageDataGenerator
# # train_data = ImageDataGenerator(rescale=1./255) # 모든 이미지를 1/255로 스케일 조정함
# train_datagen = ImageDataGenerator( # 훈련 이미지 증식 하기!! 과대적합 해결 위함
#     # 몇 개가 증식 되는지 모르것네...
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
#
# test_data = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory( # 개, 고양이 각 1,000개 씩 총 2,000개 변환
#     train_dir, # 타겟 디렉토리
#     target_size=(150,150), # 모든 이미지를 150x150으로 변경, 이유는 딥러닝 모델의 입력 차원이 150x150으로 쌓았기 때문에
#     batch_size=32,
#     class_mode='binary' # 딥러닝 모델의 loss를 binary_crossentropy로 다루기 때문
# )
# validation_generator = test_data.flow_from_directory( # 개, 고양이 각 500개 씩 총 1,000개 변환
#     validation_dir, # 타겟 디렉토리
#     target_size=(150,150), # 모든 이미지를 150x150으로 변경, 이유는 딥러닝 모델의 입력 차원이 150x150으로 쌓았기 때문에
#     batch_size=32,
#     class_mode='binary' # 딥러닝 모델의 loss를 binary_crossentropy로 다루기 때문
# )
#
# # 제너레이터는 무한 루프기 때문에 break 필수
# # for data_batch, labels_batch in train_generator:
# #     # print(train_generator)
# #     print('배치 data size: ', data_batch.shape)
# #     print('배치 label size: ', labels_batch.shape)
# #     break
#
# history = CNN.fit_generator(
#     train_generator,
#     steps_per_epoch=100, # 배치가 20개 임으로 20 * 100 = 2,000로 훈련 이미지 수가 됨
#     epochs= 100,
#     validation_data=validation_generator,
#     validation_steps=50)
# CNN.save('cats_and_dogs_small_2(add Drop out, datagen).h5')
#
# '''데이터 증식'''
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
# # 데이터 증식 한 거 확인
# # from keras.preprocessing import image
# # fnames = sorted([os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]) #훈련의 모든 데이터 이미지를 fnames의 리스트로 받아옴
# # img_path = fnames[3]
# # img = image.load_img(img_path, target_size=(150,150)) #이미지를 읽고 이미지 크기 변환
# #
# # x = image.img_to_array(img) # (150,150,3)
# # x = x.reshape((1,) + x.shape) # (1, 150,150,3)
# #
# # i = 0
# # for batch in datagen.flow(x, batch_size=1):
# #     print(len(batch), batch.shape)
# #     plt.figure(i)
# #     imgpolt = plt.imshow(image.array_to_img(batch[0]))
# #     i += 1
# #     if i % 4 == 0:
# #         break
# #
# # plt.show()
#
# '''훈련 정확도 및 손실 그래프 그리기'''
# import matplotlib.pyplot as plt
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label = 'Training acc')
# plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
# plt.title('Training and validation accuracy')
# plt.show()
#
# plt.plot(epochs, loss, 'ro', label = 'Training loss')
# plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
# plt.title('Training and validation loss')
# plt.show()

'''Application 사용 해서 구별 해보기'''
from tensorflow.keras.applications.vgg16 import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))
conv_base.summary()

# 데이터 전처리?? 특성 추출 하기
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = "/Users/bombom/PycharmProjects/Python_Project/Keras 연습/Cat_dog 분류/base_dir"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
def extract_features(directory, sample_num):
    features = np.zeros(shape = (sample_num, 4,4,512))
    labels = np.zeros(shape = (sample_num))
    generator = datagen.flow_from_directory( # (20,150,150)
        directory,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    # print(generator)
    for input_batch, label in generator:
        # print(input_batch.shape, generator.image_shape)
        features_batch = conv_base.predict(input_batch)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = label
        i += 1
        if i * batch_size >= sample_num:
            break
    return features, labels

train_feature, train_label = extract_features(train_dir, 2000)
test_feature, test_label = extract_features(test_dir, 1000)
val_feature, val_label = extract_features(validation_dir, 1000)

train_feature = np.reshape(train_feature, (2000, 4*4*512))
val_feature = np.reshape(val_feature, (1000, 4*4*512))
test_feature = np.reshape(test_feature, (1000, 4*4*512))


# 데이터 전처리
train_datagen = ImageDataGenerator( # 훈련 이미지 증식 하기!! 과대적합 해결 위함
    # 몇 개가 증식 되는지 모르것네...
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory( # 개, 고양이 각 1,000개 씩 총 2,000개 변환
    train_dir, # 타겟 디렉토리
    target_size=(150,150), # 모든 이미지를 150x150으로 변경, 이유는 딥러닝 모델의 입력 차원이 150x150으로 쌓았기 때문에
    batch_size=batch_size,
    class_mode='binary' # 딥러닝 모델의 loss를 binary_crossentropy로 다루기 때문
)
validation_generator = test_data.flow_from_directory(
    validation_dir,
    target_size=(150,150), # 모든 이미지를 150x150으로 변경, 이유는 딥러닝 모델의 입력 차원이 150x150으로 쌓았기 때문에
    batch_size=batch_size,
    class_mode='binary' # 딥러닝 모델의 loss를 binary_crossentropy로 다루기 때문
)

# 완전 분류기를 정의하고 훈련
from tensorflow.keras import models, layers, optimizers

vgg_cnn = models.Sequential()
vgg_cnn.add(conv_base)
vgg_cnn.add(layers.Flatten())
vgg_cnn.add(layers.Dense(256, activation='relu', input_dim=(4*4*512)))
vgg_cnn.add(layers.Dropout(0.5))
vgg_cnn.add(layers.Dense(1, activation='sigmoid'))
vgg_cnn.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])
vgg_cnn.summary()
vgg_cnn.fit(train_generator,
            epochs=30,
            validation_data=(validation_generator),
            validation_steps=50,
            verbose = 2)
vgg_cnn.save('vgg16을 이용한 고양이 개 분류 모델.h5')

