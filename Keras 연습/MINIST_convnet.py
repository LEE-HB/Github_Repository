from keras.datasets import mnist
from keras import layers, models
from tensorflow.keras.utils import to_categorical

(train_image, train_class), (test_image, test_class) = mnist.load_data()

train_image = train_image.reshape(60000,28, 28, 1)
train_image = train_image.astype('float32')

test_image = test_image.reshape(10000,28, 28, 1)
test_image = test_image.astype('float32')

train_class = to_categorical(train_class) #soft Max를 위한 변환
test_class = to_categorical(test_class) #soft Max를 위한 변환

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['acc'])
model.fit(train_image, train_class, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_image, test_class)
print(test_acc,test_loss)
print(64*313, len(test_image))