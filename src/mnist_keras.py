import tensorflow as tf
import tensorflow.keras as keras
import os
import pandas as pd
import src.mnist_loader as mnist
import numpy as np

from keras.datasets import mnist as keras_mnist
from keras import backend as K

img_rows, img_cols = 28, 28

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = keras_mnist.load_data()
# print(type(x_train), x_train.shape)
# print(type(y_train), y_train.shape, y_train[0])
#
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
#     print("Channel first", type(x_train), x_train.shape)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#     print("Channel NOT first", type(x_train), x_train.shape)
#
# mnist_train = mnist.MnistDataset('train.csv')
# x_train_kaggle = mnist_train.get_images()
# print(type(x_train_kaggle), x_train_kaggle.shape,
#       x_train_kaggle.size, x_train_kaggle.shape[0],  x_train_kaggle.shape[1])
#
# y_train_kaggle = mnist_train.get_labels()
# print("Y train", type(y_train_kaggle), y_train_kaggle.shape, y_train_kaggle[1], y_train_kaggle[999])
# x_train_kaggle.shape = (x_train_kaggle.shape[0],
#                     x_train_kaggle.shape[1] // img_cols, x_train_kaggle.shape[1] // img_cols, 1 )
# print(type(x_train_kaggle ), x_train_kaggle.shape)
#
# exit(0)

DATA_PATH = '../data/'


x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

print(type(x_train), x_train.shape)
print(type(y_train), y_train)

num_classes = 10

model = keras.Sequential()
#model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(1, 784, 1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
#model.add(keras.layers.Softmax())
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
mnist_train = mnist.MnistDataset('train.csv')
x_train_kaggle = mnist_train.get_images()
x_train_kaggle.shape = (x_train_kaggle.shape[0],
                    x_train_kaggle.shape[1] // img_cols, x_train_kaggle.shape[1] // img_cols, 1 )
#y_train = keras.utils.to_categorical(mnist_train.get_labels(), num_classes)
y_train = mnist_train.get_labels()
#X_train = mnist_train.get_images().reshape(mnist_train.get_images().shape +(1,) + (1,))
model.fit(x_train_kaggle, y_train, epochs=3, batch_size=100)
model_eval = model.evaluate(x_train_kaggle, y_train, batch_size=100)
print(model_eval)

model.save_weights('./data/keras_weights')

mnist_test = mnist.MnistDataset('test.csv')
model_predict = model.predict(mnist_test.get_images(), batch_size=100)

print(model_predict)



