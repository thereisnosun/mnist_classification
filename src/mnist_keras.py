import tensorflow as tf
import tensorflow.keras as keras
import os
import pandas as pd
import src.mnist_loader as mnist
import numpy as np


DATA_PATH = '../data/'


x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

print(type(x_train), x_train.shape)
print(type(y_train), y_train)


model = keras.Sequential()
#model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(1, 784, 1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(784, 1 , 1)))
model.add(keras.layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 1), padding="VALID"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Softmax())

model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
mnist_train = mnist.MnistDataset('train.csv')
model.fit(mnist_train.get_images().reshape(mnist_train.get_images().shape +(1,) + (1,)), mnist_train.get_labels(), epochs=3, batch_size=100)
model_eval = model.evaluate(mnist_train.get_images(), mnist_train.get_labels(), batch_size=100)
print(model_eval)

mnist_test = mnist.MnistDataset('test.csv')
model_predict = model.predict(mnist_test.get_images(), batch_size=100)

print(model_predict)


model.save_weights('./data/keras_weights')
