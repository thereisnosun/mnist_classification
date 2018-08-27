import tensorflow as tf
import tensorflow.keras as keras
import os
import pandas as pd
import src.mnist_loader as mnist


DATA_PATH = '../data/'




model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'))
model.add(keras.layers.MaxPool3D(pool_size=[1, 2, 1],strides=[1, 2, 1], padding="VALID"))
model.add(keras.layers.Softmax())

model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mnist_train = mnist.MnistDataset('train.csv')
model.fit(mnist_train.get_images(), mnist_train.get_labels(), epochs=3, batch_size=100)
model_eval = model.evaluate(mnist_train.get_labels(), mnist_train.get_labels(), batch_size=100)
print(model_eval)

mnist_test = mnist.MnistDataset('test.csv')
model_predict = model.predict(mnist_test.get_images(), batch_size=100)

print(model_predict)


model.save_weights('./data/keras_weights')
