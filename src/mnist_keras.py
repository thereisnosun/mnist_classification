import tensorflow as tf
import tensorflow.keras as keras
import os
import pandas as pd
import src.mnist_loader as mnist
import numpy as np

from keras.datasets import mnist as keras_mnist
from keras import backend as K

img_rows, img_cols = 28, 28


def create_prediction_file(prediction):
    df = pd.DataFrame({'ImageId': range(1, len(prediction) + 1),  'Label': prediction})
    df.to_csv('../data/predict_number.csv', index=False)

DATA_PATH = '../data/'


num_classes = 10

model = keras.Sequential()
#model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(1, 784, 1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Softmax())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

#model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#
print(model.summary())
mnist_train = mnist.MnistDataset('train.csv')
x_train_kaggle = mnist_train.get_images()
x_train_kaggle.shape = (x_train_kaggle.shape[0],
                     x_train_kaggle.shape[1] // img_cols, x_train_kaggle.shape[1] // img_cols, 1 )
y_train = keras.utils.to_categorical(mnist_train.get_labels(), num_classes)
#y_train = mnit_train.get_labels()
print(type(y_train), y_train.shape)
print(y_train[0])
model.fit(x_train_kaggle, y_train, epochs=15, batch_size=100)


model_eval = model.evaluate(x_train_kaggle, y_train, batch_size=100)
print(model_eval)

model.save_weights('./data/keras_weights')

#model.load_weights('./data/keras_weights')

mnist_test = mnist.MnistDataset('test.csv')
x_test = mnist_test.get_images()
x_test.shape = (x_test.shape[0], x_test.shape[1] // img_cols, x_test.shape[1] // img_cols, 1)
model_predict = model.predict_classes(x_test, batch_size=100, verbose=1)


print(type(model_predict), model_predict.shape, len(model_predict))
print(model_predict[0])

create_prediction_file(model_predict)




