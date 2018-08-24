import os

import numpy as np
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

DATA_PATH = '../data/'


def convert_X(X):
    X = X.values
    X = X.astype(np.float32) / 255.0
    return X


def load_dataset(dataset_path):
    csv_path = os.path.join(DATA_PATH, dataset_path)
    return pd.read_csv(csv_path)


class MnistDataset:
    def __init__(self, path):
        dataset  = load_dataset(path)
        y = dataset['label']
        dataset.drop(['label'], axis=1, inplace=True)
        self.X = convert_X(dataset)
        self.y = y.values.astype(np.int32)
        self.curr_batch = 0

    def get_next_batch(self, batch_size):
        real_curr_batch = self.curr_batch
        self.curr_batch += batch_size #TODO: check if list slicing is correct
        return self.X[real_curr_batch: real_curr_batch + batch_size], \
               self.y[real_curr_batch: real_curr_batch + batch_size]

    def size(self):
        #print(self.X.shape)
        return self.X.shape[0]

    def get_images(self):
        return self.X

    def get_labels(self):
        return self.y


class MnistDatasetTF:
    def __init__(self, is_train=True):
        if is_train:
            self.mnist = input_data.read_data_sets("/tmp/data/").train
        else:
            self.mnist = input_data.read_data_sets("/tmp/data").test

    def get_next_batch(self, batch_size):
        return self.mnist.next_batch(batch_size)

    def size(self):
        return self.mnist.num_examples

    def get_images(self):
        return self.mnist.images

    def get_labels(self):
        return self.mnist.labels