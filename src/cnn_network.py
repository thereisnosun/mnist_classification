import tensorflow as tf
import src.mnist_loader as mnist_loader
from tensorflow.python import debug as tf_debug

import os

class CNN:
    height = 28
    width = 28
    channels = 1
    n_inputs = height * width

    conv1_fmaps = 32
    conv1_ksize = 3
    conv1_stride = 1
    conv1_pad = "SAME"

    conv2_fmaps = 64
    conv2_ksize = 3
    conv2_stride = 2
    conv2_pad = "SAME"

    pool3_fmaps = conv2_fmaps

    n_fc1 = 64
    n_outputs = 10

    def __init__(self):
        with tf.name_scope("inputs"):
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_inputs], name="X")
            X_reshaped = tf.reshape(self.X, shape=[-1, self.height, self.width, self.channels])
            self.y = tf.placeholder(tf.int32, shape=[None], name="y")

        conv1 = tf.layers.conv2d(X_reshaped, filters=self.conv1_fmaps, kernel_size=self.conv1_ksize,
                                 strides=self.conv1_stride, padding=self.conv1_pad,
                                 activation=tf.nn.relu, name="conv1")
        conv2 = tf.layers.conv2d(conv1, filters=self.conv2_fmaps, kernel_size=self.conv2_ksize,
                                 strides=self.conv2_stride, padding=self.conv2_pad,
                                 activation=tf.nn.relu, name="conv2")

        with tf.name_scope("pool3"):
            pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
            pool3_flat = tf.reshape(pool3, shape=[-1, self.pool3_fmaps * 7 * 7])

        with tf.name_scope("fc1"):
            fc1 = tf.layers.dense(pool3_flat, self.n_fc1, activation=tf.nn.relu, name="fc1")

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(fc1, self.n_outputs, name="output")
            self.Y_proba = tf.nn.softmax(self.logits, name="Y_proba")

        with tf.name_scope("train"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            loss = tf.reduce_mean(xentropy)
            optimizer = tf.train.AdamOptimizer()
            self.training_op = optimizer.minimize(loss)

        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(self.logits, self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.name_scope("init_and_save"):
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        with tf.name_scope("predict"):
            self.prediction = tf.argmax(self.Y_proba, 1)

        self.sess = tf.Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def restore(self, session_path):
        self.saver.restore(self.sess, session_path)

    def train(self, mnist_loader, mnist_loader_test=None, n_epochs=10, batch_size=100, model_name="./my_mnist_model"):
        self.init.run(session=self.sess)
        for epoch in range(n_epochs):
            for iteration in range(mnist_loader.size() // batch_size):
                X_batch, y_batch = mnist_loader.get_next_batch(batch_size)
                self.sess.run(self.training_op, feed_dict={self.X: X_batch, self.y: y_batch})
            acc_train = self.accuracy.eval(feed_dict={self.X: X_batch, self.y: y_batch}, session=self.sess)
            if mnist_loader_test:
                X_test_batch, y_test_batch = mnist_loader_test.get_next_batch(batch_size)
                acc_test = self.accuracy.eval(feed_dict={self.X: X_test_batch, self.y: y_test_batch}, session=self.sess)
                print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
            else:
                print(epoch, "Train accuracy:", acc_train)

            save_path = self.saver.save(self.sess, model_name) #tf_mnist_model.ckpt
            print("The model is saved in ", save_path)

    def predict(self, mnist_loader, batch_size=None):
        if not batch_size:
            x = mnist_loader.get_images()
            return self.prediction.eval(feed_dict={self.X: x}, session=self.sess)
        else:
            x, _ = mnist_loader.get_next_batch(batch_size)
            total_result = []
            for iteration in range(mnist_loader.size() // batch_size):
                batch_result = self.prediction.eval(feed_dict={self.X: x}, session=self.sess)
                total_result.extend(batch_result.tolist())
            return total_result