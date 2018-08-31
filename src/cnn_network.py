import tensorflow as tf
import src.mnist_loader as mnist_loader
from tensorflow.examples.tutorials.mnist import input_data

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
            Y_proba = tf.nn.softmax(self.logits, name="Y_proba")

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
            correct = tf.nn.in_top_k(self.logits, self.y, 1)
            self.prediction = tf.argmax(self.logits, 1)


    def restore(self, session_path):
        #tf.reset_default_graph()
        with tf.Session() as sess:
            # saver = tf.train.import_meta_graph(session_path)
            # saver.restore(sess, )
            self.saver.restore(sess, session_path)
            # self.saver.build()

    def train(self, mnist_loader, mnist_loader_test=None, n_epochs=10, batch_size=100, model_name="./my_mnist_model"):
        with tf.Session() as sess:
            self.init.run()
            for epoch in range(n_epochs):
                for iteration in range(mnist_loader.size() // batch_size):
                    X_batch, y_batch = mnist_loader.get_next_batch(batch_size)
                    sess.run(self.training_op, feed_dict={self.X: X_batch, self.y: y_batch})
                acc_train = self.accuracy.eval(feed_dict={self.X: X_batch, self.y: y_batch})
                if mnist_loader_test:
                    acc_test = self.accuracy.eval(feed_dict={self.X: mnist_loader_test.get_images(), self.y: mnist_loader_test.get_labels()})
                    print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
                else:
                    print(epoch, "Train accuracy:", acc_train)

                save_path = self.saver.save(sess, model_name) #tf_mnist_model.ckpt
                print("The model is saved in ", save_path)

    def predict(self, mnist_loader):
        with tf.Session() as sess:
            self.init.run()
            #result = self.prediction.eval(feed_dict={self.X: mnist_loader.get_images()})
            result = sess.run(self.prediction, feed_dict={self.X: mnist_loader.get_images()})
            print (len(result), type(result))
            return result