import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
import tensorflow as tf
import numpy as np

DATA_PATH = '../data/'


def load_dataset(dataset_path):
    csv_path = os.path.join(DATA_PATH, dataset_path)
    return pd.read_csv(csv_path)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()


def create_prediction_file(test_dataset, prediction):
    df = pd.DataFrame({'ImageId': test_dataset.index + 1, 'Label': prediction})
    df.to_csv('../data/predict_number.csv', index=False)


def test_classifier(classifier):
    test_dataset = load_dataset("test.csv")
    result = classifier.predict(test_dataset)
    create_prediction_file(test_dataset, result)

def save_classifier(clf, filename):
    full_path = os.path.join(DATA_PATH, filename)
    _ = joblib.dump(clf, full_path, compress=9)


def load_classifier(filename):
    full_path = os.path.join(DATA_PATH, filename)
    return joblib.load(full_path)


def call_classifier(dataset, y, class_creator, class_name, use_old=True):
    if use_old:
        classifier = load_classifier(class_name)
    else:
        classifier = class_creator()
        classifier.fit(dataset, y)
        val_score = cross_val_score(classifier, dataset, y, cv=3, scoring="accuracy");
        print(val_score)
        save_classifier(classifier, class_name)

    test_classifier(classifier)
    print("Prediction file is created")


def convert_X(X):
    X = X.values
    X = X.astype(np.float32) / 255.0
    return X


class MnistDataset:
    def __init__(self, path):
        dataset  = load_dataset("train.csv")
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
        return self.X.shape[0]

    def get_images(self):
        return self.X

    def get_labels(self):
        return self.y

def tf_test():
    X = load_dataset("train.csv")
    y = X['label']
    X.drop(['label'], axis=1, inplace=True)
    X = convert_X(X)
    y = y.values.astype(np.int32)
    feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
    dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10, optimizer=tf.train.AdamOptimizer(1e-4),
                                         feature_columns=feature_cols, dropout=0.1)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X}, y=y, num_epochs=100, batch_size=50, shuffle=True)
    dnn_clf.train(input_fn=input_fn)

    accuracy = dnn_clf.evaluate(input_fn=input_fn)
    print("The model accuracy is - ", accuracy)

    X_test_data = load_dataset('test.csv')
    X_test = convert_X(X_test_data)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_test},
        num_epochs=1,
        shuffle=False)

    prediction = dnn_clf.predict(input_fn=predict_input_fn)
    prediction_list = []
    for curr_pred in list(prediction):
        prediction_list.append(curr_pred['class_ids'][0])
    # for predict in prediction_list:
    #     print(predict)
    create_prediction_file(X_test_data, pd.Series(prediction_list))


def build_cnn():
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

    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
        X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
        y = tf.placeholder(tf.int32, shape=[None], name="y")

    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                             strides=conv2_stride, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv2")

    with tf.name_scope("pool3"):
        pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])

    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

    with tf.name_scope("output"):
        logits = tf.layers.dense(fc1, n_outputs, name="output")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

    with tf.name_scope("train"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    ######Train the network(TODO:  move entire cnn implementation to seperate class)######
    mnist_dataset = MnistDataset("train.csv")
    mnist_test_dataset = MnistDataset("test.csv")
    n_epoch = 10
    batch_size = 100

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epoch):
            for iteration in range(mnist_dataset.size() // batch_size):
                X_batch, y_batch = mnist_dataset.get_next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: mnist_test_dataset.get_images(),
                                                y: mnist_test_dataset.get_labels()})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
            save_path = saver.save(sess, "./data/mnist_cnn_model")


def main():
    dataset = load_dataset("train.csv")
    y = dataset['label']
    dataset.drop(['label'], axis=1, inplace=True)
    print(dataset.head())
    print(dataset.info())
    #plot_digit(dataset.iloc[5].values)
    #
    # call_classifier(dataset, y, lambda:  OneVsRestClassifier(LinearSVC(random_state=13)),
    #   "one_vs_rest.pkl", False)

    # call_classifier(dataset, y, lambda: OneVsOneClassifier(SGDClassifier(random_state=42)),
    #                 "one_vs_one.pkl")

    # call_classifier(dataset, y, lambda: RandomForestClassifier(random_state=13, n_estimators=100, n_jobs=-1),
    #                 "random_forest.pkl", False)
    #
    # call_classifier(dataset, y, lambda:     DecisionTreeClassifier(max_depth=5),
    #                  "decision_tree.pkl", False)

    one_vs_rest = OneVsRestClassifier(SVC(random_state=13, kernel='linear',probability=True))
    one_vs_one = OneVsOneClassifier(SGDClassifier(random_state=42))
    rf_clf = RandomForestClassifier(random_state=13, n_estimators=100, n_jobs=-1)
    call_classifier(dataset, y, lambda: VotingClassifier(
        estimators=[('omne_rest', one_vs_rest), ('rf', rf_clf), ('one_vs_all', one_vs_one)],
        voting='soft'),
                    "voting_combine.pkl", False)


    # call_classifier(dataset, y, lambda: KNeighborsClassifier(),
    #                 "k_neighbours.pkl", False) #this shit runs waaay too long

if __name__ == "__main__":
    build_cnn()
    #run_cnn()