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
import src.mnist_loader as mnist_loader
#import src.cnn_network.CNN as CNN
import src.cnn_network as cnn_network

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
    mnist_dataset = mnist_loader.MnistDatasetTF()
    mnist_test_dataset = mnist_loader.MnistDatasetTF(False)
    n_epoch = 3
    batch_size = 100

    cnn_tf= cnn_network.CNN()
    #cnn_tf.train(mnist_dataset, mnist_test_dataset, n_epoch, batch_size, "./data/tf_mnist_model.ckpt")
    #cnn_tf.restore("./data/tf_mnist_model.ckpt")
    cnn_tf.predict(mnist_test_dataset)



def predict():
    #prediction = tf.argmax(logits, 1)
    with tf.name_scope("restore"):
        init = tf.global_variables_initializer()
        model_restore = tf.train.Saver()

    with tf.Session() as sess:
        model_restore.restore("./data/mnist_cnn_model")
        model_restore.build()

    #TODO: load saved model, use for prediction

# def test_model():
#     prediction = tf.argmax(y, 1)
#     print
#     prediction.eval(feed_dict={x: mnist.test.images})


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
    #predict()
    #run_cnn()