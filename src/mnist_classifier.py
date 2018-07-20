import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
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
    return joblib.load(filename)


def main():
    dataset = load_dataset("train.csv")
    y = dataset['label']
    dataset.drop(['label'], axis=1, inplace=True)
    print(dataset.head())
    print(dataset.info())
    #plot_digit(dataset.iloc[5].values)


    classifier = OneVsRestClassifier(LinearSVC(random_state=13))
    classifier.fit(dataset, y)

    val_score = cross_val_score(classifier, dataset, y, cv=3, scoring="accuracy");
    print(val_score)

    save_classifier(classifier, "one_vs_rest.pkl")
    test_classifier(classifier)


if __name__ == "__main__":
    main()