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



def main():
    dataset = load_dataset("train.csv")
    y = dataset['label']
    dataset.drop(['label'], axis=1, inplace=True)
    print(dataset.head())
    print(dataset.info())
    #plot_digit(dataset.iloc[5].values)

    # call_classifier(dataset, y, lambda:  OneVsRestClassifier(LinearSVC(random_state=13)),
    #   "one_vs_rest.pkl")
    #
    # call_classifier(dataset, y, lambda: OneVsOneClassifier(SGDClassifier(random_state=42)),
    #                 "one_vs_one.pkl")

    call_classifier(dataset, y, lambda: RandomForestClassifier(random_state=13, n_estimators=100, n_jobs=-1),
                    "random_forest.pkl", False)
    #
    # call_classifier(dataset, y, lambda:     DecisionTreeClassifier(max_depth=5),
    #                  "decision_tree.pkl", False)

    # call_classifier(dataset, y, lambda: KNeighborsClassifier(),
    #                 "k_neighbours.pkl", False) #this shit runs waaay too long

if __name__ == "__main__":
    main()