import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
from model import CreditModelKN

def main():
    # Load data from disk and split into training and validation sets.
    data = np.loadtxt('data/credit-data.csv', dtype=np.int, delimiter=',', skiprows=1)
    characteristics = data[:, 1:6]
    plate = data[:, 6:12]
    pbil = data[:, 12:18]
    ppay = data[:, 18:24]
    averageborrowed = pbil.mean()

    # returnratio = ppay / pbil
    # print("Print line 1000 of pay ratio:", returnratio[1000])

    X = data[:, 1:-1]
    y = data[:, -1]
    print("Shape of data:", X.shape)

    # Fit the model against the training data.
    model = CreditModelKN()

    kfold = KFold(n_splits=10)
    kfold.get_n_splits(X)

    for train_index, test_index in kfold.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)

        # Predict against test data and ensure `y_hat` returns ints.
        y_hat = model.predict(X_test)
        y_hat = np.rint(np.squeeze(y_hat)).astype(int)
        assert len(y_hat) == len(X_test)
        print("Accuracy:  {:06.3f}%".format(100 * accuracy_score(y_test, y_hat)))
        print("Precision: {:06.3f}%".format(100 * precision_score(y_test, y_hat)))
        print("Recall:    {:06.3f}%".format(100 * recall_score(y_test, y_hat)))

if __name__ == '__main__':
    main()
