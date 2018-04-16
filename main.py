import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from model import CreditModel


def main():
    # Load data from disk and split into training and validation sets.
    data = np.loadtxt('data/credit-data.csv', dtype=np.int, delimiter=',', skiprows=1)
    X, y = data[:, 1:-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Fit the model against the training data.
    model = CreditModel()
    model.fit(X_train, y_train)

    # Predict against test data and ensure `y_hat` returns ints.
    y_hat = model.predict(X_test)
    y_hat = np.rint(np.squeeze(y_hat)).astype(int)
    assert len(y_hat) == len(X_test)

    # Print out accuracy/precision/recall scores.
    print("Accuracy:  {:06.3f}%".format(100 * accuracy_score(y_test, y_hat)))
    print("Precision: {:06.3f}%".format(100 * precision_score(y_test, y_hat)))
    print("Recall:    {:06.3f}%".format(100 * recall_score(y_test, y_hat)))


if __name__ == '__main__':
    main()
