import numpy as np
from sklearn.model_selection import train_test_split

from model import CreditModel


def main():
    # Load data from disk and split into training and validation sets.
    data = np.loadtxt('data/credit-data.csv', dtype=np.int, delimiter=',', skiprows=1)
    X, y = data[:, 1:-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Fit the model against training data, and print its accuracy on test data.
    model = CreditModel()
    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)
    assert len(y_hat) == len(X_test)
    accuracy = np.sum(np.squeeze(y_hat) == y_test) / y_test.size
    print("Model accuracy: {:.3f}%".format(accuracy * 100))


if __name__ == '__main__':
    main()
