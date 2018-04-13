import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier


def load_data(path, seed=42):
    data = np.loadtxt(path, dtype=np.int, delimiter=',', skiprows=1)
    return train_test_split(data[:, 1:-1], data[:, -1], random_state=seed)


def main():
    # Load data from disk and split into training and validation sets.
    X_train, X_test, y_train, y_test = load_data('data/credit-data.csv')

    # TODO: Replace this dummy classifier with your own model!
    model = DummyClassifier(strategy='uniform')
    model.fit(X_train, y_train)

    # Print out the model accuracy after predicting against `X_test`.
    y_hat = model.predict(X_test)
    accuracy = np.sum(np.squeeze(y_hat) == y_test) / y_test.size
    print("Model accuracy: {:.6f}".format(accuracy))


if __name__ == '__main__':
    main()
