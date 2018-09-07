import numpy as np
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class CreditModelKN:
    def __init__(self):
        """
        Instantiates the model object, creating class variables if needed.
        """
        # TODO: Initialize your model object.
        models = []
        
    def fit(self, X_train, y_train):
        """
        Fits the model based on the given `X_train` and `y_train`.

        You should somehow manipulate and store this data to your model class
        so that you can make predictions on new testing data later on.
        """
        # TODO: Fit your model based on the given X and y.
        
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Returns `y_hat`, a prediction for a given `X_test` after fitting.

        You should make use of the data that you stored/computed in the
        fitting phase to make your prediction on this new testing data.
        """
        # TODO: Predict on `X_test` based on what you learned in the fit phase.
        # return np.random.randint(2, size=len(X_test))
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        min_dist = euc(row, self.X_train[0])
        min_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if (dist < min_dist):
                min_dist = dist
                min_index = i
        return self.y_train[min_index]
