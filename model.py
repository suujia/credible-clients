import numpy as np


class CreditModel:
    def __init__(self):
        """Initializes the model object."""
        # TODO: Initialize any class objects/variables you might need
        pass

    def fit(self, X, y):
        """Fits the model based on the given X and y."""
        # TODO: Fit your model
        pass

    def predict(self, X):
        """Returns y_hat for a given X, after fitting."""
        # TODO: Return a prediction for your model
        return np.random.randint(2, size=len(X))
