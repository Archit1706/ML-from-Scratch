import numpy as np
from collections import Counter


def euclidean_distance(x_1, x_2):
    distance = np.sqrt(np.sum((x_1 - x_2) ** 2))
    return distance


class KNN:
    def __init__(self, k=3):
        """Initializes the KNN model with k neighbours"""
        self.k = k

    def fit(self, X, y):
        """Stores the training data and labels"""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predicts the labels for the input data"""
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        """Predicts the label for the input data (helper function)"""
        # Compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the k nearest neighbours and their labels
        k_nearest_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # Return the majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
