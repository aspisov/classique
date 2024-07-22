import numpy as np
from collections import Counter
from decision_tree import ClassificationTree, RegressionTree
from abc import ABC, abstractmethod
from tqdm import tqdm


class RandomForest(ABC):
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=10, n_features=None):
        self.n_tress = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        if self.n_features is None:
            self.n_features = int(np.sqrt(X.shape[1]))
        for tree in tqdm(self.trees):
            X_sample, y_sample = self._bootstrap(X, y)
            tree.fit(X_sample, y_sample)

    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    @abstractmethod
    def predict(self, X):
        pass


class RandomForestClassifier(RandomForest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trees = [
            ClassificationTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features,
            )
            for _ in range(self.n_tress)
        ]
        
    def predict(self, X):
        y_pred = np.array([tree.predict(X) for tree in self.trees])
        y_pred = np.swapaxes(y_pred, 0, 1)
        y_pred = np.array([self._majority_vote(y) for y in y_pred])
        return y_pred
        
    def _majority_vote(self, y):
        most_common = Counter(y).most_common(1)
        return most_common[0][0]
        
class RandomForestRegressor(RandomForest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trees = [
            RegressionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features,
            )
            for _ in range(self.n_tress)
        ]

    def predict(self, X):
        y_pred = np.array([tree.predict(X) for tree in self.trees])
        y_pred = np.swapaxes(y_pred, 0, 1)
        y_pred = np.array([np.mean(y) for y in y_pred])
        return y_pred
        


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

    # Classification example
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_clf = RandomForestClassifier(n_trees=10, max_depth=10)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    print(f"Classification accuracy: {accuracy_score(y_test, y_pred)}")

    # Regression example
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_reg = RandomForestRegressor(n_trees=10, max_depth=10)
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)
    print(f"Regression MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"Regression R2: {r2_score(y_test, y_pred)}")