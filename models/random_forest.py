import numpy as np
from collections import Counter
from decision_tree import ClassificationTree, RegressionTree
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
import time

class RandomForest(ABC):
    """
    Abstract base class for Random Forest algorithms.
    
    Implements the common functionality for both classification and regression
    random forests, using parallel processing for tree training and prediction.

    Attributes:
        n_trees (int): Number of trees in the forest.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        max_depth (int): Maximum depth of the trees.
        n_features (int): Number of features to consider when looking for the best split.
        n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
    """
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=10, n_features=None, n_jobs=-1):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_jobs = n_jobs
        self.trees = []

    def fit(self, X, y):
        # fits decision trees in parallel
        if self.n_features is None:
            self.n_features = int(np.sqrt(X.shape[1]))
        
        # parallel training
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_tree)(X, y) for _ in range(self.n_trees)
        )

    def _train_tree(self, X, y):
        # booststap and create tree
        X_sample, y_sample = self._bootstrap(X, y)
        tree = self._create_tree()
        tree.fit(X_sample, y_sample)
        return tree

    def _bootstrap(self, X, y):
        # ordinary boostrap
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    @abstractmethod
    def _create_tree(self):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class RandomForestClassifier(RandomForest):
    def _create_tree(self):
        return ClassificationTree(
            min_samples_split=self.min_samples_split,
            max_depth=self.max_depth,
            n_features=self.n_features,
        )
    
    def predict(self, X):
        # predict most common class in parallel
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees
        )
        
        y_pred = np.array(predictions)
        y_pred = np.swapaxes(y_pred, 0, 1)
        y_pred = np.array([self._majority_vote(y) for y in y_pred])
        return y_pred
        
    def _majority_vote(self, y):
        most_common = Counter(y).most_common(1)
        return most_common[0][0]
        
class RandomForestRegressor(RandomForest):
    def _create_tree(self):
        return RegressionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features,
            )

    def predict(self, X):
        # predicts mean of trees in parallel
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees
        )
        y_pred = np.array(predictions)
        y_pred = np.swapaxes(y_pred, 0, 1)
        y_pred = np.array([np.mean(y) for y in y_pred])
        return y_pred

def run_experiment(n_samples, n_features, n_trees, n_jobs):
    # run an experiment for both classification and regression random forest, outputs accuary, r2 and time to train
    print(f"\nRunning experiment with {n_samples} samples, {n_features} features, {n_trees} trees, and {n_jobs} jobs")
    
    # Classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()
    rf_clf = RandomForestClassifier(n_trees=n_trees, max_depth=15, n_jobs=n_jobs)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    end_time = time.time()
    print(f"Classification accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification time: {end_time - start_time:.2f} seconds")

    # Regression
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()
    rf_reg = RandomForestRegressor(n_trees=n_trees, max_depth=15, n_jobs=n_jobs)
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)
    end_time = time.time()
    print(f"Regression MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"Regression R2: {r2_score(y_test, y_pred)}")
    print(f"Regression time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

    # run experiments with different configurations
    # run_experiment(n_samples=1000, n_features=10, n_trees=50, n_jobs=1)
    run_experiment(n_samples=1000, n_features=10, n_trees=50, n_jobs=-1)