import numpy as np
from decision_tree import ClassificationTree, RegressionTree
from abc import ABC, abstractmethod
from tqdm import tqdm

class GradientBoosting(ABC):
    """
    Abstract base class for Gradient Boosting algorithms.
    
    Implements the common functionality for both classification and regression.

    Attributes:
        n_estimators (int): Number of trees in the forest.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        max_depth (int): Maximum depth of the trees.
        learning_rate (float): Learning rate shrinks the contribution of each tree by `learning_rate`.
    """
    def __init__(
        self, n_estimators=100, min_samples_split=2, max_depth=10, learning_rate=0.1
    ):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.F0 = None

    def fit(self, X, y):
        # at first we only have one leaf
        self.F0 = self._initial_prediction(y)
        F = np.full(len(y), self.F0)
        
        for _ in tqdm(range(self.n_estimators)):
            residuals = self._gradient(y, F)
            tree = RegressionTree(self.min_samples_split, self.max_depth)
            # next tree tries to predict residuals
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # update predictions
            F += self.learning_rate * tree.predict(X)
        
    @abstractmethod
    def predict(self, X):
        pass
        
    @abstractmethod
    def _initial_prediction(self, y):
        pass
    
    @abstractmethod
    def _gradient(self, y, F):
        pass
    
class GradientBoostingRegressor(GradientBoosting):
    def _initial_prediction(self, y):
        return np.mean(y)
    
    def _gradient(self, y, F):
        return y - F
    
    def predict(self, X):
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

class GradientBoostingClassifier(GradientBoosting):
    def _initial_prediction(self, y):
        p = np.mean(y)
        return np.log(p / (1 - p))  # log-odds
    
    def _gradient(self, y, F):
        p = self._sigmoid(F)  # convert to probability
        return y - p
    
    def predict_proba(self, X):
        F = np.full(len(X), self.F0)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return self._sigmoid(F)  # convert to probability
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
if __name__ == "__main__":
    from sklearn.datasets import make_regression, make_classification
    from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    
    # Regression example
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=50, max_depth=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Regression - R2: {r2_score(y_test, y_pred)}")
    print(f"Regression - MSE: {mean_squared_error(y_test, y_pred)}")
    
    # Classification example
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Classification - Accuracy: {accuracy_score(y_test, y_pred)}")
    probs = model.predict_proba(X_test)
    print(f"Classification - ROC AUC: {roc_auc_score(y_test, probs)}")
    