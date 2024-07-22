import numpy as np
from collections import Counter
from abc import ABC, abstractmethod


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree(ABC):
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    @abstractmethod
    def predict(self, X):
        pass

    def _build_tree(self, X, y, depth=0):
        n_samples = len(X)
        n_labels = len(np.unique(y))

        # check stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            return self._create_leaf(y)

        # find best split
        feature, threshold = self._best_split(X, y)

        # recursively build left and right subtrees
        left_idx, right_idx = self._split(X[:, feature], threshold)
        left = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    @abstractmethod
    def _create_leaf(self, y):
        pass

    @abstractmethod
    def _information_gain(self, X_column, y, threshold):
        pass

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None

        # random subset of features
        if self.n_features is None:
            subset_of_features = range(X.shape[1])
        else:
            subset_of_features = np.random.choice(
                range(X.shape[1]), self.n_features, replace=False
            )

        for feature_idx in subset_of_features:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # calculate informaition gain
                gain = self._information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split(self, X_column, threshold):
        left_idx = np.argwhere(X_column <= threshold).flatten()
        right_idx = np.argwhere(X_column > threshold).flatten()
        return left_idx, right_idx

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class ClassificationTree(DecisionTree):
    def _information_gain(self, X_column, y, threshold):
        # partent entropy
        parent_entropy = self._entropy(y)

        # left and right child entropy
        left_idx, right_idx = self._split(X_column, threshold)
        left_entropy = self._entropy(y[left_idx])
        right_entropy = self._entropy(y[right_idx])

        # weighted average
        children_entropy = (
            left_entropy * len(left_idx) + right_entropy * len(right_idx)
        ) / len(y)
        information_gain = parent_entropy - children_entropy
        return information_gain

    def _entropy(self, y):
        entropy = 0

        for label in np.unique(y):
            p = y[y == label].shape[0] / y.shape[0]
            entropy -= p * np.log2(p) if p > 0 else 0

        return entropy
    
    def _create_leaf(self, y):
        counter = Counter(y)
        if not counter:  # Check if counter is empty
            raise ValueError("The label list is empty.")
        return Node(value=counter.most_common(1)[0][0])


class RegressionTree(DecisionTree):

    def _information_gain(self, X_column, y, threshold):
        # parent mean
        parent_var = np.var(y)

        # left and right child mean
        left_idx, right_idx = self._split(X_column, threshold)
        
        left_var = np.var(y[left_idx]) if len(left_idx) > 0 else 0
        right_var = np.var(y[right_idx]) if len(right_idx) > 0 else 0

        # weighted average
        children_var = (left_var * len(left_idx) + right_var * len(right_idx)) / len(y)
        var_reduction = parent_var - children_var
        return var_reduction
    
    def _create_leaf(self, y):
        return Node(value=np.mean(y))
    
    
    
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, r2_score

    # Classification example
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_clf = ClassificationTree(max_depth=5)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    print(f"Classification accuracy: {accuracy_score(y_test, y_pred)}")

    # Regression example
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_reg = RegressionTree(max_depth=5)
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)
    print(f"Regression R2: {r2_score(y_test, y_pred)}")
