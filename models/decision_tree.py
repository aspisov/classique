import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        
        # initialized by regression and classification tree
        self._get_leaf_value = None
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0):
        n_samples = len(X)
        n_labels = len(np.unique(y))
        
        # check stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._get_leaf_value(y)
            return Node(value=leaf_value)
        
        # find best split
        feature, threshold = self._best_split(X, y)
        
        # recursively build left and right subtrees
        left_idx, right_idx = self._split(X[:, feature], threshold)
        left = self._build_tree(X[left_idx, :], y[left_idx], depth+1)
        right = self._build_tree(X[right_idx, :], y[right_idx], depth+1)
        
        return Node(feature=feature, threshold=threshold, left=left, right=right)
    
    def _information_gain(self, X_column, y, threshold):
        pass
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        
        for feature_idx in range(X.shape[1]):
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
    def fit(self, X, y):
        self._get_leaf_value = self._most_common_label
        super(ClassificationTree, self).fit(X, y)
    
    def _information_gain(self, X_column, y, threshold):
        # partent entropy
        parent_entropy = self._entropy(y)
        
        # left and right child entropy
        left_idx, right_idx = self._split(X_column, threshold)
        left_entropy = self._entropy(y[left_idx])
        right_entropy = self._entropy(y[right_idx])
        
        # weighted average
        children_entropy = (left_entropy * len(left_idx) + right_entropy * len(right_idx)) / len(y)
        information_gain = parent_entropy - children_entropy
        return information_gain
    
    def _entropy(self, y):
        entropy = 0
        
        for label in np.unique(y):
            p = y[y == label].shape[0] / y.shape[0]
            entropy -= p * np.log2(p) if p > 0 else 0
        
        return entropy
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]


class RegressionTree(DecisionTree):
    def fit(self, X, y):
        self._get_leaf_value = np.mean
        super(RegressionTree, self).fit(X, y)
    
    def _information_gain(self, X_column, y, threshold):
        # parent mean
        parent_var = np.var(y)
        
        # left and right child mean
        left_idx, right_idx = self._split(X_column, threshold)
        left_var = np.var(y[left_idx])
        right_mean = np.var(y[right_idx])
        
        # weighted average
        children_var = (left_var * len(left_idx) + right_mean * len(right_idx)) / len(y)
        var_reduction = parent_var - children_var
        return var_reduction
