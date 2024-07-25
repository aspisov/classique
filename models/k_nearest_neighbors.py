import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import mode

class KNeighborsBase(ABC):
    """
    Base class for KNeighborsRegressor and KNeighborsClassifier
    
    Attributes:
        k (int): Number of neighbors
        metric (str): Distance metric
    """
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    @abstractmethod
    def _predict(self, x):
        # makes prediction for individual example
        pass
        
    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2)**2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported distance metric")

class KNeighborsClassifier(KNeighborsBase):
    def _predict(self, x):
        distances = np.array([self._distance(x, x_train) for x_train in self.X_train])
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return mode(k_nearest_labels, keepdims=False).mode

class KNeighborsRegressor(KNeighborsBase):
    def _predict(self, x):
        distances = np.array([self._distance(x, x_train) for x_train in self.X_train])
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_targets = self.y_train[k_indices]
        return np.mean(k_nearest_targets)

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    
    # Classification
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, 
                               n_clusters_per_class=1, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = KNeighborsClassifier(k=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Classification Accuracy: {accuracy_score(y_pred, y_test):.2f}")
    
    # Visualize classification results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', alpha=0.7, s=50)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.3, s=100)
    plt.title('KNN Classification Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter, ticks=range(3), label='Class')
    plt.show()
    
    # Regression
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsRegressor(k=5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"Regression MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Regression R2 score: {r2_score(y_test, y_pred):.2f}")
    