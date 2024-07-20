import numpy as np

def precision(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    false_positives = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    return true_positives / (true_positives + false_positives)

def recall(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    false_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return true_positives / (true_positives + false_negatives)

def accuracy(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    true_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    return (true_positives + true_negatives) / len(y_true)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)

def confusion_matrix(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    false_positives = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    false_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    true_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    return np.array([[true_positives, false_positives], [false_negatives, true_negatives]])

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)