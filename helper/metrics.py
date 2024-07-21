import numpy as np

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def precision(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    false_positives = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    false_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return true_positives / (true_positives + false_negatives)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def confusion_matrix(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    false_positives = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    false_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    true_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    return np.array(
        [[true_positives, false_positives], [false_negatives, true_negatives]]
    )


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    """
    Computes the coefficient of determination.
    """
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def roc_auc_score(y_true, y_pred):
    """
    Computes area under the ROC curve.
    """
    indicator = lambda a, b: 0.5 if a == b else a < b

    n = len(y_true)
    numerator = 0
    denominator = 0

    for i in range(n):
        for j in range(n):
            numerator += (y_true[i] < y_true[j]) * indicator(y_pred[i], y_pred[j])
            denominator += y_true[i] < y_true[j]

    return numerator / denominator

def roc_curve(y_true, y_scores):
    """
    Computes the Receiver Operating Characteristic (ROC) curve.
    """
    thresholds = np.unique(y_scores)
    tpr = []
    fpr = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    return np.array(fpr), np.array(tpr), thresholds
