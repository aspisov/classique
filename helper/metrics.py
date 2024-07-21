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

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true).mean()

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (y_true + y_pred))

def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(np.mean((np.log(y_true + 1) - np.log(y_pred + 1)) ** 2))

def r2_score(y_true, y_pred):
    """
    Computes the coefficient of determination.
    """
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

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

def roc_auc_score(y_true, y_scores):
    """
    Computes the Area Under the Receiver Operating Characteristic (ROC) curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return np.trapz(tpr, fpr)
