import numpy as np


def get_confusion_matrix(y_hat: np.ndarray, y_true: np.ndarray):
    assert len(y_hat) == len(y_true)

    no_of_classes = np.max(y_true) + 1
    confusion_matrix = np.zeros((no_of_classes, no_of_classes), dtype=int)

    for y_predicted, y_true in zip(y_hat, y_true):
        confusion_matrix[y_true][y_predicted] += 1

    return confusion_matrix


def get_accuracy(confusion_matrix: np.ndarray):
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)


def get_f1(confusion_matrix: np.ndarray):
    precision = get_precision(confusion_matrix)
    recall = get_recall(confusion_matrix)
    return 2 * (precision * recall) / (precision + recall)


def get_precision(confusion_matrix: np.ndarray):
    assert confusion_matrix.shape == (2, 2)
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])


def get_recall(confusion_matrix: np.ndarray):
    assert confusion_matrix.shape == (2, 2)
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
