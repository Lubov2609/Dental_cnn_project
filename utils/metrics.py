import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def calculate_metrics(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred, average="macro")

    f1 = f1_score(y_true, y_pred, average="macro")

    return acc, recall, f1