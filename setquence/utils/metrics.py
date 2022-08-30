from typing import Dict

import numpy as np
from sklearn import metrics as metrics
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve
from sklearn.preprocessing import label_binarize


def classification_roc_auc(y, y_probs):
    n_classes = np.unique(y).shape[0]
    y_test = label_binarize(y, classes=list(range(n_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if n_classes > 2:
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_probs[:, i])
            roc_auc[f"roc_auc_class_{i}"] = auc(fpr[i], tpr[i])
    else:
        _fpr, _tpr, _ = roc_curve(y_test.ravel(), y_probs.ravel())
        roc_auc["roc_auc_binary"] = auc(_fpr, _tpr)

    return roc_auc


def classification_metrics(y, y_pred, y_probs) -> Dict:
    calc_averaged_metrics = {}
    calc_class_metrics = {}
    calc_metrics = {}

    n_classes = np.unique(y).shape[0]
    _average = "macro" if n_classes > 2 else "binary"

    (
        calc_averaged_metrics[f"{_average}_precision"],
        calc_averaged_metrics[f"{_average}_recall"],
        calc_averaged_metrics[f"{_average}_f-score"],
        _,
    ) = precision_recall_fscore_support(y, y_pred, average=_average)

    calc_averaged_metrics["accuracy"] = metrics.accuracy_score(y, y_pred)

    (calc_metrics["precision"], calc_metrics["recall"], calc_metrics["f-score"], _,) = precision_recall_fscore_support(
        y, y_pred, average=None
    )

    for i in range(n_classes):
        for k, v in calc_metrics.items():
            calc_class_metrics[f"{k}_class_{i}"] = v[i]

    calc_class_metrics.update(calc_averaged_metrics)

    if n_classes <= 2:
        calc_class_metrics.update(classification_roc_auc(y, y_probs))

    return calc_class_metrics
