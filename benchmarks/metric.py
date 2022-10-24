import numpy as np
from sklearn.metrics import classification_report, top_k_accuracy_score


def scoring_mnist(y_true, y_pred, zero_division=0):
    y_pred = np.argmax(y_pred, axis=-1)

    report = classification_report(
        y_true, y_pred, zero_division=zero_division, output_dict=True
    )

    metrics = {}
    metrics["accuracy"] = report["accuracy"]
    metrics["macro avg precision"] = report["macro avg"]["precision"]
    metrics["macro avg recall"] = report["macro avg"]["recall"]
    metrics["macro avg f1-score"] = report["macro avg"]["f1-score"]
    metrics["weighted avg precision"] = report["weighted avg"]["precision"]
    metrics["weighted avg recall"] = report["weighted avg"]["recall"]
    metrics["weighted avg f1-score"] = report["weighted avg"]["f1-score"]
    metrics["support"] = report["macro avg"]["support"]

    return metrics


def scoring_imagenet(y_true, y_pred):

    labels = len(y_pred[0])

    metrics = {}
    metrics["top1_accuracy"] = top_k_accuracy_score(
        y_true, y_pred, k=1, labels=range(labels)
    )
    metrics["top5_accuracy"] = top_k_accuracy_score(
        y_true, y_pred, k=5, labels=range(labels)
    )

    return metrics


################################################################


metrics = {"mnist": scoring_mnist, "imagenet": scoring_imagenet}


def scoring_metric(dataset, y_true, y_pred, *args, **kwargs):
    return metrics[dataset](y_true, y_pred, *args, **kwargs)
