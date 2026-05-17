import numpy as np
import torch

def accuracy(ypred, ytrue):
    return np.mean(ytrue == ypred)


def cross_entropy_eval(model, X, y_true):
    A_out, _ = model.forward(X)
    return model.cross_entropy(A_out, y_true)

def confusion_matrix(model, X, y_true):
    y_pred = model.predict(X)
    n_classes = len(np.unique(y_true))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm

def f1_macro(model, X, y_true):
    y_pred = model.predict(X)
    n_classes = len(np.unique(y_true))
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)
    return np.mean(f1s)

def evaluate_numpy_model(model, X, y):
    probs, _ = model.forward(X)
    y_pred   = probs.argmax(axis=1)
    return {
        "acc":  accuracy(y_pred, y),
        "ce":   cross_entropy_eval(model, X, y),
        "f1":   f1_macro(model, X, y),
        "cm":   confusion_matrix(model, X, y),
        "pred": y_pred,
    }

def f1_macro_from_preds(y_true, y_pred, n_classes=49):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1s.append(2 * precision * recall / (precision + recall + 1e-8))
    return np.mean(f1s)

def confusion_matrix_from_preds(y_true, y_pred, n_classes=49):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def evaluate_torch_model(model, loader, device, n_classes=49):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            total_loss += criterion(logits, y_b).item() * len(y_b)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(y_b.cpu().numpy())
            n += len(y_b)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return {
        "acc":  accuracy(y_pred, y_true),
        "ce":   total_loss / n,
        "f1":   f1_macro_from_preds(y_true, y_pred, n_classes),
        "cm":   confusion_matrix_from_preds(y_true, y_pred, n_classes),
        "pred": y_pred,
        "true": y_true,
    }