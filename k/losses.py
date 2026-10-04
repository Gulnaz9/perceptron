import numpy as np

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    n = len(y_pred)
    n_classes = y_pred.shape[1]
    y_onehot = np.zeros((n, n_classes))
    y_onehot[np.arange(n), y_true] = 1

    y_pred_safe = np.clip(y_pred, 1e-12, 1.0)
    log_probs = np.log(y_pred_safe)
    loss = -np.sum(y_onehot * log_probs) / n
    return loss


def cross_entropy_grad(y_pred, y_true):
    n = len(y_pred)
    n_classes = y_pred.shape[1]        
    y_onehot = np.zeros((n, n_classes))
    y_onehot[np.arange(n), y_true] = 1

    return (y_pred - y_onehot) / n