import numpy as np
import pandas as pd

from layers import DenseLayer
from losses import cross_entropy_loss
from train import NeuralNetwork


def build_network():
    return NeuralNetwork([
        DenseLayer(30, 24, activation="sigmoid", seed=42),
        DenseLayer(24, 24, activation="sigmoid", seed=42),
        DenseLayer(24, 24, activation="sigmoid", seed=42),
        DenseLayer(24, 2,  activation=None,      seed=42),
    ])


def confusion_matrix(y_true, y_pred_labels, n_classes=2):
    """Возвращает матрицу ошибок: cm[i, j] = сколько раз истинный класс i предсказан как j"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred_labels):
        cm[t, p] += 1
    return cm


def main():
    
    valid_df = pd.read_csv("../data_valid.csv")
    feature_cols = [c for c in valid_df.columns if c != "diagnosis"]

    X_valid = valid_df[feature_cols].values.astype(np.float64)
    y_valid = valid_df["diagnosis"].values.astype(np.int64)

    print(f"Валидационная выборка: {X_valid.shape}")
    print(f"Баланс классов: 0 -> {(y_valid == 0).sum()}, 1 -> {(y_valid == 1).sum()}")
    print()

    # Создаём сеть и загружаем веса
    network = build_network()
    network.load("../saved_model.npz")
    print("Модель загружена: ../saved_model.npz")
    print()

    # Forward на валидации
    y_pred_proba = network.forward(X_valid)         
    y_pred_labels = np.argmax(y_pred_proba, axis=1) 

    # Метрики
    loss = cross_entropy_loss(y_pred_proba, y_valid)
    accuracy = np.mean(y_pred_labels == y_valid)

    print(f"Бинарная кросс-энтропия: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({int(accuracy * len(y_valid))}/{len(y_valid)})")
    print()

    # Confusion matrix
    cm = confusion_matrix(y_valid, y_pred_labels)
    print("Confusion matrix (строки — истина, столбцы — предсказание):")
    print(f"                 предсказано 0   предсказано 1")
    print(f"истинно 0 (B)    {cm[0,0]:>11d}   {cm[0,1]:>11d}")
    print(f"истинно 1 (M)    {cm[1,0]:>11d}   {cm[1,1]:>11d}")
    print()

    # Дополнительные метрики 
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Precision (класс 1, M): {precision:.4f}")
    print(f"Recall    (класс 1, M): {recall:.4f}")
    print(f"F1-score  (класс 1, M): {f1:.4f}")


if __name__ == "__main__":
    main()