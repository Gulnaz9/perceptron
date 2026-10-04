import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from layers import DenseLayer
from losses import softmax, cross_entropy_loss, cross_entropy_grad


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dL_dz):
        for l in reversed(self.layers):
            dL_dz = l.backward(dL_dz)

    def update(self, lr):
        for l in self.layers:
            l.update(lr)

    def save(self, path):
        # Сохраняем W и b каждого слоя в .npz
        arrays = {}
        for i, layer in enumerate(self.layers):
            arrays[f"W_{i}"] = layer.W
            arrays[f"b_{i}"] = layer.b
        np.savez(path, **arrays)


def compute_loss(network, X, y):
    y_pred = network.forward(X)
    return cross_entropy_loss(y_pred,y)


def compute_accuracy(network, X, y):
    y_pred = network.forward(X)
    predictions = np.argmax(y_pred, axis = 1)
    return np.mean(predictions == y)


def train_model(network, X_train, y_train, X_valid, y_valid,
                epochs=70, batch_size=8, lr=0.0314, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X_train)

    history = {
        "train_loss": [], "valid_loss": [],
        "train_acc": [], "valid_acc": [],
    }

    for epoch in range(epochs):
        # Перемешать train в начале каждой эпохи
        indices = rng.permutation(n)

        # Пройти по мини-батчам
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # 1. forward
            y_pred = network.forward(X_batch)
            # 2. градиент на выходе
            dL_dz = cross_entropy_grad(y_pred, y_batch)
            # 3. backward
            network.backward(dL_dz)
            # 4. обновить веса
            network.update(lr)

        # Логирование
        train_loss = compute_loss(network, X_train, y_train)
        train_acc = compute_accuracy(network, X_train, y_train)
        valid_loss = compute_loss(network, X_valid, y_valid)
        valid_acc = compute_accuracy(network, X_valid, y_valid)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)

        print(f"epoch {epoch+1:02d}/{epochs} - "
              f"loss: {train_loss:.4f} - val_loss: {valid_loss:.4f} - "
              f"acc: {train_acc:.4f} - val_acc: {valid_acc:.4f}")

    return history


def plot_history(history, save_path="../plots/learning_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["valid_loss"], label="valid")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["valid_acc"], label="valid")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Графики сохранены: {save_path}")


def main():
    # ---- Данные ----
    train_df = pd.read_csv("../data_training.csv")
    valid_df = pd.read_csv("../data_valid.csv")

    feature_cols = [c for c in train_df.columns if c != "diagnosis"]

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df["diagnosis"].values.astype(np.int64)

    X_valid = valid_df[feature_cols].values.astype(np.float64)
    y_valid = valid_df["diagnosis"].values.astype(np.int64)

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_valid.shape}")

    # ---- Сеть ----
    network = NeuralNetwork([
        DenseLayer(30, 24, activation="sigmoid", seed=42),
        DenseLayer(24, 24, activation="sigmoid", seed=42),
        DenseLayer(24, 24, activation="sigmoid", seed=42),
        DenseLayer(24, 2,  activation=None,      seed=42),
    ])

    # ---- Обучение ----
    history = train_model(
        network,
        X_train, y_train, X_valid, y_valid,
        epochs=70, batch_size=8, lr=0.0314, seed=42,
    )

    # ---- Сохранение модели ----
    network.save("../saved_model.npz")
    print("Модель сохранена: ../saved_model.npz")

    # ---- Графики ----
    plot_history(history)


if __name__ == "__main__":
    main()