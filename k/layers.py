import numpy as np

class DenseLayer:
    def __init__(self, in_features, out_features, activation=None, seed=None):
        """
        in_features:  сколько чисел приходит на вход слоя
        out_features: сколько нейронов в слое
        activation:   "sigmoid" или None
        seed:         для воспроизводимости
        """
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        rng = np.random.default_rng(seed)

        # He-инициализация: W ~ Uniform(-limit, +limit), limit = sqrt(6 / in_features)
        limit = np.sqrt(6.0 / in_features)
        self.W = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros(out_features)

        # Сюда будем складывать промежуточные значения в forward / backward
        self.x = None       # вход слоя
        self.a = None       # выход слоя 
        self.dL_dW = None   # градиент по W
        self.dL_db = None   # градиент по b

    def forward(self, x):
        self.x = x
        z = x @ self.W + self.b
        if self.activation == "sigmoid":
            self.a = 1 / (1 + np.exp(-z))
        else:
            self.a = z
        return self.a

    def backward(self, dL_da):
        if self.activation == "sigmoid":
            dL_dz = dL_da * self.a * (1 - self.a)
        else: 
            dL_dz = dL_da
        self.dL_dW = self.x.T @ dL_dz
        self.dL_db = dL_dz.sum(axis=0)
        dL_dx = dL_dz @ self.W.T
        return dL_dx

    def update(self, lr):
        self.W -= lr * self.dL_dW
        self.b -= lr * self.dL_db