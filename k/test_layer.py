import numpy as np
from layers import DenseLayer

layer = DenseLayer(in_features=2, out_features=3, activation="sigmoid", seed=0)

x = np.array([[1.0, 2.0]])
a = layer.forward(x)

print("W shape:", layer.W.shape)      # ожидаем (2, 3)
print("b shape:", layer.b.shape)      # ожидаем (3,)
print("a shape:", a.shape)            # ожидаем (1, 3)
print("a values:", a)                 # все между 0 и 1
print()

# backward 
dL_da = np.ones_like(a)               # градиент из единиц
dL_dx = layer.backward(dL_da)

print("dL_dW shape:", layer.dL_dW.shape)   # (2, 3)
print("dL_db shape:", layer.dL_db.shape)   # (3,)
print("dL_dx shape:", dL_dx.shape)         # (1, 2)
print("dL_dx values:", dL_dx)
print()

# update 
W_before = layer.W.copy()
layer.update(lr=0.1)
print("W changed:", not np.allclose(W_before, layer.W))  # True