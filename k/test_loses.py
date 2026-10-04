import numpy as np
from losses import softmax, cross_entropy_loss, cross_entropy_grad

z = np.array([[2.0, 1.0]])
p = softmax(z)
print("softmax([2, 1]) =", p)
print("Ожидаем ≈ [0.731, 0.269]")
print("Сумма:", p.sum())
print()

z_big = np.array([[1000.0, 999.0]])
p_big = softmax(z_big)
print("softmax([1000, 999]) =", p_big)
print("Ожидаем ≈ [0.731, 0.269]  (без NaN!)")
print()

y_pred = np.array([[0.99, 0.01]])   # сеть думает класс 0
y_true = np.array([0])              # истина — класс 0
loss = cross_entropy_loss(y_pred, y_true)
print("loss (уверенно правильно) =", loss)
print("Ожидаем ≈ 0.01 (маленький штраф)")
print()

y_pred = np.array([[0.01, 0.99]])   # сеть думает класс 1
y_true = np.array([0])              # а истина — класс 0
loss = cross_entropy_loss(y_pred, y_true)
print("loss (уверенно неправильно) =", loss)
print("Ожидаем ≈ 4.6 (большой штраф)")
print()

y_pred = np.array([[0.7, 0.3]])
y_true = np.array([1])              # истина — класс 1
grad = cross_entropy_grad(y_pred, y_true)
print("grad =", grad)
print("Ожидаем: [0.7, -0.7]  (ŷ - y_onehot)")