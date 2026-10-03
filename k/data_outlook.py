import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data.csv", header=None)

columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
df.columns = columns

print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

print("\nБаланс классов:")
print(df["diagnosis"].value_counts())

print("\nПропуски:")
print(df.isnull().sum().sum())

print("\nСтатистика признаков:")
print(df.describe().T)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------

os.makedirs("../plots", exist_ok=True)

fig, ax = plt.subplots(figsize=(5, 4))
counts = df["diagnosis"].value_counts()
ax.bar(counts.index, counts.values, color=["#4CAF50", "#F44336"])
ax.set_title("Баланс классов")
ax.set_ylabel("Количество")
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha="center")
plt.tight_layout()
plt.savefig("../plots/class_balance.png", dpi=120)
plt.close()

# Распределения первых 6 признаков 
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, ax in enumerate(axes.flat):
    col = f"feature_{i+1}"
    ax.hist(df[df.diagnosis == "B"][col], bins=30, alpha=0.6,
            label="B (benign)", color="#4CAF50")
    ax.hist(df[df.diagnosis == "M"][col], bins=30, alpha=0.6,
            label="M (malignant)", color="#F44336")
    ax.set_title(col)
    ax.legend()
plt.suptitle("Распределения первых 6 признаков по классам", fontsize=14)
plt.tight_layout()
plt.savefig("../plots/features_hist.png", dpi=120)
plt.close()

# Boxplot: как признак разделяет классы 
features_to_show = ["feature_1", "feature_3", "feature_4", "feature_24",
                    "feature_26", "feature_27"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, ax in enumerate(axes.flat):
    col = features_to_show[i]
    data = [df[df.diagnosis == "B"][col], df[df.diagnosis == "M"][col]]
    ax.boxplot(data, tick_labels=["B", "M"])
    ax.set_title(col)
plt.suptitle("Boxplot: разделимость классов по признакам", fontsize=14)
plt.tight_layout()
plt.savefig("../plots/features_boxplot.png", dpi=120)
plt.close()

# Heatmap корреляций 
numeric_df = df.drop(columns=["id", "diagnosis"])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticklabels(corr.columns, fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Корреляционная матрица признаков")
plt.tight_layout()
plt.savefig("../plots/correlation_heatmap.png", dpi=120)
plt.close()
