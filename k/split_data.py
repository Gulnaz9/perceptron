import numpy as np
import pandas as pd
import os


df = pd.read_csv("../data.csv", header=None)
columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
df.columns = columns

df = df.drop(columns=["id"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

assert df["diagnosis"].notna().all(), "Ошибка: неизвестные значения в diagnosis"

rng = np.random.default_rng(42)
indices = rng.permutation(len(df))
n_valid = int(len(df) * 0.2)

valid_idx = indices[:n_valid]
train_idx = indices[n_valid:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_valid = df.iloc[valid_idx].reset_index(drop=True)

print(f"Train: {df_train.shape}")
print(f"Valid: {df_valid.shape}")

# Нормализация 
feature_cols = [c for c in df.columns if c != "diagnosis"]

mean = df_train[feature_cols].mean()
std = df_train[feature_cols].std()


std = std.replace(0, 1.0)

df_train[feature_cols] = (df_train[feature_cols] - mean) / std
df_valid[feature_cols] = (df_valid[feature_cols] - mean) / std


df_train.to_csv("../data_training.csv", index=False)
df_valid.to_csv("../data_valid.csv", index=False)


norm_df = pd.DataFrame({
    "feature": feature_cols,
    "mean": mean.values,
    "std": std.values,
})
norm_df.to_csv("../normalization_params.csv", index=False)


print("\nСредние по train после нормализации (должны быть ~0):")
print(df_train[feature_cols].mean().round(3).to_string())

print("\nСтд по train после нормализации (должны быть ~1):")
print(df_train[feature_cols].std().round(3).to_string())

print("\nБаланс классов в train:")
print(df_train["diagnosis"].value_counts().to_string())

print("\nБаланс классов в valid:")
print(df_valid["diagnosis"].value_counts().to_string())