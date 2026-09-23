#!/usr/bin/env python3
"""Atribuicao media por classe, sobre TODAS as janelas de teste do WESAD.

A figura antiga (`gradient_by_class.png`, do compare_attention_patterns) compara
UMA amostra de stress com UMA de nao-stress -- o que ja esta nos estudos de caso
e nao suporta nenhuma afirmacao sobre a classe. Esta produz a media de
|d(y)/d(x)| sobre as janelas de cada classe real, com o erro padrao.

    PYTHONPATH=. python3 gradient_class_means.py
"""
import os, warnings
warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split

from Data import normalize_data
from Models import PositionalEncoding

FEATURES = ["hr_mean", "nni_mean", "nni_min", "nni_max",
            "sdnn", "rmssd", "nn50", "pnn50"]
CSV = "/home/eduarda/Desktop/ProcessedECGData_v5.csv"

d = pd.read_csv(CSV).dropna(subset=FEATURES + ["label"])
d = d[d["file"].str.contains("ECG_values")].reset_index(drop=True)
X, y = d[FEATURES].to_numpy(float), d["label"].to_numpy().astype(int)
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
_, xten = normalize_data(xtr, xte)
xte_r = xten.reshape(len(xte), -1, 1)

model = tf.keras.models.load_model(
    "saved_models/wesad_model.h5", compile=False,
    custom_objects={"PositionalEncoding": PositionalEncoding})

# Gradiente por amostra, em lote unico: a rede e pequena e o teste tem 189 janelas.
x = tf.constant(xte_r, dtype=tf.float32)
with tf.GradientTape() as t:
    t.watch(x)
    out = model(x)
g = np.abs(t.gradient(out, x).numpy()[:, :, 0])     # (N, 8)

# Escala comum aos dois perfis, para as alturas serem comparaveis entre classes.
scale = g.mean(axis=0).max()
means = {c: g[yte == c].mean(axis=0) / scale for c in (1, 0)}
sems = {c: g[yte == c].std(axis=0, ddof=1) / np.sqrt((yte == c).sum()) / scale
        for c in (1, 0)}

print(f"N teste {len(yte)}  ({(yte==1).sum()} stress, {(yte==0).sum()} nao-stress)")
for c, lab in ((1, "stress"), (0, "nao-stress")):
    print(f"\n{lab}:")
    for j in np.argsort(-means[c]):
        print(f"  {FEATURES[j]:9s} {means[c][j]:.3f} +/- {sems[c][j]:.3f}")

fig, ax = plt.subplots(figsize=(12, 6))
xp, w = np.arange(len(FEATURES)), 0.35
ax.bar(xp - w/2, means[1], w, yerr=sems[1], capsize=3, label="Stress windows",
       color="#e74c3c", alpha=0.85)
ax.bar(xp + w/2, means[0], w, yerr=sems[0], capsize=3, label="No-stress windows",
       color="#3498db", alpha=0.85)
ax.set_xlabel("HRV Features", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean Gradient Magnitude (Normalised)", fontsize=12, fontweight="bold")
ax.set_title("Gradient Attribution by Class (WESAD test partition, N=189)",
             fontsize=14, fontweight="bold")
ax.set_xticks(xp); ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out_path = "results/gradient_by_class_mean.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"\nGuardado em {out_path}")
