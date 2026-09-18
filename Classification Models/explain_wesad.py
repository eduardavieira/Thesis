#!/usr/bin/env python3
"""
Treina o Model D no WESAD (dados V5) e corre a explicabilidade sobre ele.

    python3 explain_wesad.py [--dataset WESAD] [--seed 42]

Porque so no WESAD. No conjunto combinado as atribuicoes NUNCA poderiam
confirmar a fisiologia: o SWELL tem os sinais de todas as features invertidos
(hr_mean d=-0,42 contra +1,73 no WESAD), porque a sua condicao neutra e sempre
o primeiro bloco da sessao. Era essa a raiz do ponto 5 da supervisora -- os
exemplos contrariavam a fisiologia porque o alvo misturava dois estados opostos.

No WESAD (TSST, +25,7 bpm, d medio 1,13) as atribuicoes devem alinhar com a
Tabela 6 da tese.

Define model / x_test_reshape / y_test / numeric_features em memoria e depois
executa run_explainability.py, que deteta o modelo em memoria e salta o
carregamento do .h5 da V4.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ANTES de importar Models: Models.py:176 termina plot_history() com plt.show(),
# que bloqueia a espera que a janela seja fechada.
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
from sklearn.model_selection import train_test_split

from Data import normalize_data
from Models import build_model, train_evaluate_model

p = argparse.ArgumentParser()
p.add_argument("--csv", default="/home/eduarda/Desktop/ProcessedECGData_v5.csv")
p.add_argument("--dataset", default="WESAD", choices=["WESAD", "SWELL", "all"])
p.add_argument("--seed", type=int, default=42)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--save", default="saved_models/wesad_model.h5")
p.add_argument("--load", action="store_true",
               help="carregar o modelo ja guardado em --save em vez de treinar. "
                    "Usar para regerar figuras (~2 min em vez de ~15) depois de "
                    "mexer em rotulos ou no codigo de explicabilidade.")
a = p.parse_args()

# Seleccao por NOME, nao por posicao: o iloc[:, 2:10] do Data.py parte-se em
# qualquer CSV que tenha a coluna sdnn_index.
numeric_features = ["hr_mean", "nni_mean", "nni_min", "nni_max",
                    "sdnn", "rmssd", "nn50", "pnn50"]

d = pd.read_csv(a.csv).dropna(subset=numeric_features + ["label"])
if a.dataset != "all":
    w = d["file"].str.contains("ECG_values")
    d = d[w if a.dataset == "WESAD" else ~w].reset_index(drop=True)

X = d[numeric_features].to_numpy(float)
y = d["label"].to_numpy().astype(int)
print(f"{a.dataset}: {len(d)} janelas, {y.mean()*100:.2f}% stress, "
      f"{d['file'].str.extract(r'^(S[0-9]+)')[0].nunique()} sujeitos\n")

tf.keras.backend.clear_session()
tf.random.set_seed(a.seed)
np.random.seed(a.seed)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=a.seed, stratify=y)
x_train_norm, x_test_norm = normalize_data(x_train, x_test)
x_train_reshape = x_train_norm.reshape(len(x_train), -1, 1)
x_test_reshape = x_test_norm.reshape(len(x_test), -1, 1)

if a.load:
    # O split e deterministico (mesma seed, mesmo CSV, mesmo filtro), por isso
    # x_test_reshape/y_test correspondem exactamente aos do treino original.
    from tensorflow.keras.models import load_model
    from Models import PositionalEncoding
    print(f"A carregar {a.save} (sem treinar)...")
    model = load_model(a.save, custom_objects={"PositionalEncoding": PositionalEncoding})
else:
    model = build_model("model5_cnn_transformer_hybrid_v2", (x_train_reshape.shape[1], 1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
                  loss="binary_crossentropy",
                  metrics=["accuracy",
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    train_evaluate_model(model, 32, a.epochs, x_train_reshape, x_test_reshape, y_train, y_test)
    os.makedirs("saved_models", exist_ok=True)
    model.save(a.save)
    print(f"\nModelo guardado em {a.save}")

from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
prob = model.predict(x_test_reshape, verbose=0).flatten()
pred = (prob > 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"\n{'='*58}\nMATRIZ DE CONFUSAO (para a Tabela 8)\n{'='*58}")
print(f"                     Prev: nao-stress   Prev: stress")
print(f"  Real: nao-stress   {tn:14d} {fp:14d}")
print(f"  Real: stress       {fn:14d} {tp:14d}")
print(f"\n  N={len(y_test)}  acc {accuracy_score(y_test,pred)*100:.2f}  "
      f"prec {precision_score(y_test,pred,zero_division=0)*100:.2f}  "
      f"rec {recall_score(y_test,pred)*100:.2f}  "
      f"spec {tn/(tn+fp)*100:.2f}  AUC {roc_auc_score(y_test,prob):.3f}")

print(f"\n{'='*58}\nEXPLICABILIDADE\n{'='*58}")
# run_explainability.py testa se `model`, `x_test_reshape` e `y_test` existem
# no scope; existindo, salta o carregamento do .h5 da V4.
exec(open("run_explainability.py").read())
