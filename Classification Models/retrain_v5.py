#!/usr/bin/env python3
"""
Treina o Model D (CNN-Transformer) num CSV a escolha, sem tocar no main.py.

    python3 retrain_v5.py /home/eduarda/Desktop/ProcessedECGData_v5.csv [--seeds 5]

Usa exactamente o mesmo pipeline do main.py -- split estratificado 80/20,
normalize_data(), reshape para (N, 8, 1), train_evaluate_model() -- para que o
resultado seja comparavel com a V4. A unica diferenca e correr varias seeds,
que e o que a supervisora pediu no ponto 10 (sem repeticoes, a diferenca entre
59,70% e 60,68% nao significa nada).

Seleciona as features por NOME e nao por posicao: o iloc[:, 2:10] do Data.py
parte-se em qualquer CSV que tenha a coluna sdnn_index.
"""
import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# TEM de vir antes de importar Models: Models.py:176 termina plot_history() com
# plt.show(), que bloqueia a espera que a janela seja fechada. Com backend Agg
# o plt.show() e um no-op e a execucao corre sem supervisao. Sem isto o processo
# para em cada seed ate alguem fechar a figura a mao.
import matplotlib

matplotlib.use("Agg")

import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split

from Data import normalize_data
from Models import build_model, train_evaluate_model

# F2 = 8 features do dominio temporal (o que a tese usa nos resultados finais).
# F1 = as mesmas mais as duas de frequencia. A Contribuicao 1 da tese assenta
# em F2 (60,68%) ser melhor que F1 (60,00%) -- 0,68 pp, uma seed, dados sujos.
FEAT_F2 = ["hr_mean", "nni_mean", "nni_min", "nni_max",
           "sdnn", "rmssd", "nn50", "pnn50"]
FEAT_F1 = FEAT_F2 + ["LF/HF ratio", "Total power"]

# Mapeamento letra -> (funcao em Models.py, clipnorm, epocas), com os
# hiperparametros por arquitetura que a Seccao 3.6 da tese documenta:
# clipnorm=0.5 e 100 epocas para as recorrentes/convolucionais,
# clipnorm=1.0 e 150 epocas para o Transformer.
ARCH = {
    "A": ("model2_simple", 0.5, 100, "LSTM"),
    "B": ("model2_new", 0.5, 100, "Deep LSTM"),
    "C": ("model4_cnn_lstm_hybrid", 0.5, 100, "CNN-LSTM"),
    "D": ("model5_cnn_transformer_hybrid_v2", 1.0, 150, "CNN-Transformer"),
}

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--models", default="A,B,C,D",
               help="letras separadas por virgula (default A,B,C,D)")
p.add_argument("--out", default="results_v5.csv",
               help="CSV onde acumular os resultados por seed")
p.add_argument("--features", default="F2", choices=["F1", "F2"],
               help="F2 = 8 temporais (default). F1 = as 10, com frequencia.")
p.add_argument("--cv", default="random", choices=["random", "group", "loso"],
               help="random = split 80/20 estratificado (o da tese). "
                    "group = GroupKFold k=5 por sujeito.")
p.add_argument("--dataset", default="all", choices=["all", "SWELL", "WESAD"],
               help="restringir a um dataset. Os dois tem protocolos de inducao "
                    "muito diferentes: no WESAD (TSST) a frequencia sobe 25,7 bpm "
                    "na condicao de stress (d de Cohen medio 1,13); no SWELL "
                    "(email/pressao de tempo) DESCE 4,6 bpm e os sinais das "
                    "features invertem-se (d medio 0,37). Treinar nos dois "
                    "juntos e pedir ao modelo que separe rotulos incompativeis.")
a = p.parse_args()

FEATURES = FEAT_F1 if a.features == "F1" else FEAT_F2
d = pd.read_csv(a.csv).dropna(subset=FEATURES + ["label"])
if a.dataset != "all":
    _w = d["file"].str.contains("ECG_values")
    d = d[_w if a.dataset == "WESAD" else ~_w].reset_index(drop=True)
X = d[FEATURES].to_numpy(dtype=float)
y = d["label"].to_numpy().astype(int)

# Os prefixos S1..S25 COLIDEM entre WESAD e SWELL: o id de sujeito tem de
# incluir o dataset, senao juntam-se pessoas diferentes no mesmo grupo.
# pd.Series e nao np.array porque arrays numpy de strings nao concatenam com '+'.
_ds = pd.Series(np.where(d["file"].str.contains("ECG_values"), "WESAD", "SWELL"),
                index=d.index)
groups = (_ds + "_" + d["file"].str.extract(r"^(S\d+)")[0]).to_numpy()

print(f"{a.csv}\n  {len(d)} janelas, {y.mean() * 100:.2f}% stress, "
      f"{len(FEATURES)} features ({a.features})")
print(f"  modelos: {a.models}   cv: {a.cv}"
      + (f"   ({len(np.unique(groups))} sujeitos)" if a.cv in ("group","loso")
         else f"   seeds: {a.seeds}") + "\n")

wanted = [m.strip().upper() for m in a.models.split(",")]
bad = [m for m in wanted if m not in ARCH]
if bad:
    sys.exit(f"ERRO: modelo(s) desconhecido(s) {bad}. Validos: {list(ARCH)}")

res = []
for letter in wanted:
    # NAO chamar esta variavel 'fn': mais abaixo o unpack da matriz de confusao
    # usa tn/fp/fn/tp e sobrescrevia-a, partindo a 2a seed de cada arquitetura.
    arch_fn, clipnorm, epochs, nice = ARCH[letter]

    if a.cv == "loso":
        # Um fold por sujeito. No WESAD sao 15 folds em vez de 5, o que triplica
        # os pontos de teste e aperta bastante o intervalo de confianca da AUC
        # (com GroupKFold k=5 o SD era 0,087).
        splits = list(LeaveOneGroupOut().split(X, y, groups))
    elif a.cv == "group":
        splits = list(GroupKFold(n_splits=5).split(X, y, groups))
    else:
        splits = [train_test_split(np.arange(len(y)), test_size=0.2,
                                   random_state=s, stratify=y)
                  for s in range(42, 42 + a.seeds)]

    for k, (itr, ite) in enumerate(splits):
        seed = 42 + k
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Em GroupKFold os indices saem ordenados por ficheiro, e o
        # validation_split do Keras corta os ultimos 20% SEM baralhar -- daria
        # um conjunto de validacao de um so sujeito. Baralhar resolve; so
        # afecta o early stopping, o teste continua disjunto por sujeito.
        itr = np.random.permutation(itr)
        xtr, xte, ytr, yte = X[itr], X[ite], y[itr], y[ite]
        xtr, xte = normalize_data(xtr, xte)
        xtr = xtr.reshape(len(xtr), xtr.shape[1], 1)
        xte = xte.reshape(len(xte), xte.shape[1], 1)

        model = build_model(arch_fn, (xtr.shape[1], 1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=clipnorm),
            loss="binary_crossentropy",
            metrics=["accuracy",
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")])

        lab = f"fold {k+1}" if a.cv in ("group","loso") else f"seed {seed}"
        print(f"--- Model {letter} ({nice})  {lab} " + "-" * 30)
        t0 = time.time()
        train_evaluate_model(model, 32, epochs, xtr, xte, ytr, yte)
        mins = (time.time() - t0) / 60

        prob = model.predict(xte, verbose=0).flatten()
        pred = (prob > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        r = dict(model=letter, arch=nice, seed=seed, cv=a.cv, features=a.features,
                 acc=accuracy_score(yte, pred) * 100,
                 prec=precision_score(yte, pred, zero_division=0) * 100,
                 rec=recall_score(yte, pred) * 100,
                 spec=tn / (tn + fp) * 100,
                 f1=f1_score(yte, pred),
                 auc=roc_auc_score(yte, prob),
                 prauc=average_precision_score(yte, prob),
                 params=model.count_params(),
                 minutes=mins,
                 major=max(yte.mean(), 1 - yte.mean()) * 100)
        res.append(r)
        pd.DataFrame(res).to_csv(a.out, index=False)  # grava a cada seed
        print(f"    MODEL {letter} {lab}: acc {r['acc']:.2f}  prec {r['prec']:.2f}  "
              f"rec {r['rec']:.2f}  spec {r['spec']:.2f}  AUC {r['auc']:.3f}  "
              f"({mins:.1f} min)\n")

df = pd.DataFrame(res)
print("=" * 78)
n = len(splits) if a.cv in ("group","loso") else a.seeds
print(f"TABELA 7 (V5, {a.features}, cv={a.cv})  —  {n} corridas, media +/- SD")
print("=" * 78)
print(f"  {'':<18}{'acc':>14}{'prec':>14}{'recall':>14}{'spec':>14}{'AUC':>13}")
for letter in wanted:
    g = df[df.model == letter]
    name = f"Model {letter} ({ARCH[letter][3]})"
    print(f"  {name:<18}"
          + "".join(f"{g[k].mean():8.2f}+-{g[k].std():<5.2f}"
                    for k in ("acc", "prec", "rec", "spec"))
          + f"{g['auc'].mean():7.3f}+-{g['auc'].std():<5.3f}")
print(f"\n  {'':<18}{'params':>12}{'min/treino':>14}")
for letter in wanted:
    g = df[df.model == letter]
    print(f"  Model {letter:<17}{int(g['params'].iloc[0]):>12,}{g['minutes'].mean():>14.2f}")
print(f"\n  majority classifier: {df['major'].mean():.2f}%")
print(f"\n  Melhor exatidao: Model {df.groupby('model')['acc'].mean().idxmax()}"
      f"   Melhor recall: Model {df.groupby('model')['rec'].mean().idxmax()}"
      f"   Melhor AUC: Model {df.groupby('model')['auc'].mean().idxmax()}")
print(f"\n  V4 reportava (1 seed, dados sujos): A 60.00 | B 60.00 | C 59.70 | D 60.68")
print(f"  Resultados por seed gravados em {a.out}")
