#!/usr/bin/env python3
"""
Transferencia entre datasets: treinar num, testar no outro.

    python3 transfer_test.py --csv .../ProcessedECGData_v5.csv \
                             --series .../NNSeries_v5.npz

Porque interessa. O tamanho do efeito das features e de sinal oposto nos dois
datasets: no WESAD (TSST) a frequencia sobe 25,7 bpm na condicao de stress
(d de Cohen +1,73 em hr_mean); no SWELL (email/pressao de tempo, com a condicao
neutra sempre no primeiro bloco) desce 4,6 bpm (d -0,42).

Se os rotulos forem mesmo incompativeis, um modelo treinado num dataset deve
acertar ABAIXO do acaso no outro -- AUC < 0,5, ou seja sistematicamente errado
e nao aleatoriamente errado. Uma AUC de 0,5 significaria apenas ausencia de
sinal transferivel; abaixo de 0,5 prova que o sinal existe e aponta ao contrario.

Nota: com AUC < 0,5 a exatidao nao e a metrica informativa, porque depende do
balanco de classes de cada dataset (WESAD 35,6% stress, SWELL 60,7%).
"""
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")

import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from train_sequence import FEATURES, build

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--series", required=True)
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--out", default="results_transfer.csv")
a = p.parse_args()

raw = pd.read_csv(a.csv)
z = np.load(a.series)
M, Lg = z["nni"], z["length"]
keep = raw[FEATURES + ["label"]].notna().all(axis=1).to_numpy()
raw, M = raw[keep].reset_index(drop=True), M[keep]
is_w = raw["file"].str.contains("ECG_values").to_numpy()
Xf = raw[FEATURES].to_numpy(float)
y = raw["label"].to_numpy().astype(int)

print(f"WESAD {is_w.sum()} janelas ({y[is_w].mean()*100:.1f}% stress) | "
      f"SWELL {(~is_w).sum()} janelas ({y[~is_w].mean()*100:.1f}% stress)\n")

res = []
for tr_name, tr_mask in (("WESAD", is_w), ("SWELL", ~is_w)):
    te_name = "SWELL" if tr_name == "WESAD" else "WESAD"
    te_mask = ~tr_mask
    print(f"{'='*62}\nTREINO em {tr_name}  ->  TESTE em {te_name}\n{'='*62}")

    # Normalizacao com estatisticas SO do dataset de treino: e o cenario
    # realista de aplicar um modelo a dados novos de outra origem.
    fmu, fsd = Xf[tr_mask].mean(0), Xf[tr_mask].std(0)
    Ftr = np.clip((Xf[tr_mask] - fmu) / fsd, -5, 5)
    Fte = np.clip((Xf[te_mask] - fmu) / fsd, -5, 5)
    real = M[tr_mask] > 0
    mu, sd = M[tr_mask][real].mean(), M[tr_mask][real].std()
    prep = lambda m: np.where(m > 0, (m - mu) / sd, 0.0)[..., None]
    mask = lambda m: (m > 0).astype("float32")[..., None]
    Str, Ste = prep(M[tr_mask]), prep(M[te_mask])
    Ktr, Kte = mask(M[tr_mask]), mask(M[te_mask])
    ytr, yte = y[tr_mask], y[te_mask]

    for nm, mk in (("LogReg", lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
                   ("RandomForest", lambda: RandomForestClassifier(
                       n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1))):
        m = mk().fit(Ftr, ytr)
        pr = m.predict_proba(Fte)[:, 1]
        auc = roc_auc_score(yte, pr)
        res.append(dict(train=tr_name, test=te_name, model=nm, seed=0,
                        acc=accuracy_score(yte, m.predict(Fte)) * 100, auc=auc))
        print(f"  {nm:<22} AUC {auc:.3f}  {'<- ABAIXO DO ACASO' if auc < 0.5 else ''}")

    for arch, nm in (("summary", "MLP resumo"), ("seq", "Conv1D so serie"),
                     ("both", "Conv1D serie+resumo")):
        aucs = []
        for seed in range(42, 42 + a.seeds):
            tf.keras.backend.clear_session()
            tf.random.set_seed(seed)
            np.random.seed(seed)
            xtr = {"seq": [Str, Ktr], "summary": [Ftr], "both": [Str, Ktr, Ftr]}[arch]
            xte = {"seq": [Ste, Kte], "summary": [Fte], "both": [Ste, Kte, Fte]}[arch]
            model = build(arch, M.shape[1], len(FEATURES))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
                          loss="binary_crossentropy", metrics=["accuracy"])
            cw = dict(enumerate(compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)))
            cw[1] *= 1.2
            model.fit(xtr, ytr, validation_split=0.2, epochs=a.epochs, batch_size=64,
                      class_weight=cw, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(
                          monitor="val_loss", patience=20, restore_best_weights=True)])
            prob = model.predict(xte, verbose=0).flatten()
            auc = roc_auc_score(yte, prob)
            aucs.append(auc)
            res.append(dict(train=tr_name, test=te_name, model=nm, seed=seed,
                            acc=accuracy_score(yte, (prob > 0.5).astype(int)) * 100, auc=auc))
        aucs = np.array(aucs)
        print(f"  {nm:<22} AUC {aucs.mean():.3f} +- {aucs.std(ddof=1) if len(aucs)>1 else 0:.3f}"
              f"  {'<- ABAIXO DO ACASO' if aucs.mean() < 0.5 else ''}")
    print()

pd.DataFrame(res).to_csv(a.out, index=False)
print(f"Gravado em {a.out}")
print("\nLeitura: AUC = 0,5 significa nenhum sinal transferivel. AUC < 0,5 significa")
print("sinal transferivel mas INVERTIDO -- os rotulos dos dois datasets opoem-se.")
