#!/usr/bin/env python3
"""Baseline so com frequencia cardiaca no WESAD, nos tres protocolos.

Pergunta (3.a avaliacao externa, 2026-09-29): quanto do desempenho em sujeitos
novos vem so da frequencia cardiaca? Mesmo protocolo do wesad_baselines.py
(split estratificado 80/20 com seeds 42-46, GroupKFold k=5, LOSO; StandardScaler
ajustado so no treino; LogReg sem class_weight, como na tese).

Entradas comparadas:
  hr_mean      so a FC media da janela
  n_beats      so o numero de intervalos da janela (o comprimento da mascara do G)
  hr+nbeats    as duas
  8 features   as 8 do dominio temporal (= linha "Logistic regression" da tese)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["hr_mean", "nni_mean", "nni_min", "nni_max", "sdnn", "rmssd", "nn50", "pnn50"]
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="/home/eduarda/Desktop/ProcessedECGData_v6_wesad.csv")
ap.add_argument("--series", default="/home/eduarda/Desktop/NNSeries_v6_wesad.npz")
ap.add_argument("--out", default="results_wesad_v6_hr_only.csv")
args = ap.parse_args()

d = pd.read_csv(args.csv)
z = np.load(args.series)
assert len(d) == len(z["length"]), "CSV e series com numero de janelas diferente"
d["n_beats"] = z["length"]
d = d.dropna(subset=FEATURES + ["label"])
d = d[d["file"].str.contains("ECG_values")].reset_index(drop=True)   # WESAD
y = d["label"].to_numpy().astype(int)
groups = d["file"].str.extract(r"^(S\d+)")[0].to_numpy()
print(f"WESAD: {len(d)} janelas, {y.mean()*100:.2f}% stress, {len(np.unique(groups))} sujeitos")

INPUTS = {"hr_mean": ["hr_mean"], "n_beats": ["n_beats"],
          "hr+nbeats": ["hr_mean", "n_beats"], "8 features": FEATURES}

rows = []
for cv in ("random", "group", "loso"):
    if cv == "random":
        splits = [train_test_split(np.arange(len(y)), test_size=0.2, random_state=s, stratify=y)
                  for s in range(42, 47)]
    elif cv == "group":
        splits = list(GroupKFold(n_splits=5).split(d, y, groups))
    else:
        splits = list(LeaveOneGroupOut().split(d, y, groups))
    for k, (itr, ite) in enumerate(splits):
        for nm, cols in INPUTS.items():
            X = d[cols].to_numpy(float)
            sc = StandardScaler().fit(X[itr])
            m = LogisticRegression(max_iter=2000).fit(sc.transform(X[itr]), y[itr])
            pr = m.predict_proba(sc.transform(X[ite]))[:, 1]
            pred = (pr >= 0.5).astype(int)
            yte = y[ite]
            tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
            rows.append(dict(input=nm, cv=cv, fold=k,
                             subject=groups[ite][0] if cv == "loso" else "",
                             acc=accuracy_score(yte, pred) * 100,
                             rec=tp / (tp + fn) * 100 if tp + fn else np.nan,
                             spec=tn / (tn + fp) * 100 if tn + fp else np.nan,
                             auc=roc_auc_score(yte, pr) if len(set(yte)) > 1 else np.nan,
                             prauc=average_precision_score(yte, pr)))

r = pd.DataFrame(rows)
r.to_csv(args.out, index=False)
for cv in ("random", "group", "loso"):
    print(f"\n== cv={cv} ({r[r.cv == cv].fold.nunique()} folds), media (DP)")
    g = r[r.cv == cv].groupby("input", sort=False)[["acc", "rec", "spec", "auc", "prauc"]].agg(["mean", "std"])
    print(g.round(3).to_string())

# ---------------------------------------------------------------- comparacoes
# Emparelhadas por fold com as corridas V6 da tese (GroupKFold e LOSO sao
# deterministas, os folds coincidem com os do train_sequence.py e do
# wesad_baselines.py; a LogReg de 8 features reproduz a da tese ao decimal).
from scipy import stats

NETS = {"E": "summary", "F": "seq", "G": "both"}


def fold_metrics(cv, name):
    if name == "LogReg":
        return r[(r.cv == cv) & (r.input == "8 features")].sort_values("fold").reset_index(drop=True)
    if name == "hr_mean":
        return r[(r.cv == cv) & (r.input == "hr_mean")].sort_values("fold").reset_index(drop=True)
    if name == "RF":
        b = pd.read_csv("results_wesad_v6_baselines.csv")
        return b[(b.cv == cv) & (b.model == "RandomForest")].sort_values("fold").reset_index(drop=True)
    if name in NETS:
        return pd.read_csv(f"results_wesad_v6_{cv}_{NETS[name]}.csv").reset_index(drop=True)
    x = pd.read_csv(f"results_wesad_v6_{cv}_ABCD.csv")
    return x[x.model == name].reset_index(drop=True)


print("\n\n== Diferencas emparelhadas por fold (media, IC95 t, p do t emparelhado)")
for cv in ("group", "loso"):
    for a_, b_ in (("G", "hr_mean"), ("LogReg", "hr_mean"), ("G", "LogReg")):
        A, B = fold_metrics(cv, a_), fold_metrics(cv, b_)
        for met in ("auc", "acc", "rec"):
            dlt = (A[met] - B[met]).dropna().to_numpy()
            lo, hi = stats.t.interval(0.95, len(dlt) - 1, loc=dlt.mean(), scale=stats.sem(dlt))
            print(f"  {cv:5s} {a_:6s} - {b_:7s} {met:4s} {dlt.mean():+7.3f}  "
                  f"[{lo:+.3f}; {hi:+.3f}]  p={stats.ttest_rel(A[met], B[met], nan_policy='omit').pvalue:.3f}")

print("\n== Sensibilidade e especificidade por sujeito (media +/- DP entre folds)")
for name in ("hr_mean", "LogReg", "RF", "E", "A", "B", "C", "D", "F", "G"):
    s = "  " + f"{name:8s}"
    for cv in ("group", "loso"):
        x = fold_metrics(cv, name)
        s += (f"  {cv}: sens {x.rec.mean():6.2f} +/- {x.rec.std():5.2f}"
              f"  spec {x.spec.mean():6.2f} +/- {x.spec.std():5.2f}")
    print(s)
