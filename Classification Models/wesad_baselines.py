#!/usr/bin/env python3
"""LogReg / RandomForest / majority no WESAD, nos tres protocolos.

Replica o pipeline do retrain_v5.py: features por NOME, split estratificado
80/20 com seeds 42-46, GroupKFold k=5 e LOSO por sujeito, normalizacao ajustada
so no treino. Definicoes dos classificadores iguais as do transfer_test.py.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["hr_mean", "nni_mean", "nni_min", "nni_max",
            "sdnn", "rmssd", "nn50", "pnn50"]
CSV = "/home/eduarda/Desktop/ProcessedECGData_v5.csv"

d = pd.read_csv(CSV).dropna(subset=FEATURES + ["label"])
d = d[d["file"].str.contains("ECG_values")].reset_index(drop=True)   # WESAD
X = d[FEATURES].to_numpy(float)
y = d["label"].to_numpy().astype(int)
groups = d["file"].str.extract(r"^(S\d+)")[0].to_numpy()
print(f"WESAD: {len(d)} janelas, {y.mean()*100:.2f}% stress, "
      f"{len(np.unique(groups))} sujeitos")

# ATENCAO: as corridas registadas no facts.md usaram a LogReg SEM class_weight
# e a RandomForest COM class_weight="balanced". Reproduzido ao decimal nos tres
# protocolos. As duas variantes da LogReg ficam na CSV: a AUC e praticamente
# igual (0,919 vs 0,920), mas a exatidao difere 3 pp porque o peso de classe
# desloca o limiar.
MK = {"LogReg": lambda: LogisticRegression(max_iter=2000),
      "LogReg_balanced": lambda: LogisticRegression(max_iter=2000,
                                                    class_weight="balanced"),
      "RandomForest": lambda: RandomForestClassifier(
          n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
      "RandomForest_unweighted": lambda: RandomForestClassifier(
          n_estimators=200, random_state=42, n_jobs=-1)}

rows = []
for cv in ("random", "group", "loso"):
    if cv == "random":
        splits = [train_test_split(np.arange(len(y)), test_size=0.2,
                                   random_state=s, stratify=y) for s in range(42, 47)]
    elif cv == "group":
        splits = list(GroupKFold(n_splits=5).split(X, y, groups))
    else:
        splits = list(LeaveOneGroupOut().split(X, y, groups))

    for k, (itr, ite) in enumerate(splits):
        sc = StandardScaler().fit(X[itr])
        xtr, xte = sc.transform(X[itr]), sc.transform(X[ite])
        ytr, yte = y[itr], y[ite]
        maj = max(ytr.mean(), 1 - ytr.mean())
        pred_maj = np.full(len(yte), int(ytr.mean() > 0.5))
        rows.append(dict(model="Majority", cv=cv, fold=k,
                         acc=accuracy_score(yte, pred_maj) * 100, spec=np.nan,
                         auc=0.5, major=maj * 100))
        for nm, mk in MK.items():
            m = mk().fit(xtr, ytr)
            pr = m.predict_proba(xte)[:, 1]
            pred = m.predict(xte)
            tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
            rows.append(dict(model=nm, cv=cv, fold=k,
                             acc=accuracy_score(yte, pred) * 100,
                             spec=tn / (tn + fp) * 100 if tn + fp else np.nan,
                             auc=roc_auc_score(yte, pr) if len(set(yte)) > 1 else np.nan,
                             major=maj * 100))

r = pd.DataFrame(rows)
r.to_csv("results_wesad_baselines.csv", index=False)
for cv in ("random", "group", "loso"):
    print(f"\n== cv={cv}  ({r[r.cv==cv].fold.nunique()} folds)")
    g = r[r.cv == cv].groupby("model")[["acc", "spec", "auc"]].agg(["mean", "std"])
    print(g.round(3))
