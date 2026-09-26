#!/usr/bin/env python3
"""
Re-corre todas as validações que sustentam references/facts.md.

    python3 validate.py [caminho/para/ProcessedECGData.csv]

Sem argumentos usa /home/eduarda/Desktop/ProcessedECGData.csv (a que main.py
referencia). NAO usar hrv_data_final/ProcessedECGData.csv -- tem uma coluna
sdnn_index toda NaN que parte a selecao posicional de features.

LOSO demora ~5 min. Passar --fast para saltar.
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                     StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

CSV = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
    else "/home/eduarda/Desktop/ProcessedECGData.csv"
FAST = "--fast" in sys.argv

FEATURES = ["hr_mean", "nni_mean", "nni_min", "nni_max",
            "sdnn", "rmssd", "nn50", "pnn50"]

# Valores da tese, para comparacao lado a lado.
MODEL_D = dict(acc=60.68, prec=62.13, rec=76.95, f1=0.690)


def head(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


def load():
    d = pd.read_csv(CSV)
    missing = [c for c in FEATURES if c not in d.columns]
    if missing:
        sys.exit(f"ERRO: faltam colunas {missing} em {CSV}\n"
                 f"      colunas presentes: {list(d.columns)}")
    d = d.dropna(subset=FEATURES + [d.columns[1]])
    # WESAD: S#_baseline / S#_stress_ECG_values. SWELL: S#_neutral / stress_I / stress_T.
    d["ds"] = np.where(d["file"].str.contains("ECG_values"), "WESAD", "SWELL")
    # Os prefixos S1..S25 colidem entre datasets -- prefixar e obrigatorio.
    d["subj"] = d["ds"] + "_" + d["file"].str.extract(r"^(S\d+)")[0]
    return d


def clip_scale(xtr, xte):
    """Mesmo pre-processamento de Data.py:normalize_data (params so do treino)."""
    q1, q3 = np.percentile(xtr, 25, axis=0), np.percentile(xtr, 75, axis=0)
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    xtr, xte = np.clip(xtr, lo, hi), np.clip(xte, lo, hi)
    s = StandardScaler().fit(xtr)
    return np.clip(s.transform(xtr), -5, 5), np.clip(s.transform(xte), -5, 5)


d = load()
X = d[FEATURES].to_numpy()
y = d["label"].to_numpy().astype(int)
g = d["subj"].to_numpy()

head("DATASET")
print(f"  ficheiro            {CSV}")
print(f"  janelas validas     {len(d)}")
print(f"  sujeitos            {d['subj'].nunique()}  "
      f"({d[d.ds=='WESAD']['subj'].nunique()} WESAD + "
      f"{d[d.ds=='SWELL']['subj'].nunique()} SWELL)   <- a tese diz 40")
xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  treino / teste      {len(ytr)} / {len(yte)}")
print(f"  stress no teste     {yte.mean() * 100:.2f}%  "
      f"({yte.sum()} stress, {len(yte) - yte.sum()} nao-stress)")
print("\n  fracao de stress por dataset:")
for ds, s in d.groupby("ds"):
    print(f"    {ds:<6} {s['label'].mean() * 100:5.1f}%   n={len(s)}")

head("BASELINES  (split da tese: test_size=0.2, seed=42, estratificado)")
xtrn, xten = clip_scale(xtr, xte)
print(f"  {'modelo':<28}{'acc':>7}{'prec':>8}{'rec':>8}{'F1':>7}{'ROC-AUC':>9}{'PR-AUC':>8}")
for name, clf in [
    ("Majority (sempre stress)", DummyClassifier(strategy="most_frequent")),
    ("Regressao logistica", LogisticRegression(max_iter=2000)),
    ("LogReg (balanced)", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ("Random Forest (200)", RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)),
]:
    clf.fit(xtrn, ytr)
    p = clf.predict(xten)
    try:
        pr = clf.predict_proba(xten)[:, 1]
        auc, pra = roc_auc_score(yte, pr), average_precision_score(yte, pr)
    except Exception:
        auc = pra = float("nan")
    print(f"  {name:<28}{accuracy_score(yte, p) * 100:7.2f}"
          f"{precision_score(yte, p, zero_division=0) * 100:8.2f}"
          f"{recall_score(yte, p) * 100:8.2f}{f1_score(yte, p):7.3f}{auc:9.3f}{pra:8.3f}")
print(f"  {'Model D (tese)':<28}{MODEL_D['acc']:7.2f}{MODEL_D['prec']:8.2f}"
      f"{MODEL_D['rec']:8.2f}{MODEL_D['f1']:7.3f}{'n/a':>9}{'n/a':>8}")
margem = MODEL_D["acc"] - yte.mean() * 100
print(f"\n  margem do Model D sobre o majority classifier: {margem:.2f} pp")

head("SUBJECT-INDEPENDENT")
mk = {"LogReg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
      "RandomForest": lambda: RandomForestClassifier(
          n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)}
print("  GroupKFold k=5 (agrupado por sujeito):")
for name, f in mk.items():
    accs, recs = [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        a, b = clip_scale(X[tr], X[te])
        p = f().fit(a, y[tr]).predict(b)
        accs.append(accuracy_score(y[te], p))
        recs.append(recall_score(y[te], p, zero_division=0))
    print(f"    {name:<14} acc {np.mean(accs) * 100:5.2f}% (SD {np.std(accs) * 100:4.2f})"
          f"   recall {np.mean(recs) * 100:5.2f}%")

if FAST:
    print("  LOSO: saltado (--fast)")
else:
    print(f"\n  LOSO ({d['subj'].nunique()} sujeitos, demora ~5 min):")
    for name, f in mk.items():
        accs = []
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            if len(np.unique(y[tr])) < 2:
                continue
            a, b = clip_scale(X[tr], X[te])
            accs.append(accuracy_score(y[te], f().fit(a, y[tr]).predict(b)))
        print(f"    {name:<14} acc {np.mean(accs) * 100:5.2f}% "
              f"(SD {np.std(accs) * 100:5.2f}, n={len(accs)})")
print(f"\n  ambos abaixo do majority classifier ({yte.mean() * 100:.2f}%)")

head("CONTAMINACAO POR ARTEFACTOS  (filtro declarado: 500-1200 ms)")
print("  Se o filtro tivesse funcionado: nni_min>=500, nni_max<=1200, hr_mean em [50,120].")
bad = {"nni_min < 500 ms": d.nni_min < 500, "nni_max > 1200 ms": d.nni_max > 1200,
       "hr_mean > 120 bpm": d.hr_mean > 120, "hr_mean < 50 bpm": d.hr_mean < 50}
for k, m in bad.items():
    print(f"    {k:<20} {m.sum():5d}  ({m.mean() * 100:4.1f}%)")
imp = bad["nni_min < 500 ms"] | bad["nni_max > 1200 ms"] | \
      bad["hr_mean > 120 bpm"] | bad["hr_mean < 50 bpm"]
print(f"    {'QUALQUER VIOLACAO':<20} {imp.sum():5d}  ({imp.mean() * 100:4.1f}%)")
print(f"\n  gamas: nni_min [{d.nni_min.min():.0f}, {d.nni_min.max():.0f}] ms   "
      f"nni_max [{d.nni_max.min():.0f}, {d.nni_max.max():.0f}] ms   "
      f"hr_mean [{d.hr_mean.min():.1f}, {d.hr_mean.max():.1f}] bpm")
print(f"\n  contaminacao na classe stress    : {imp[d.label == 1].mean() * 100:.1f}%")
print(f"  contaminacao na classe nao-stress: {imp[d.label == 0].mean() * 100:.1f}%")
print(f"  P(stress | artefacto)    = {d.loc[imp, 'label'].mean() * 100:.1f}%")
print(f"  P(stress | sem artefacto)= {d.loc[~imp, 'label'].mean() * 100:.1f}%"
      f"   (base rate {d.label.mean() * 100:.1f}%)")

print("\n  demo do bug de mutacao da lista (RPeakDetection_HRV_RemoteMachine.py:174):")
rr = [0.4, 0.45, 0.8, 1.5, 1.6, 0.9]
orig = list(rr)
for v in rr:
    if v < 0.5 or v > 1.2:
        rr.remove(v)
print(f"    input  {orig}")
print(f"    output {rr}   <- deviam sobrar so [0.8, 0.9]")

head("AMOSTRAS DOS ESTUDOS DE CASO  (Figs. 16/18 e 17/19; amostras 3 e 50 do agente)")
idx = np.arange(len(d))
_, ite, _, _ = train_test_split(idx, y, test_size=0.2, random_state=42, stratify=y)
for k, what in ((3, "Figs. 16 e 18 - stress 85.0%"), (50, "Figs. 17 e 19 - nao-stress 84.7%")):
    r = d.iloc[ite[k]]
    print(f"\n  indice {k} ({what})")
    print(f"    {r['file']}  [{r['ds']}]  label={int(r['label'])}")
    for c in FEATURES:
        flag = ""
        if c == "nni_min" and r[c] < 500:
            flag = "  <-- FORA DO FILTRO"
        elif c == "nni_max" and r[c] > 1200:
            flag = "  <-- FORA DO FILTRO"
        elif c == "sdnn" and r[c] > 200:
            flag = "  <-- implausivel a 30 s"
        print(f"      {c:<9} {r[c]:9.2f}   (mediana {d[c].median():8.2f}){flag}")

head("CONFUNDIMENTO ENTRE DATASETS")
yds = (d.ds == "WESAD").astype(int).to_numpy()
pr = cross_val_predict(
    RandomForestClassifier(n_estimators=200, random_state=42,
                           class_weight="balanced", n_jobs=-1),
    StandardScaler().fit_transform(X), yds,
    cv=StratifiedKFold(5, shuffle=True, random_state=0),
    method="predict_proba", n_jobs=-1)[:, 1]
print(f"  ROC-AUC a distinguir WESAD de SWELL so pelas 8 features: "
      f"{roc_auc_score(yds, pr):.3f}   (0.5 = indistinguivel)")
print(f"  exatidao balanceada: {balanced_accuracy_score(yds, (pr > 0.5).astype(int)) * 100:.2f}%")

head("REDUNDANCIA ENTRE FEATURES  (apoia o argumento da Sec. 4.4.3)")
c = d[FEATURES].corr()
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        if abs(c.iloc[i, j]) > 0.8:
            print(f"    {FEATURES[i]:<9} ~ {FEATURES[j]:<9} r = {c.iloc[i, j]:+.3f}")

head("FIDELIDADE - a leitura correta")
print(f"  queda absoluta ao mascarar 2 features   60.68% -> 58.48%  =  2.20 pp")
print(f"  margem sobre o majority classifier                        =  {margem:.2f} pp")
print(f"  fracao da margem destruida pela ablacao                   =  {2.20 / margem * 100:.1f}%")
print("  => fidelidade ALTA, nao moderada. A Sec. 4.4.3 le isto ao contrario.")
print()
