#!/usr/bin/env python3
"""Calibracao so com repouso: normalizar cada sujeito pelas suas janelas de repouso.

Pergunta (3.a avaliacao externa, 2026-09-29): a calibracao da 4.6.3 usa janelas de
stress rotuladas do proprio sujeito. A alternativa realista, usada por Prajod et al.
(2024) e Byun et al. (2025), e normalizar cada pessoa pelo seu repouso, que um
utilizador consegue dar sem ter de provocar um episodio de stress.

Mesmo LOSO e mesma divisao temporal do personalised_protocol.py: por sujeito e por
classe, a 1.a metade das janelas calibra e a 2.a testa. Aqui so se usam as janelas
de NAO-stress da 1.a metade (o repouso inicial). Cada sujeito de treino e
normalizado pelo seu proprio repouso (1.a metade do baseline); o sujeito de teste
pelo seu. Depois StandardScaler ajustado no treino e LogReg sem pesos (como na
tese), limiar 0,5.

  centre  subtrai a media do repouso do sujeito a cada feature
  zscore  subtrai a media e divide pelo DP do repouso do sujeito
  none    sem normalizacao por sujeito (= condicao A da tese)

Compara com a condicao B da tese (limiar proprio, que precisa de janelas de
stress) lendo results_v6_personalised.csv.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["hr_mean", "nni_mean", "nni_min", "nni_max", "sdnn", "rmssd", "nn50", "pnn50"]
p = argparse.ArgumentParser()
p.add_argument("--csv", default="/home/eduarda/Desktop/ProcessedECGData_v6_wesad.csv")
p.add_argument("--frac", type=float, default=0.5)
p.add_argument("--personalised", default="results_v6_personalised.csv")
p.add_argument("--out", default="results_v6_rest_normalisation.csv")
a = p.parse_args()

d = pd.read_csv(a.csv)
d = d[(d[FEATURES + ["label"]].notna().all(axis=1)) & d["file"].str.contains("ECG_values")].reset_index(drop=True)
X = d[FEATURES].to_numpy(float)
y = d["label"].to_numpy().astype(int)
subj = d["file"].str.extract(r"^(S\d+)")[0].to_numpy()
subjects = sorted(set(subj), key=lambda s: int(s[1:]))


def split_subject(mask):
    """Igual ao personalised_protocol.py: temporal e por classe."""
    cal, tst = [], []
    idx = np.where(mask)[0]
    for c in (0, 1):
        ic = idx[y[idx] == c]
        k = min(max(1, int(round(len(ic) * a.frac))), len(ic) - 1)
        cal += list(ic[:k]); tst += list(ic[k:])
    return np.array(sorted(cal)), np.array(sorted(tst))


splits = {s: split_subject(subj == s) for s in subjects}
rest = {s: splits[s][0][y[splits[s][0]] == 0] for s in subjects}   # repouso de calibracao


def per_subject(Xc, mode):
    """Normaliza cada linha pelo repouso de calibracao do seu sujeito."""
    out = Xc.copy()
    for s in subjects:
        r = X[rest[s]][:, cols]
        m = subj == s
        if mode == "centre":
            out[m] = Xc[m] - r.mean(0)
        elif mode == "zscore":
            out[m] = (Xc[m] - r.mean(0)) / np.maximum(r.std(0), 1e-6)
    return out


rows = []
for inp, cols in (("8 features", list(range(8))), ("hr_mean", [0])):
    for mode in ("none", "centre", "zscore"):
        Xn = per_subject(X[:, cols], mode)
        for s in subjects:
            cal, tst = splits[s]
            tr = np.where(subj != s)[0]
            sc = StandardScaler().fit(Xn[tr])
            m = LogisticRegression(max_iter=2000).fit(sc.transform(Xn[tr]), y[tr])
            pr = m.predict_proba(sc.transform(Xn[tst]))[:, 1]
            pred = (pr > 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y[tst], pred, labels=[0, 1]).ravel()
            rows.append(dict(input=inp, mode=mode, subject=s,
                             acc=accuracy_score(y[tst], pred) * 100,
                             sens=tp / (tp + fn) * 100, spec=tn / (tn + fp) * 100,
                             auc=roc_auc_score(y[tst], pr)))

r = pd.DataFrame(rows)
r.to_csv(a.out, index=False)
print("LOSO, 2.a metade de cada condicao; media (DP) entre os 15 sujeitos\n")
summ = r.groupby(["input", "mode"], sort=False).agg(
    acc=("acc", "mean"), acc_sd=("acc", "std"), worst=("acc", "min"),
    spec_min=("spec", "min"), sens=("sens", "mean"), auc=("auc", "mean"),
    below70=("acc", lambda v: int((v < 70).sum())))
print(summ.round(2).to_string())

pp = pd.read_csv(a.personalised)
print("\nReferencia (results_v6_personalised.csv, mesmas janelas de teste):")
ref = pp.groupby(["model", "condition"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"),
                                             worst=("acc", "min"), spec_min=("spec", "min"),
                                             sens=("sens", "mean"),
                                             below70=("acc", lambda v: int((v < 70).sum())))
print(ref.round(2).to_string())

print("\nTestes emparelhados por sujeito (t; Wilcoxon):")
A = pp[(pp.model == "LogReg") & (pp.condition == "A_population")].set_index("subject").acc
B = pp[(pp.model == "LogReg") & (pp.condition == "B_threshold")].set_index("subject").acc
GB = pp[(pp.model == "G") & (pp.condition == "B_threshold")].set_index("subject").acc
for inp in ("8 features", "hr_mean"):
    for mode in ("centre", "zscore"):
        c = r[(r.input == inp) & (r["mode"] == mode)].set_index("subject").acc
        for lab, ref_ in (("LogReg A", A), ("LogReg B", B), ("G B", GB)):
            dd = (c - ref_.reindex(c.index)).to_numpy()
            print(f"  {inp:10s} {mode:6s} - {lab:8s}: {dd.mean():+6.2f} pp  "
                  f"p={stats.ttest_1samp(dd, 0).pvalue:.3f}; {stats.wilcoxon(dd).pvalue:.3f}")
