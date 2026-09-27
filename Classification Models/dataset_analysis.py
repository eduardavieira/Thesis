#!/usr/bin/env python3
"""
Numeros da Seccao 4.1 (comparabilidade dos datasets) e 4.1.1 (controlo negativo)
que nao vem de treinar redes. Reconstruido em 2026-09-26: os scripts originais
estavam no scratchpad e perderam-se. Validado contra os valores V5 da tese.

    python3 dataset_analysis.py --csv .../ProcessedECGData_v5.csv --series .../NNSeries_v5.npz
    python3 dataset_analysis.py --csv .../ProcessedECGData_v6.csv --series .../NNSeries_v6.npz \
        --pooled-out results_v6_pooled_baselines.csv

  1. d de Cohen por feature e dataset (DP agregado ponderado), FC media por classe
  2. Consistencia por sujeito: FC(stress) - FC(nao-stress), mediana e contagem
  3. SWELL: declive da FC ao longo de cada bloco; 2.a metade do neutral vs stress;
     exclusao das 16 janelas do filme (apendice A.1.1)
  4. Baselines do combinado (majority, LogReg, RF, regra "so sei o dataset"),
     split aleatorio 5 seeds e GroupKFold k=5 por dataset+sujeito
  5. Classificador de dominio (WESAD vs SWELL): 8 features e serie NN
  6. Contagens de janelas e duracao por dataset/classe
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import (GroupKFold, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.preprocessing import StandardScaler

F = ["hr_mean", "nni_mean", "nni_min", "nni_max", "sdnn", "rmssd", "nn50", "pnn50"]
p = argparse.ArgumentParser()
p.add_argument("--csv", required=True)
p.add_argument("--series", required=True)
p.add_argument("--pooled-out", default=None)
a = p.parse_args()

d = pd.read_csv(a.csv)
Z = np.load(a.series)
M, L = Z["nni"], Z["length"]
ok = d[F + ["label"]].notna().all(1).to_numpy()
d, M, L = d[ok].reset_index(drop=True), M[ok], L[ok]
d["ds"] = np.where(d.file.str.contains("ECG_values"), "WESAD", "SWELL")
d["subj"] = d.file.str.extract(r"^(S\d+)")[0]
d["cond"] = d.file.str.replace("_ECG_values.csv", "", regex=False).str.replace(".csv", "", regex=False) \
    .str.split("_", n=1).str[1]
d["pos"] = d.groupby("file").cumcount()                      # indice da janela no ficheiro
d["n_in_file"] = d.groupby("file").file.transform("size")

print(f"Janelas: {len(d)}  stress {d.label.mean()*100:.2f}%")
for ds, g in d.groupby("ds"):
    print(f"  {ds}: {len(g)} janelas, {g.label.mean()*100:.2f}% stress, {g.subj.nunique()} sujeitos, "
          f"nao-stress {int((g.label==0).sum())} / stress {int(g.label.sum())}")

# ------------------------------------------------------------ 1. d de Cohen
print("\n1. d de Cohen (stress - nao-stress)")
rows = []
for ds in ["WESAD", "SWELL"]:
    g = d[d.ds == ds]
    for f in F:
        x1, x0 = g.loc[g.label == 1, f], g.loc[g.label == 0, f]
        n1, n0 = len(x1), len(x0)
        sp = np.sqrt(((n1 - 1) * x1.var() + (n0 - 1) * x0.var()) / (n1 + n0 - 2))
        rows.append((ds, f, (x1.mean() - x0.mean()) / sp))
cd = pd.DataFrame(rows, columns=["ds", "f", "d"]).pivot(index="f", columns="ds", values="d").loc[F]
print(cd.round(2).to_string())
print("  media |d|:", cd.abs().mean().round(2).to_dict())
for ds in ["WESAD", "SWELL"]:
    g = d[d.ds == ds]
    h0, h1 = g.loc[g.label == 0, "hr_mean"].mean(), g.loc[g.label == 1, "hr_mean"].mean()
    print(f"  {ds}: hr_mean {h0:.1f} -> {h1:.1f} bpm ({h1-h0:+.1f})")

# ------------------------------------------------------------ 2. por sujeito
print("\n2. FC(stress) - FC(nao-stress) por sujeito")
for ds in ["WESAD", "SWELL"]:
    g = d[d.ds == ds]
    m = g.groupby(["subj", "label"]).hr_mean.mean().unstack()
    m = m.dropna()
    diff = m[1] - m[0]
    print(f"  {ds}: {len(diff)} sujeitos, mediana {diff.median():+.2f} bpm, "
          f"mais alta no stress {int((diff>0).sum())}, mais baixa {int((diff<0).sum())}")

# ------------------------------------------------------------ 3. SWELL: blocos
# Receita reconstruida: OLS da FC centrada por ficheiro na posicao relativa da
# janela no bloco (0..1), todos os ficheiros de cada condicao juntos. Na V5 da
# +0,13 (p=0,78) / +2,26 / +6,64; a tese tinha -0,03 (p=0,98) / 2,16 / 6,61.
print("\n3. SWELL: FC ao longo de cada bloco (posicao relativa 0..1, centrada por ficheiro)")
s = d[d.ds == "SWELL"].copy()
s["t"] = s.pos / (s.n_in_file - 1).clip(lower=1)
for c in ["neutral", "stress_I", "stress_T"]:
    g = s[s.cond == c]
    y = g.hr_mean - g.groupby("file").hr_mean.transform("mean")
    r = stats.linregress(g.t, y)
    print(f"  {c:9s}: declive {r.slope:+.2f} bpm por bloco, p={r.pvalue:.3g}  (n={len(g)})")
# 2.a metade do neutral vs stress, por sujeito
# Receita reconstruida (a original perdeu-se; na V5 dava -4,38 e esta da -4,46):
# diferenca das medias ao nivel da janela, stress - 2.a metade do neutral.
neu2 = s[(s.cond == "neutral") & (s.t >= 0.5)]
dw = s[s.label == 1].hr_mean.mean() - neu2.hr_mean.mean()
dd = (s[s.label == 1].groupby("subj").hr_mean.mean() - neu2.groupby("subj").hr_mean.mean()).dropna()
print(f"  2.a metade do neutral vs stress: {dw:+.2f} bpm (janelas); por sujeito "
      f"mediana {dd.median():+.2f}, negativos {int((dd<0).sum())}/{len(dd)}")
# Filme: 16 primeiras janelas de cada ficheiro
sf = s[s.pos >= 16]
m = sf.groupby(["subj", "label"]).hr_mean.mean().unstack().dropna()
df_ = m[1] - m[0]
print(f"  sem as 16 janelas do filme: mediana {df_.median():+.2f} bpm, "
      f"negativos {int((df_<0).sum())}/{len(df_)}")

# ------------------------------------------------------------ 4. baselines do combinado
print("\n4. Combinado: baselines (5 seeds aleatorio / GroupKFold k=5)")
X = d[F].to_numpy(float)
y = d.label.to_numpy().astype(int)
ds_arr = d.ds.to_numpy()
groups = (d.ds + "_" + d.subj).to_numpy()
rows = []
for cv in ["random", "group"]:
    splits = ([train_test_split(np.arange(len(y)), test_size=0.2, random_state=s_, stratify=y)
               for s_ in range(42, 47)] if cv == "random"
              else list(GroupKFold(5).split(X, y, groups)))
    for k, (tr, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])
        maj = int(y[tr].mean() > 0.5)
        rows.append(dict(model="Majority", cv=cv, fold=k,
                         acc=accuracy_score(y[te], np.full(len(te), maj)) * 100,
                         spec=100.0 if maj == 0 else 0.0, auc=0.5))
        # regra "so sei o dataset": classe prevalente de cada dataset no treino
        rule = {ds_: int(y[tr][ds_arr[tr] == ds_].mean() > 0.5) for ds_ in ["WESAD", "SWELL"]}
        pr = np.array([rule[x] for x in ds_arr[te]])
        rows.append(dict(model="Dataset-only", cv=cv, fold=k,
                         acc=accuracy_score(y[te], pr) * 100, spec=np.nan, auc=np.nan))
        for nm, mk in [("LogReg", lambda: LogisticRegression(max_iter=2000)),
                       ("RandomForest", lambda: RandomForestClassifier(
                           n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1))]:
            m_ = mk().fit(sc.transform(X[tr]), y[tr])
            pp = m_.predict_proba(sc.transform(X[te]))[:, 1]
            tn, fp, fn, tp = confusion_matrix(y[te], pp > 0.5, labels=[0, 1]).ravel()
            rows.append(dict(model=nm, cv=cv, fold=k, acc=accuracy_score(y[te], pp > 0.5) * 100,
                             spec=tn / (tn + fp) * 100, auc=roc_auc_score(y[te], pp)))
pb = pd.DataFrame(rows)
print(pb.groupby(["cv", "model"])[["acc", "spec", "auc"]].agg(["mean", "std"]).round(3).to_string())
if a.pooled_out:
    pb.to_csv(a.pooled_out, index=False)
    print(f"  -> {a.pooled_out}")

# ------------------------------------------------------------ 5. dominio
print("\n5. Classificador de dominio (WESAD=1), RF 200, StratifiedKFold 5, AUC")
yd = (ds_arr == "WESAD").astype(int)
cvk = StratifiedKFold(5, shuffle=True, random_state=42)
rf = lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
pp = cross_val_predict(rf(), X, yd, cv=cvk, method="predict_proba")[:, 1]
print(f"  8 features: AUC {roc_auc_score(yd, pp):.3f}")
pp = cross_val_predict(rf(), L[:, None].astype(float), yd, cv=cvk, method="predict_proba")[:, 1]
print(f"  so o n.o de batimentos: AUC {roc_auc_score(yd, pp):.3f}")


def series_feats(r):
    x = r[r > 0].astype(float)
    dx = np.diff(x)
    q700 = 1000.0 / 700
    # A 700 Hz, 7 amostras = 10 ms: truncados ao ms, os intervalos nunca acabam
    # em 3, 6 ou 9. Na V5 esta receita da AUC 0,994 (a tese tinha 0,909, com
    # uma receita que se perdeu).
    # residuo de quantizacao: intervalos truncados a ms inteiros so tomam os
    # valores int(k * 1000/700) a 700 Hz; distancia ao valor mais proximo possivel
    k = np.round(x / q700)
    cand = np.stack([np.floor((k + j) * q700) for j in (-1, 0, 1)])
    resid = np.min(np.abs(cand - x), axis=0).mean()
    return [x.mean(), x.std(), x.min(), x.max(), len(x), np.abs(dx).mean(), np.sqrt(np.mean(dx ** 2)),
            np.median(x), resid]


S = np.array([series_feats(r) for r in M])
pp = cross_val_predict(rf(), S, yd, cv=cvk, method="predict_proba")[:, 1]
print(f"  serie NN (estatisticas + residuo 700 Hz): AUC {roc_auc_score(yd, pp):.3f}")
imp = rf().fit(S, yd).feature_importances_
print("  importancias:", dict(zip(["mean", "std", "min", "max", "n", "mad", "rmssd", "median", "resid_700Hz"],
                                   np.round(imp, 3))))

# ------------------------------------------------------------ 7. RF so no SWELL
# Seccao 4.6 ("Relation to Previous Work"): quanto cai a RF dentro do SWELL
# quando a particao passa de aleatoria a por sujeito. Na V5: 65,19 -> 56,45.
print("\n7. RandomForest so no SWELL: aleatorio (5 seeds) vs GroupKFold k=5")
m_s = (ds_arr == "SWELL")
Xs, ys, gs = X[m_s], y[m_s], d.subj.to_numpy()[m_s]
for cv in ["random", "group"]:
    splits = ([train_test_split(np.arange(len(ys)), test_size=0.2, random_state=s_, stratify=ys)
               for s_ in range(42, 47)] if cv == "random" else list(GroupKFold(5).split(Xs, ys, gs)))
    accs = []
    for tr, te in splits:
        sc = StandardScaler().fit(Xs[tr])
        m_ = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced",
                                    n_jobs=-1).fit(sc.transform(Xs[tr]), ys[tr])
        accs.append(accuracy_score(ys[te], m_.predict(sc.transform(Xs[te]))) * 100)
    print(f"  {cv}: {np.mean(accs):.2f} +/- {np.std(accs, ddof=1):.2f}")
