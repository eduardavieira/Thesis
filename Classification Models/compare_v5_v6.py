#!/usr/bin/env python3
"""
V5 (durações do time_duration.csv) vs V6 (duração real = amostras/fs), WESAD.

    python3 compare_v5_v6.py

A V6 corrige as janelas do S2_baseline (22,9 s em vez de 30 s) e os desvios de
-2% a +7% dos outros ficheiros do WESAD. Mesmos scripts, mesmas seeds e folds.
Nota: com 907 janelas em vez de 941, as particoes aleatorias da seed s ja nao
sao as mesmas nas duas versoes; so os valores medios sao comparaveis.
"""
import pandas as pd

MODELS = ["A", "B", "C", "D", "E", "F", "G", "LogReg", "RandomForest", "Majority"]


def load(prefix, cv):
    rows = []
    d = pd.read_csv(f"{prefix}_{cv}_ABCD.csv")
    rows += [(r.model, r.acc, r.spec, r.auc, r.prauc) for r in d.itertuples()]
    for f, name in [("summary", "E"), ("seq", "F"), ("both", "G")]:
        d = pd.read_csv(f"{prefix}_{cv}_{f}.csv")
        rows += [(name, r.acc, r.spec, r.auc, r.prauc) for r in d.itertuples()]
    d = pd.read_csv(f"{prefix}_baselines.csv")
    d = d[d.cv == cv]
    rows += [(r.model, r.acc, r.spec, r.auc, r.prauc) for r in d.itertuples()]
    df = pd.DataFrame(rows, columns=["model", "acc", "spec", "auc", "prauc"])
    return df[df.model.isin(MODELS)]


for cv in ["random", "group", "loso"]:
    a = load("results_wesad_pr", cv).groupby("model").agg(["mean", "std"])
    b = load("results_wesad_v6", cv).groupby("model").agg(["mean", "std"])
    print(f"\n=== {cv} ===")
    print(f"{'modelo':13s} {'acc V5':>8s} {'acc V6':>8s} {'dif':>6s}   "
          f"{'AUC V5':>6s} {'AUC V6':>6s} {'dif':>7s}   {'PR V5':>6s} {'PR V6':>6s}   {'SD AUC V6':>9s}")
    for m in MODELS:
        if m not in a.index or m not in b.index:
            continue
        x, y = a.loc[m], b.loc[m]
        print(f"{m:13s} {x[('acc','mean')]:8.2f} {y[('acc','mean')]:8.2f} "
              f"{y[('acc','mean')] - x[('acc','mean')]:+6.2f}   "
              f"{x[('auc','mean')]:6.3f} {y[('auc','mean')]:6.3f} "
              f"{y[('auc','mean')] - x[('auc','mean')]:+7.3f}   "
              f"{x[('prauc','mean')]:6.3f} {y[('prauc','mean')]:6.3f}   "
              f"{y[('auc','std')]:9.3f}")
    rng = b.drop(index="Majority", errors="ignore")[("auc", "mean")]
    print(f"gama de AUC dos treinados V6: {rng.min():.3f}-{rng.max():.3f}")
