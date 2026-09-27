#!/usr/bin/env python3
"""
Teste t emparelhado corrigido de Nadeau & Bengio (2003) no split aleatorio.

    python3 corrected_ttest.py                    # CSVs results_wesad_pr_*
    python3 corrected_ttest.py --prefix results_wesad_v6

Porque. As 5 seeds do split aleatorio sao 5 particoes 80/20 dos MESMOS dados:
os conjuntos de treino (e de teste) sobrepoem-se, as diferencas por seed nao
sao independentes e o t-test emparelhado habitual subestima a variancia. A
correccao multiplica a variancia da diferenca por (1/J + n_teste/n_treino):

    t = media(d) / sqrt((1/J + n2/n1) * var(d))      J = 5, n2/n1 = 0,25

Com J = 5 o t desce 1,5x. Nao se aplica ao GroupKFold nem ao LOSO, onde os
conjuntos de teste sao disjuntos.

Familias com Holm, como na Seccao 3.6: G (e F) contra os 7 modelos das
features resumo, por metrica. No fim, o IC95% da diferenca F2 - F1 (Model D).
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="results_wesad_pr")
p.add_argument("--ratio", type=float, default=0.25, help="n_teste / n_treino")
p.add_argument("--f1", default="results_wesad_F1_D.csv")
p.add_argument("--f2", default="results_wesad_F2_D.csv")
a = p.parse_args()


def load(cv):
    rows = []
    d = pd.read_csv(f"{a.prefix}_{cv}_ABCD.csv")
    rows += [(r.model, r.seed, r.acc, r.auc, r.prauc) for r in d.itertuples()]
    for f, name in [("summary", "E"), ("seq", "F"), ("both", "G")]:
        d = pd.read_csv(f"{a.prefix}_{cv}_{f}.csv")
        rows += [(name, r.seed, r.acc, r.auc, r.prauc) for r in d.itertuples()]
    d = pd.read_csv(f"{a.prefix}_baselines.csv")
    d = d[d.cv == cv]
    rows += [(r.model, r.fold, r.acc, r.auc, r.prauc) for r in d.itertuples()]
    return pd.DataFrame(rows, columns=["model", "k", "acc", "auc", "prauc"])


def holm(p):
    p = np.asarray(p)
    adj, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def paired(x, y, corrected):
    d = np.asarray(x) - np.asarray(y)
    J = len(d)
    v = (1 / J + a.ratio) * d.var(ddof=1) if corrected else d.var(ddof=1) / J
    t = d.mean() / np.sqrt(v)
    return d.mean(), 2 * stats.t.sf(abs(t), J - 1)


R = load("random")
get = lambda m, met: R[R.model == m].sort_values("k")[met].to_numpy()
others = ["A", "B", "C", "D", "E", "LogReg", "RandomForest"]

for ref in ["G", "F"]:
    for met in ["acc", "auc", "prauc"]:
        res = [(o, *paired(get(ref, met), get(o, met), False),
                paired(get(ref, met), get(o, met), True)[1]) for o in others]
        hn, hc = holm([r[2] for r in res]), holm([r[3] for r in res])
        print(f"\n{ref} - x  [{met}]   diff    p      Holm   |  p NB   Holm NB")
        for (o, dm, pn, pc), h1, h2 in zip(res, hn, hc):
            s1, s2 = "*" if h1 < 0.05 else " ", "*" if h2 < 0.05 else " "
            print(f"  {o:13s} {dm:+8.4f} {pn:.4f} {h1:.4f}{s1} | {pc:.4f} {h2:.4f}{s2}")

# F2 - F1 no Model D: IC95% da diferenca (naive e corrigido)
f1 = pd.read_csv(a.f1).sort_values("seed").acc.to_numpy()
f2 = pd.read_csv(a.f2).sort_values("seed").acc.to_numpy()
d = f2 - f1
J, tc = len(d), stats.t.ppf(0.975, len(d) - 1)
for lab, v in [("naive", d.var(ddof=1) / J), ("Nadeau-Bengio", (1 / J + a.ratio) * d.var(ddof=1))]:
    h = tc * np.sqrt(v)
    print(f"\nF2 - F1 exatidao ({lab}): {d.mean():+.2f} pp, IC95% [{d.mean() - h:+.2f}; {d.mean() + h:+.2f}]")
