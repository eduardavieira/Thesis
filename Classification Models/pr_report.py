#!/usr/bin/env python3
"""Todos os numeros do texto que dependem da corrida WESAD de 2026-09-26."""
import itertools
import numpy as np, pandas as pd
from scipy.stats import ttest_rel
from confidence_intervals import wesad, NAMES, TRAINED, ci

def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj

W = {cv: wesad(cv) for cv in ("random", "group", "loso")}
NEURAL = ["summary", "A", "B", "C", "D", "seq", "both"]
SUMM = ["A", "B", "C", "D", "summary", "LogReg", "RandomForest"]
sh = lambda m: NAMES[m].split(" (")[0]

for cv, R in W.items():
    print(f"\n=========== {cv}")
    for m in ["Majority"] + TRAINED:
        d = R[m]
        print(f"  {NAMES[m]:28s} acc {d.acc.mean():6.2f} ± {d.acc.std():5.2f}  spec {d.spec.mean():6.2f}  "
              f"auc {d.auc.mean():.3f} ± {d.auc.std():.3f}  prauc {d.prauc.mean():.3f} ± {d.prauc.std():.3f}  "
              f"CI auc [{ci(d.auc)[1]:.3f}, {ci(d.auc)[2]:.3f}]")
    for met in ("auc", "acc", "prauc"):
        rk = sorted(TRAINED, key=lambda m: -R[m][met].mean())
        print(f"  ranking {met}: " + ", ".join(f"{sh(m)} {R[m][met].mean():.3f}" for m in rk))

R = W["random"]
print("\n=========== RANDOM: planeado, Holm (familia = 7 modelos de resumo)")
for tgt in ("both", "seq"):
    for met in ("acc", "auc", "prauc"):
        ps = [ttest_rel(R[tgt][met], R[o][met]).pvalue for o in SUMM]
        q = holm(ps)
        print(f"  {sh(tgt)} {met}: " + "  ".join(f"{sh(o)} {(R[tgt][met]-R[o][met]).mean():+.3f} p={p:.4f} adj={a:.3f}{'*' if a<0.05 else ''}"
                                             for o, p, a in zip(SUMM, ps, q)))
for a, b in (("both", "seq"), ("D", "C")):
    for met in ("acc", "auc"):
        print(f"  {sh(a)}-{sh(b)} {met}: {(R[a][met]-R[b][met]).mean():+.3f} p={ttest_rel(R[a][met], R[b][met]).pvalue:.3f}")
print("  G seed 42 (linha da CSV): acc %.2f" % R["both"].loc[R["both"].seed == 42, "acc"].iloc[0])

print("\n=========== SUBJECT-WISE: 36 pares, Holm por protocolo e metrica")
for cv in ("group", "loso"):
    R = W[cv]
    for met in ("acc", "auc", "prauc"):
        pairs = list(itertools.combinations(TRAINED, 2))
        ps = [ttest_rel(R[x][met], R[y][met]).pvalue for x, y in pairs]
        q = holm(ps)
        raw = [(x, y, p) for (x, y), p in zip(pairs, ps) if p < 0.05]
        print(f"  {cv} {met}: {len(raw)} pares p<0.05 sem correcao; min p ajustado {q.min():.3f}; sobrevivem: "
              + (", ".join(f"{sh(x)}-{sh(y)}" for (x, y), a in zip(pairs, q) if a < 0.05) or "nenhum"))
    for x, y in (("LogReg", "D"), ("both", "D"), ("LogReg", "both")):
        print(f"  {cv} {sh(x)}-{sh(y)} auc {(R[x].auc-R[y].auc).mean():+.3f} p={ttest_rel(R[x].auc, R[y].auc).pvalue:.3f}")

print("\n=========== quedas de exatidao aleatorio -> GroupKFold / LOSO")
for cv in ("group", "loso"):
    dr = {m: W["random"][m].acc.mean() - W[cv][m].acc.mean() for m in TRAINED}
    lo, hi = min(dr, key=dr.get), max(dr, key=dr.get)
    print(f"  {cv}: {dr[lo]:.2f} ({sh(lo)}) a {dr[hi]:.2f} ({sh(hi)})")
for cv in ("group", "loso"):
    a = [W[cv][m].auc.mean() for m in TRAINED]; s = [W[cv][m].auc.std() for m in TRAINED]
    print(f"  {cv}: AUC {min(a):.3f}-{max(a):.3f}, SD {min(s):.3f}-{max(s):.3f}")

print("\n=========== LOSO por sujeito (fold 0 = S10)")
S = W["loso"]
for m in TRAINED:
    d = S[m]; f0 = d.iloc[0]
    print(f"  {NAMES[m]:28s} S10: auc {f0.auc:.3f} spec {f0.spec:.2f} acc {f0.acc:.2f} | gama acc {d.acc.min():.1f}-{d.acc.max():.1f} spec {d.spec.min():.1f}-{d.spec.max():.1f}")
allacc = pd.concat([S[m].acc for m in TRAINED]); allspec = pd.concat([S[m].spec for m in TRAINED])
print(f"  todas: acc {allacc.min():.1f}-{allacc.max():.1f}  spec {allspec.min():.1f}-{allspec.max():.1f}")
tr_auc = [W[cv][m].auc.mean() for cv in ("group", "loso") for m in TRAINED]
print(f"\n  AUC subject-wise de todos os treinados: {min(tr_auc):.3f}-{max(tr_auc):.3f}")
