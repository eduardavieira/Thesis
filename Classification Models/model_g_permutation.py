#!/usr/bin/env python3
"""
O Model G usa a ORDEM dos batimentos? Teste de permutacao dentro da janela.

    python3 model_g_permutation.py      # precisa de saved_models/wesad_model_G.weights.h5

Porque. A ablacao da Seccao 4.4 substitui a serie pela media da propria janela.
Isso tira a ordem, mas tambem toda a variabilidade, e cria uma entrada que o
modelo nunca viu (serie constante com sdnn > 0 no ramo resumo). Nao isola a
ordem. Baralhar os intervalos dentro de cada janela mantem os valores (e com
eles a media, o minimo, o maximo, o sdnn e o numero de batimentos) e so destroi
a ordem.

Tres condicoes, R permutacoes cada, sobre as 189 janelas de teste (seed 42,
mesma particao do explain_model_g.py):
  flat      serie = media da propria janela (reproduz a ablacao da tese)
  perm      serie baralhada, resumo intacto. O rmssd/nn50/pnn50 do resumo
            continuam a reflectir a ordem verdadeira.
  perm+rec  serie baralhada e as 8 features recalculadas a partir dela: nenhuma
            das duas entradas guarda informacao de ordem. E o teste limpo.

rmssd, nn50 e pnn50 dependem da ordem (diferencas sucessivas); hr_mean,
nni_mean, nni_min, nni_max e sdnn nao.
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from train_sequence import FEATURES, build

p = argparse.ArgumentParser()
p.add_argument("--csv", default="/home/eduarda/Desktop/ProcessedECGData_v5.csv")
p.add_argument("--series", default="/home/eduarda/Desktop/NNSeries_v5.npz")
p.add_argument("--weights", default="saved_models/wesad_model_G.weights.h5")
p.add_argument("--perms", type=int, default=50)
p.add_argument("--out", default="results_model_g_permutation.csv")
a = p.parse_args()

raw = pd.read_csv(a.csv)
M = np.load(a.series)["nni"]
keep = raw[FEATURES + ["label"]].notna().all(1).to_numpy() & raw.file.str.contains("ECG_values").to_numpy()
raw, M = raw[keep].reset_index(drop=True), M[keep]
Xf = raw[FEATURES].to_numpy(float)
y = raw.label.to_numpy().astype(int)

itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
np.random.seed(42)
itr = np.random.permutation(itr)            # como no train_sequence.py / explain_model_g.py
real = M[itr] > 0
mu, sd = M[itr][real].mean(), M[itr][real].std()
fmu, fsd = Xf[itr].mean(0), Xf[itr].std(0)
prep_S = lambda m: np.where(m > 0, (m - mu) / sd, 0.0)[..., None].astype("float32")
prep_F = lambda x: np.clip((x - fmu) / fsd, -5, 5).astype("float32")
Mte, yte = M[ite], y[ite]
K = (Mte > 0).astype("float32")[..., None]


def features(nni):
    """As 8 features exactamente como no ETL (compute_features)."""
    d = np.diff(nni)
    nn50 = int(np.sum(np.abs(d) > 50))
    return [np.mean(60000.0 / nni), np.mean(nni), np.min(nni), np.max(nni),
            np.std(nni, ddof=1), np.sqrt(np.mean(d ** 2)), nn50, nn50 / len(d) * 100.0]


# Verificacao: recalcular a partir da serie guardada reproduz a CSV
rec = np.array([features(r[r > 0]) for r in Mte])
err = np.abs(rec - Xf[ite]).max(0)
print("erro max. ao recalcular as features a partir da serie:",
      dict(zip(FEATURES, np.round(err, 6))))

m = build("both", M.shape[1], len(FEATURES))
m.load_weights(a.weights)
P = lambda Mx, Fx: m([prep_S(Mx), K, prep_F(Fx)], training=False).numpy().flatten()


def score(prob):
    return accuracy_score(yte, prob > 0.5) * 100, roc_auc_score(yte, prob)


rows = []
acc0, auc0 = score(P(Mte, Xf[ite]))
rows.append(("intact", 0, acc0, auc0))

flat = np.where(Mte > 0, np.array([r[r > 0].mean() for r in Mte])[:, None], 0.0)
rows.append(("flat", 0, *score(P(flat, Xf[ite]))))

rng = np.random.default_rng(0)
for k in range(a.perms):
    Mp = Mte.copy()
    for i, r in enumerate(Mp):
        n = int((r > 0).sum())
        Mp[i, :n] = rng.permutation(r[:n])
    Fp = np.array([features(r[r > 0]) for r in Mp])
    rows.append(("perm", k, *score(P(Mp, Xf[ite]))))
    rows.append(("perm+rec", k, *score(P(Mp, Fp))))

df = pd.DataFrame(rows, columns=["condition", "rep", "acc", "auc"])
df.to_csv(a.out, index=False)
print(f"\nModel G, WESAD, N={len(yte)}, {a.perms} permutacoes")
print(f"{'condicao':10s} {'exatidao':>18s} {'AUC':>18s}   queda acc / AUC")
for c, g in df.groupby("condition", sort=False):
    print(f"{c:10s} {g.acc.mean():8.2f} +/- {g.acc.std(ddof=1) if len(g) > 1 else 0:5.2f}"
          f"   {g.auc.mean():7.3f} +/- {g.auc.std(ddof=1) if len(g) > 1 else 0:5.3f}"
          f"   {g.acc.mean() - acc0:+6.2f} pp / {g.auc.mean() - auc0:+.3f}")
print(f"\n-> {a.out}")
