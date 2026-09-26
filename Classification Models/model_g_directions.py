#!/usr/bin/env python3
"""
Sentido dos efeitos aprendidos pelo Model G (Secao 4.4, "Summary Features").

    python3 model_g_directions.py      # precisa de saved_models/wesad_model_G.weights.h5

Duas vias independentes sobre as 189 janelas de teste (seed 42, mesma particao
do explain_model_g.py):
  1. Spearman entre o valor de cada feature resumo e o seu integrated gradient
  2. Perturbacao directa: +0,5 SD numa feature resumo, tudo o resto fixo
Mais dois testes ao modelo inteiro:
  3. Serie inteira -0,5 SD (FC mais alta) com o resumo fixo
  4. Intervalos x0,9 com a serie E o resumo recalculados de forma coerente
  5. Spearman entre o valor de cada batimento e o seu IG (todos os batimentos)

Resultado registado (2026-09-24): hr_mean, rmssd e nn50 com efeito condicional
contrario a fisiologia nas duas vias; o modelo inteiro vai a favor (P sobe em
87% das janelas com intervalos x0,9); por batimento rho = -0,12.
"""
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from train_sequence import FEATURES, build

raw = pd.read_csv("/home/eduarda/Desktop/ProcessedECGData_v5.csv")
M = np.load("/home/eduarda/Desktop/NNSeries_v5.npz")["nni"]
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
S = np.where(M[ite] > 0, (M[ite] - mu) / sd, 0.0)[..., None].astype("float32")
K = (M[ite] > 0).astype("float32")[..., None]
F = np.clip((Xf[ite] - fmu) / fsd, -5, 5).astype("float32")

m = build("both", M.shape[1], len(FEATURES))
m.load_weights("saved_models/wesad_model_G.weights.h5")
P = lambda s, k, f: m([s, k, f], training=False).numpy().flatten()
p = P(S, K, F)

# ---- 1. IG por feature resumo e por batimento (50 passos, regra do trapezio)
al = np.linspace(0, 1, 51, dtype="float32")
igF, igS = [], []
for i in range(len(ite)):
    Sp = tf.constant(al[:, None, None] * S[i][None])
    Fp = tf.constant(al[:, None] * F[i][None])
    Kp = tf.constant(np.repeat(K[i][None], 51, 0))
    with tf.GradientTape() as t:
        t.watch([Sp, Fp])
        o = m([Sp, Kp, Fp], training=False)
    gs, gf = t.gradient(o, [Sp, Fp])
    igF.append(F[i] * (gf[:-1] + gf[1:]).numpy().mean(0) / 2)
    igS.append((S[i] * (gs[:-1] + gs[1:]).numpy().mean(0) / 2)[:, 0])
igF = np.array(igF)

w, wy = Xf[ite], y[ite]
print("feature   d(teste)  rho IG~valor   dP(+0,5 SD, resto fixo)")
for j, f in enumerate(FEATURES):
    a, b = w[wy == 1, j], w[wy == 0, j]
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    r, _ = spearmanr(F[:, j], igF[:, j])
    F2 = F.copy(); F2[:, j] += 0.5
    print(f"{f:9s} {d:+6.2f}    {r:+.2f}          {P(S, K, F2).mean() - p.mean():+.4f}")

# ---- 3. serie mais rapida, resumo fixo
S2 = np.where(K > 0, S - 0.5, 0.0).astype("float32")
print(f"\nserie -0,5 SD, resumo fixo: dP = {P(S2, K, F).mean() - p.mean():+.4f}")

# ---- 4. janela inteira coerente: intervalos x0,9, serie e resumo recalculados
def feats(mm):
    out = []
    for r in mm:
        x = r[r > 0]; d = np.diff(x)
        out.append([60000 / x.mean(), x.mean(), x.min(), x.max(), x.std(ddof=1),
                    np.sqrt((d ** 2).mean()), (np.abs(d) > 50).sum(), (np.abs(d) > 50).mean() * 100])
    return np.array(out)
Mt = M[ite].astype(float); Mt2 = np.where(Mt > 0, Mt * 0.9, 0)
prep_f = lambda x: np.clip((x - fmu) / fsd, -5, 5).astype("float32")
prep_s = lambda x: np.where(x > 0, (x - mu) / sd, 0)[..., None].astype("float32")
pa, pb = P(prep_s(Mt), K, prep_f(feats(Mt))), P(prep_s(Mt2), K, prep_f(feats(Mt2)))
print(f"intervalos x0,9 (serie+resumo coerentes): dP = {pb.mean() - pa.mean():+.4f}; "
      f"sobe em {(pb > pa).mean() * 100:.0f}% das janelas")

# ---- 5. por batimento
Kb = K[..., 0] > 0
r, pv = spearmanr(np.concatenate([S[i][Kb[i], 0] for i in range(len(ite))]),
                  np.concatenate([igS[i][Kb[i]] for i in range(len(ite))]))
print(f"por batimento: Spearman(intervalo, IG) = {r:+.2f} (p={pv:.1e})")

X = raw[FEATURES]
print(f"\ncorrelacoes: nn50-pnn50 {X.nn50.corr(X.pnn50):.3f}  nn50-rmssd {X.nn50.corr(X.rmssd):.3f}  "
      f"sdnn-rmssd {X.sdnn.corr(X.rmssd):.3f}")
