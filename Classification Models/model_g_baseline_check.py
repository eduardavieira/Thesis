#!/usr/bin/env python3
"""De onde vem o f(baseline) = 0,916 do Model G? (3.a avaliacao externa, 2026-09-29)

A tese (4.4.4) atribui-o aos pesos de classe. O avaliador propoe outra causa: a
baseline dos integrated gradients e uma serie PLANA (todos os intervalos iguais a
media do treino), uma entrada que nunca aparece no treino e que, sem variabilidade,
se parece com stress. Este script separa as duas hipoteses com os pesos V6 da tese
(seed 42, mesma particao da 4.4), sem treinar nada:

  1. baseline da tese: serie plana + resumo na media
  2. serie com a variabilidade de cada janela de teste, mas recentrada na media do
     treino (a media da janela passa a ser a do treino), resumo na media
  3. serie plana + resumo real / serie real + resumo na media
  4. media de P(stress) nas janelas reais de treino vs prevalencia: o efeito dos
     pesos de classe na probabilidade media (os pesos 1,67/0,78 multiplicam as
     odds por ~2,1 no maximo)
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from train_sequence import FEATURES, build

p = argparse.ArgumentParser()
p.add_argument("--csv", default="/home/eduarda/Desktop/ProcessedECGData_v6_wesad.csv")
p.add_argument("--series", default="/home/eduarda/Desktop/NNSeries_v6_wesad.npz")
p.add_argument("--weights", default="saved_models/wesad_model_G_v6.weights.h5")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

# mesma preparacao do explain_model_g.py
raw = pd.read_csv(a.csv)
z = np.load(a.series)
M = z["nni"]
keep = raw[FEATURES + ["label"]].notna().all(axis=1).to_numpy()
keep &= raw["file"].str.contains("ECG_values").to_numpy()
raw, M = raw[keep].reset_index(drop=True), M[keep]
Xf = raw[FEATURES].to_numpy(float)
y = raw["label"].to_numpy().astype(int)
itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, random_state=a.seed, stratify=y)
tf.random.set_seed(a.seed); np.random.seed(a.seed)
itr = np.random.permutation(itr)
real = M[itr] > 0
mu, sd = M[itr][real].mean(), M[itr][real].std()
prep_seq = lambda m: np.where(m > 0, (m - mu) / sd, 0.0)[..., None].astype("float32")
prep_mask = lambda m: (m > 0).astype("float32")[..., None]
fmu, fsd = Xf[itr].mean(0), Xf[itr].std(0)
prep_f = lambda x: np.clip((x - fmu) / fsd, -5, 5).astype("float32")
Str, Ktr, Ftr = prep_seq(M[itr]), prep_mask(M[itr]), prep_f(Xf[itr])
Ste, Kte, Fte = prep_seq(M[ite]), prep_mask(M[ite]), prep_f(Xf[ite])
ytr, yte = y[itr], y[ite]

model = build("both", M.shape[1], len(FEATURES))
model.load_weights(a.weights)
f = lambda S, K, F: model([S, K, F], training=False).numpy().flatten()


def centred(S, K):
    """Tira a cada janela a sua media (so nas posicoes reais): fica a variabilidade."""
    n = K.sum(1, keepdims=True)
    m = (S * K).sum(1, keepdims=True) / n
    return ((S - m) * K).astype("float32")


def show(lab, p, cls=None):
    s = f"  {lab:<58s} media {p.mean():.3f}  mediana {np.median(p):.3f}"
    if cls is not None:
        s += f"  | nao-stress {p[cls == 0].mean():.3f}  stress {p[cls == 1].mean():.3f}"
    print(s)


print(f"Model G V6, seed {a.seed}, teste N={len(yte)}\n")
zS, zF = np.zeros_like(Ste), np.zeros_like(Fte)
show("0. entradas reais", f(Ste, Kte, Fte), yte)
show("1. baseline da tese (serie plana, resumo na media)", f(zS, Kte, zF), yte)
show("2. serie recentrada (variabilidade real), resumo na media", f(centred(Ste, Kte), Kte, zF), yte)
show("3a. serie plana, resumo real", f(zS, Kte, Fte), yte)
show("3b. serie real, resumo na media", f(Ste, Kte, zF), yte)

# 2b. o mesmo com janelas de TREINO (para nao depender do teste), por classe
show("2b. idem com as janelas de treino", f(centred(Str, Ktr), Ktr, np.zeros_like(Ftr)), ytr)

# 4. calibracao global no treino: probabilidade media vs prevalencia
ptr = f(Str, Ktr, Ftr)
prev = ytr.mean()
odds = lambda q: q / (1 - q)
print(f"\n  4. treino: prevalencia {prev:.3f}; P(stress) media {ptr.mean():.3f}")
print(f"     odds medias previstas / odds da prevalencia = "
      f"{odds(np.clip(ptr, 1e-6, 1 - 1e-6)).mean() / odds(prev):.2f}"
      f"  (pesos 1,67/0,78 -> fator maximo esperado {1.67/0.78:.2f})")
# que P(stress) teria a baseline se os pesos fossem a unica causa? odds 0,916 / 2,14
p0 = 0.916
print(f"     0,916 sem o fator dos pesos: odds {odds(p0):.2f} / {1.67/0.78:.2f} -> "
      f"P = {odds(p0)/(1.67/0.78)/(1+odds(p0)/(1.67/0.78)):.3f}"
      f"  (prevalencia no treino {prev:.3f})")
