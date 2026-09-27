#!/usr/bin/env python3
"""
Referencia para a fidelidade do Model G (pedido da avaliacao externa).

    python3 model_g_faithfulness_ref.py --csv ... --series ... --weights ...

A fidelidade da tese e a queda de exatidao quando as 2 features resumo de maior
atribuicao media vao para a media do treino (0 em z). Sem referencia nao se sabe
se a queda e grande: aqui calcula-se a queda para os 28 pares possiveis e a
posicao do par escolhido pelas atribuicoes.
"""
import argparse, itertools, os, warnings
warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from train_sequence import FEATURES, build

p = argparse.ArgumentParser()
p.add_argument("--csv", default="/home/eduarda/Desktop/ProcessedECGData_v5.csv")
p.add_argument("--series", default="/home/eduarda/Desktop/NNSeries_v5.npz")
p.add_argument("--weights", default="saved_models/wesad_model_G.weights.h5")
p.add_argument("--top2", default="nn50,sdnn", help="par escolhido pelas atribuicoes")
a = p.parse_args()
raw = pd.read_csv(a.csv); M = np.load(a.series)["nni"]
keep = raw[FEATURES + ["label"]].notna().all(1).to_numpy() & raw.file.str.contains("ECG_values").to_numpy()
raw, M = raw[keep].reset_index(drop=True), M[keep]
Xf = raw[FEATURES].to_numpy(float); y = raw.label.to_numpy().astype(int)
itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
np.random.seed(42); itr = np.random.permutation(itr)
real = M[itr] > 0; mu, sd = M[itr][real].mean(), M[itr][real].std()
fmu, fsd = Xf[itr].mean(0), Xf[itr].std(0)
S = np.where(M[ite] > 0, (M[ite] - mu) / sd, 0.0)[..., None].astype("float32")
K = (M[ite] > 0).astype("float32")[..., None]
Fz = np.clip((Xf[ite] - fmu) / fsd, -5, 5).astype("float32")
m = build("both", M.shape[1], len(FEATURES)); m.load_weights(a.weights)
acc = lambda F: accuracy_score(y[ite], m([S, K, F], training=False).numpy().flatten() > 0.5) * 100
a0 = acc(Fz)
drops = {}
for i, j in itertools.combinations(range(8), 2):
    F = Fz.copy(); F[:, [i, j]] = 0.0
    drops[(FEATURES[i], FEATURES[j])] = a0 - acc(F)
d = pd.Series(drops).sort_values(ascending=False)
top = tuple(sorted(a.top2.split(","), key=FEATURES.index))
print(f"exatidao intacta {a0:.2f}")
print(f"queda media nos 28 pares {d.mean():.2f} pp (SD {d.std():.2f}), mediana {d.median():.2f}, max {d.max():.2f}")
print(f"par das atribuicoes {top}: queda {drops[top]:.2f} pp, posicao {list(d.index).index(top)+1} de 28")
print(d.round(2).head(6).to_string())
