#!/usr/bin/env python3
"""
Explicabilidade do Model G (Dual-Input CNN) no WESAD.

    python3 explain_model_g.py            # treina (~2 min), guarda e analisa
    python3 explain_model_g.py --load     # reutiliza os pesos guardados

Porque um script novo. O pipeline do Model D (explain_wesad.py +
run_explainability.py + Explainability.py) assume uma unica entrada de forma
(8, 1). O Model G tem tres: a serie NN (73, 1), a mascara (73, 1) e as 8
features resumo (8,). A mascara e estrutural -- diz que posicoes sao batimentos
reais -- e nunca e atribuida: fica fixa em todos os calculos.

Particao. Replica exactamente o split do train_sequence.py com seed 42, que e o
mesmo do explain_wesad.py (train_test_split com random_state=42 e stratify=y
sobre as mesmas 941 linhas na mesma ordem), por isso as 189 janelas de teste
sao as mesmas da analise do Model D e os estudos de caso sao comparaveis.

O que calcula:
  1. Matriz de confusao e metricas (nova tabela do Cap. 4)
  2. |dy/dF| nas 8 features resumo: importancia global e media por classe,
     directamente comparaveis com as figuras do Model D
  3. Integrated gradients sobre as duas entradas. A completude permite
     decompor f(x) - f(baseline) na parte da serie e na parte do resumo --
     quanto da decisao vem de cada ramo
  4. Ablacao por ramo: serie substituida pela media da propria janela (tira a
     dinamica batimento a batimento, mantem o nivel), serie na baseline, resumo
     na baseline, ambos
  5. Fidelidade (top-2 features resumo a zero) e estabilidade (Spearman sob
     ruido sigma=0,05), com os protocolos do Model D
  6. Estudos de caso TP / TN / FP / FN com os criterios do Model D, com a serie
     NN desenhada e colorida pela atribuicao de cada batimento
  7. Latencia de inferencia, com o mesmo metodo do measure_inference.py

Nota: o TensorFlow em CPU nao e determinista. O modelo treinado aqui nao
reproduz ao decimal a linha seed 42 do results_wesad_random_both.csv.
"""
import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from train_sequence import FEATURES, build

p = argparse.ArgumentParser()
p.add_argument("--csv", default="/home/eduarda/Desktop/ProcessedECGData_v5.csv")
p.add_argument("--series", default="/home/eduarda/Desktop/NNSeries_v5.npz")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--weights", default="saved_models/wesad_model_G.weights.h5")
p.add_argument("--out", default="results/model_G")
p.add_argument("--load", action="store_true",
               help="carregar os pesos de --weights em vez de treinar")
p.add_argument("--ig-steps", type=int, default=50)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
os.makedirs(os.path.dirname(a.weights), exist_ok=True)

# ------------------------------------------------------------------ dados
raw = pd.read_csv(a.csv)
z = np.load(a.series)
M, Lg = z["nni"], z["length"]
if len(raw) != len(M):
    raise SystemExit(f"ERRO: CSV com {len(raw)} linhas, npz com {len(M)}")
keep = raw[FEATURES + ["label"]].notna().all(axis=1).to_numpy()
keep &= raw["file"].str.contains("ECG_values").to_numpy()          # WESAD
raw, M, Lg = raw[keep].reset_index(drop=True), M[keep], Lg[keep]
Xf = raw[FEATURES].to_numpy(float)
y = raw["label"].to_numpy().astype(int)
print(f"WESAD: {len(raw)} janelas, {y.mean()*100:.2f}% stress")

itr, ite = train_test_split(np.arange(len(y)), test_size=0.2,
                            random_state=a.seed, stratify=y)
tf.keras.backend.clear_session()
tf.random.set_seed(a.seed)
np.random.seed(a.seed)
itr = np.random.permutation(itr)          # como no train_sequence.py

real = M[itr] > 0
mu, sd = M[itr][real].mean(), M[itr][real].std()
prep_seq = lambda m: np.where(m > 0, (m - mu) / sd, 0.0)[..., None].astype("float32")
prep_mask = lambda m: (m > 0).astype("float32")[..., None]
fmu, fsd = Xf[itr].mean(0), Xf[itr].std(0)
prep_f = lambda x: np.clip((x - fmu) / fsd, -5, 5).astype("float32")

Str, Ktr, Ftr = prep_seq(M[itr]), prep_mask(M[itr]), prep_f(Xf[itr])
Ste, Kte, Fte = prep_seq(M[ite]), prep_mask(M[ite]), prep_f(Xf[ite])
ytr, yte = y[itr], y[ite]
files_te = raw["file"].to_numpy()[ite]

# ------------------------------------------------------------------ modelo
model = build("both", M.shape[1], len(FEATURES))
if a.load:
    model.load_weights(a.weights)
    print(f"Pesos carregados de {a.weights}")
else:
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
                  loss="binary_crossentropy", metrics=["accuracy"])
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)))
    cw[1] *= 1.2
    model.fit([Str, Ktr, Ftr], ytr, validation_split=0.2, epochs=a.epochs,
              batch_size=64, class_weight=cw, verbose=0,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                                   restore_best_weights=True),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                       patience=10, min_lr=1e-7)])
    model.save_weights(a.weights)
    print(f"Pesos guardados em {a.weights}")


def f(S, K, F):
    return model([S, K, F], training=False).numpy().flatten()


# ------------------------------------------------------------------ 1. metricas
prob = f(Ste, Kte, Fte)
pred = (prob > 0.5).astype(int)
tn, fp, fn_, tp = confusion_matrix(yte, pred).ravel()
acc = accuracy_score(yte, pred)
print(f"\n{'='*62}\n1. MATRIZ DE CONFUSAO (Model G, seed {a.seed}, N={len(yte)})\n{'='*62}")
print(f"                     Prev: nao-stress   Prev: stress")
print(f"  Real: nao-stress   {tn:14d} {fp:14d}")
print(f"  Real: stress       {fn_:14d} {tp:14d}")
print(f"  acc {acc*100:.2f}  prec {precision_score(yte, pred)*100:.2f}  "
      f"rec {recall_score(yte, pred)*100:.2f}  spec {tn/(tn+fp)*100:.2f}  "
      f"AUC {roc_auc_score(yte, prob):.3f}  params {model.count_params():,}")


# ------------------------------------------------------------------ 2. gradientes
def grads(S, K, F):
    """|dy/dS| (mascarado) e |dy/dF|, por janela."""
    S, K, F = map(lambda t: tf.constant(t, tf.float32), (S, K, F))
    with tf.GradientTape() as t:
        t.watch([S, F])
        out = model([S, K, F], training=False)
    gS, gF = t.gradient(out, [S, F])
    return np.abs(gS.numpy()[..., 0]) * K.numpy()[..., 0], np.abs(gF.numpy())


gS, gF = grads(Ste, Kte, Fte)
imp = gF.mean(0)
imp_n = imp / imp.max()
print(f"\n{'='*62}\n2. IMPORTANCIA GLOBAL (|dy/dF| medio, dividido pelo maximo)\n{'='*62}")
for j in np.argsort(-imp_n):
    print(f"  {FEATURES[j]:9s} {imp_n[j]:.3f}")
top3 = np.sort(imp_n)[::-1]
print(f"  top-3 = {top3[:3].sum()/top3.sum()*100:.1f}% da massa; "
      f"razao 1.a/ultima {top3[0]/top3[-1]:.2f}")

order = np.argsort(imp_n)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh([FEATURES[i] for i in order], imp_n[order], color="#2c7fb8")
ax.set_xlabel("Mean Gradient Magnitude (Normalised)", fontweight="bold")
ax.set_title("Model G: global importance of the summary features (WESAD test, N=189)",
             fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{a.out}/feature_importance_global_G.png", dpi=300); plt.close()

scale = imp.max()
means = {c: gF[yte == c].mean(0) / scale for c in (1, 0)}
sems = {c: gF[yte == c].std(0, ddof=1) / np.sqrt((yte == c).sum()) / scale for c in (1, 0)}
print("\n  por classe real (media +/- EP, escala comum):")
print(f"  {'feature':9s} {'stress':>15s} {'nao-stress':>15s}  sobrepoe?")
for j in range(len(FEATURES)):
    lo1, hi1 = means[1][j] - sems[1][j], means[1][j] + sems[1][j]
    lo0, hi0 = means[0][j] - sems[0][j], means[0][j] + sems[0][j]
    ov = "sim" if (lo1 <= hi0 and lo0 <= hi1) else "NAO"
    print(f"  {FEATURES[j]:9s} {means[1][j]:7.3f}+/-{sems[1][j]:.3f} "
          f"{means[0][j]:7.3f}+/-{sems[0][j]:.3f}   {ov}")
fig, ax = plt.subplots(figsize=(12, 6))
xp, w = np.arange(len(FEATURES)), 0.35
ax.bar(xp - w/2, means[1], w, yerr=sems[1], capsize=3, label="Stress windows", color="#e74c3c")
ax.bar(xp + w/2, means[0], w, yerr=sems[0], capsize=3, label="No-stress windows", color="#3498db")
ax.set_xticks(xp); ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_ylabel("Mean Gradient Magnitude (Normalised)", fontweight="bold")
ax.set_title("Model G: gradient attribution by class (WESAD test, N=189)", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{a.out}/gradient_by_class_mean_G.png", dpi=300); plt.close()


# ------------------------------------------------------------------ 3. IG
def integrated_gradients(S, K, F, steps=a.ig_steps):
    """IG de uma janela. Baseline: serie a 0 nas posicoes reais (= intervalo
    medio do treino), mesma mascara, resumo a 0 (= media do treino)."""
    al = np.linspace(0, 1, steps + 1, dtype="float32")[:, None, None]
    Sp = tf.constant(al * S[None])
    Fp = tf.constant(al[:, :, 0] * F[None])
    Kp = tf.constant(np.repeat(K[None], steps + 1, 0))
    with tf.GradientTape() as t:
        t.watch([Sp, Fp])
        out = model([Sp, Kp, Fp], training=False)
    gs, gf = t.gradient(out, [Sp, Fp])
    # regra do trapezio
    avg = lambda g: (g[:-1] + g[1:]).numpy().mean(0) / 2
    igS = (S * avg(gs))[..., 0] * K[..., 0]
    igF = F * avg(gf)
    f0 = float(out[0, 0]); f1 = float(out[-1, 0])
    return igS, igF, f0, f1


IG = [integrated_gradients(Ste[i], Kte[i], Fte[i]) for i in range(len(yte))]
igS_tot = np.array([r[0].sum() for r in IG])
igF_tot = np.array([r[1].sum() for r in IG])
f0 = np.array([r[2] for r in IG])
compl = np.abs(igS_tot + igF_tot - (np.array([r[3] for r in IG]) - f0))
share = np.abs(igS_tot) / (np.abs(igS_tot) + np.abs(igF_tot) + 1e-12)
print(f"\n{'='*62}\n3. INTEGRATED GRADIENTS\n{'='*62}")
print(f"  f(baseline): media {f0.mean():.4f}  min {f0.min():.4f}  max {f0.max():.4f}"
      f"  (varia com o comprimento da mascara)")
print(f"  erro de completude: mediana {np.median(compl):.4f}  max {compl.max():.4f}")
print(f"  fraccao |IG serie| / (|IG serie| + |IG resumo|):")
for lab, sel in (("todas", slice(None)), ("stress", yte == 1), ("nao-stress", yte == 0)):
    s = share[sel]
    print(f"    {lab:11s} media {s.mean():.3f}  mediana {np.median(s):.3f}  "
          f"[{np.percentile(s,25):.3f}, {np.percentile(s,75):.3f}]")
igF_abs = np.abs(np.array([r[1] for r in IG])).mean(0)
print("  |IG| medio por feature resumo:")
for j in np.argsort(-igF_abs):
    print(f"    {FEATURES[j]:9s} {igF_abs[j]:.4f}")


# ------------------------------------------------------------------ 4. ablacao por ramo
def evaluate(S, F, lab):
    pr = f(S, Kte, F)
    ac = accuracy_score(yte, pr > 0.5)
    au = roc_auc_score(yte, pr)
    print(f"  {lab:44s} acc {ac*100:6.2f}  AUC {au:.3f}  "
          f"(queda {(acc-ac)*100:+.2f} pp)")
    return ac, au


flat = np.where(Kte > 0, (Ste * Kte).sum(1, keepdims=True) / Kte.sum(1, keepdims=True), 0.0)
print(f"\n{'='*62}\n4. ABLACAO POR RAMO\n{'='*62}")
abl = {
    "intacto": evaluate(Ste, Fte, "intacto"),
    "serie_plana": evaluate(flat.astype("float32"), Fte,
                            "serie = media da propria janela (sem dinamica)"),
    "serie_base": evaluate(np.zeros_like(Ste), Fte, "serie na baseline"),
    "resumo_base": evaluate(Ste, np.zeros_like(Fte), "resumo na baseline"),
    "ambos_base": evaluate(np.zeros_like(Ste), np.zeros_like(Fte), "ambos na baseline"),
}


# ------------------------------------------------------------------ 5. fidelidade e estabilidade
print(f"\n{'='*62}\n5. FIDELIDADE E ESTABILIDADE (protocolos do Model D)\n{'='*62}")
top2 = np.argsort(imp)[-2:]
Fm = Fte.copy(); Fm[:, top2] = 0
acc_m = accuracy_score(yte, f(Ste, Kte, Fm) > 0.5)
print(f"  top-2 resumo: {[FEATURES[j] for j in top2]}")
print(f"  fidelidade: {acc:.4f} -> {acc_m:.4f}, queda {acc-acc_m:.4f}")

rng = np.random.default_rng(42)
sims_F, sims_S = [], []
for i in range(min(100, len(yte))):
    s, k, fe = Ste[i:i+1], Kte[i:i+1], Fte[i:i+1]
    bS, bF = grads(s, k, fe)
    for _ in range(3):
        sp = (s + rng.normal(0, 0.05, s.shape) * k).astype("float32")
        fp_ = (fe + rng.normal(0, 0.05, fe.shape)).astype("float32")
        pS, pF = grads(sp, k, fp_)
        c, _ = spearmanr(bF[0], pF[0])
        if not np.isnan(c): sims_F.append(c)
        m = k[0, :, 0] > 0
        c, _ = spearmanr(bS[0][m], pS[0][m])
        if not np.isnan(c): sims_S.append(c)
print(f"  estabilidade resumo: {np.mean(sims_F):.4f} (SD {np.std(sims_F):.4f}, n={len(sims_F)})")
print(f"  estabilidade serie : {np.mean(sims_S):.4f} (SD {np.std(sims_S):.4f}, n={len(sims_S)})")


# ------------------------------------------------------------------ 6. estudos de caso
cases = {
    "true_positive_high_conf": np.where((yte == 1) & (pred == 1) & (prob > 0.8))[0],
    "true_negative_high_conf": np.where((yte == 0) & (pred == 0) & (prob < 0.2))[0],
    "false_positive": np.where((yte == 0) & (pred == 1))[0],
    "false_negative": np.where((yte == 1) & (pred == 0))[0],
}
print(f"\n{'='*62}\n6. ESTUDOS DE CASO\n{'='*62}")
print(f"  contagens: TP p>0,8 {len(cases['true_positive_high_conf'])} · "
      f"TN p<0,2 {len(cases['true_negative_high_conf'])} · FP {fp} · FN {fn_}")
rows = []
for name, idx in cases.items():
    if len(idx) == 0:
        continue
    i = idx[0]
    igS, igF, b0, _ = IG[i]
    n = int(Kte[i].sum())
    nn = M[ite][i][:n]
    zf = Fte[i]
    print(f"\n  {name}: indice {i}, {files_te[i]}, p={prob[i]:.3f}, {n} batimentos")
    print(f"    IG total: serie {igS.sum():+.4f}  resumo {igF.sum():+.4f}  "
          f"(f(base)={b0:.3f}; fracao serie {share[i]:.2f})")
    print(f"    {'feature':9s} {'bruto':>9s} {'z':>7s} {'|grad|/max':>11s} {'IG':>9s}")
    gmax = gF[i].max()
    for j in np.argsort(-np.abs(igF)):
        print(f"    {FEATURES[j]:9s} {Xf[ite][i][j]:9.2f} {zf[j]:+7.2f} "
              f"{gF[i][j]/gmax:11.2f} {igF[j]:+9.4f}")
    jmax = int(np.argmax(np.abs(igS[:n])))
    print(f"    batimento com maior |IG|: #{jmax+1} ({nn[jmax]:.0f} ms, IG {igS[jmax]:+.4f}); "
          f"intervalo min {nn.min():.0f} no #{int(nn.argmin())+1}")
    rows.append(dict(case=name, idx=i, file=files_te[i], prob=prob[i],
                     ig_series=igS.sum(), ig_summary=igF.sum(), f_base=b0,
                     **{f"ig_{k}": v for k, v in zip(FEATURES, igF)}))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    o = np.argsort(np.abs(igF))
    a1.barh([FEATURES[j] for j in o], igF[o],
            color=["#e74c3c" if v > 0 else "#3498db" for v in igF[o]])
    a1.axvline(0, color="k", lw=0.8)
    a1.set_xlabel("Integrated gradients (summary features)", fontweight="bold")
    a1.set_title(f"Summary branch, total {igF.sum():+.3f}", fontweight="bold")
    a1.grid(axis="x", alpha=0.3)
    v = np.abs(igS[:n]).max() + 1e-12
    a2.plot(np.arange(1, n + 1), nn, color="0.6", lw=1, zorder=1)
    sc = a2.scatter(np.arange(1, n + 1), nn, c=igS[:n], cmap="coolwarm",
                    vmin=-v, vmax=v, s=45, edgecolors="k", lw=0.4, zorder=2)
    plt.colorbar(sc, ax=a2, label="Integrated gradients per beat")
    a2.set_xlabel("Beat", fontweight="bold"); a2.set_ylabel("NN interval (ms)", fontweight="bold")
    a2.set_title(f"NN series branch, total {igS.sum():+.3f}", fontweight="bold")
    a2.grid(alpha=0.3)
    fig.suptitle(f"{name.replace('_', ' ')} · {files_te[i]} · "
                 f"P(stress) = {prob[i]:.3f}", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{a.out}/{name}_G.png", dpi=300); plt.close()
pd.DataFrame(rows).to_csv(f"{a.out}/case_studies_G.csv", index=False)


# ------------------------------------------------------------------ 7. latencia
one = [Ste[:1], Kte[:1], Fte[:1]]
model.predict(one, verbose=0)
t0 = time.time()
for _ in range(1000):
    model.predict(one, verbose=0)
print(f"\n{'='*62}\n7. LATENCIA\n{'='*62}")
print(f"  model.predict, 1 janela, 1000 corridas: {(time.time()-t0):.2f} ms por janela")

pd.DataFrame([dict(acc=acc, auc=roc_auc_score(yte, prob), tn=tn, fp=fp, fn=fn_, tp=tp,
                   faithfulness=acc - acc_m, stab_summary=np.mean(sims_F),
                   stab_series=np.mean(sims_S), f_base_mean=f0.mean(),
                   series_share_mean=share.mean(),
                   **{f"abl_{k}_acc": v[0] for k, v in abl.items()},
                   **{f"abl_{k}_auc": v[1] for k, v in abl.items()},
                   **{f"imp_{k}": v for k, v in zip(FEATURES, imp_n)})]
             ).to_csv(f"{a.out}/summary_G.csv", index=False)
print(f"\nFiguras e CSVs em {a.out}/")
