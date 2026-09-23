#!/usr/bin/env python3
"""Protocolo personalizado: o que vale calibrar ao proprio utilizador.

O LOSO da Seccao 4.2 mostra AUC 0,93-0,95 com exatidao de 37,5% a 100% e
especificidade de 0% a 100% entre sujeitos: a ordenacao e quase perfeita e a
decisao e inutil. A hipotese e que falta o limiar por pessoa, nao o sinal.
Esta experiencia testa isso com tres condicoes sobre AS MESMAS janelas de teste:

  A  modelo de populacao (treinado nos outros 14 sujeitos), limiar fixo 0,5
  B  o MESMO modelo, limiar escolhido nas janelas de calibracao do sujeito
     (indice de Youden na ROC de calibracao) -- nao ha re-treino
  C  modelo re-treinado com os outros 14 MAIS as janelas de calibracao do
     sujeito, limiar fixo 0,5

Divisao dentro do sujeito: TEMPORAL, nao aleatoria. Por cada (sujeito, classe),
as primeiras 50% das janelas pela ordem do ficheiro vao para calibracao e as
restantes para teste. Uma divisao aleatoria intercalaria janelas vizinhas no
tempo entre calibracao e teste, o que e fuga de sessao e daria um resultado
optimista.

    PYTHONPATH=. python3 personalised_protocol.py [--models D,LogReg,RF] [--frac 0.5]
"""
import argparse, contextlib, io, os, warnings
warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib; matplotlib.use("Agg")   # antes de importar Models (plt.show)
import numpy as np, pandas as pd, tensorflow as tf
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from Data import normalize_data
from sklearn.utils.class_weight import compute_class_weight

from Models import build_model

FEATURES = ["hr_mean", "nni_mean", "nni_min", "nni_max",
            "sdnn", "rmssd", "nn50", "pnn50"]

p = argparse.ArgumentParser()
p.add_argument("--csv", default="/home/eduarda/Desktop/ProcessedECGData_v5.csv")
p.add_argument("--models", default="D,LogReg,RF")
p.add_argument("--frac", type=float, default=0.5,
               help="fraccao das janelas de cada classe do sujeito usada para "
                    "calibrar (default 0,5)")
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--out", default="results_personalised.csv")
a = p.parse_args()

d = pd.read_csv(a.csv).dropna(subset=FEATURES + ["label"])
d = d[d["file"].str.contains("ECG_values")].reset_index(drop=True)
X = d[FEATURES].to_numpy(float)
y = d["label"].to_numpy().astype(int)
subj = d["file"].str.extract(r"^(S\d+)")[0].to_numpy()
subjects = sorted(set(subj), key=lambda s: int(s[1:]))
print(f"WESAD: {len(d)} janelas, {len(subjects)} sujeitos, "
      f"calibracao = {a.frac:.0%} das janelas de cada classe (divisao temporal)\n")


def split_subject(mask):
    """Indices de calibracao e de teste de um sujeito, temporal e por classe."""
    cal, tst = [], []
    idx = np.where(mask)[0]                     # ja em ordem de ficheiro
    for c in (0, 1):
        ic = idx[y[idx] == c]
        k = max(1, int(round(len(ic) * a.frac)))
        k = min(k, len(ic) - 1)                 # deixa sempre >=1 para teste
        cal += list(ic[:k]); tst += list(ic[k:])
    return np.array(sorted(cal)), np.array(sorted(tst))


def metrics(yt, prob, thr):
    pred = (prob > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(yt, pred) * 100,
                sens=tp / (tp + fn) * 100 if tp + fn else np.nan,
                spec=tn / (tn + fp) * 100 if tn + fp else np.nan,
                auc=roc_auc_score(yt, prob) if len(set(yt)) > 1 else np.nan,
                thr=thr)


def youden(yt, prob):
    fpr, tpr, thr = roc_curve(yt, prob)
    return float(thr[np.argmax(tpr - fpr)])


def fit_deep(xtr, ytr, xev, seed, epochs):
    """Treino do Model D com os hiperparametros da Seccao 3.6.

    Replica o bloco de treino do train_evaluate_model() do Models.py -- pesos
    balanceados com o multiplicador de 1,2 na classe stress, early stopping e
    ReduceLROnPlateau sobre val_loss, validation_split=0,2 -- mas sem a parte de
    avaliacao dessa funcao, que chama classification_report() com dois nomes de
    classe e rebenta quando as predicoes de um fold saem todas da mesma classe.
    """
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed); np.random.seed(seed)
    perm = np.random.permutation(len(ytr))      # validation_split corta os ultimos 20%
    xtr, ytr = xtr[perm], ytr[perm]

    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    cw = dict(enumerate(cw)); cw[1] = cw.get(1, 1.0) * 1.2

    m = build_model("model5_cnn_transformer_hybrid_v2", (xtr.shape[1], 1))
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
              loss="binary_crossentropy", metrics=["accuracy"])
    m.fit(xtr, ytr, batch_size=32, epochs=epochs, validation_split=0.2,
          class_weight=cw, verbose=0,
          callbacks=[
              tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                               restore_best_weights=True),
              tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                   patience=15, min_lr=1e-7)])
    return m.predict(xev, verbose=0).flatten()


def fit_flat(kind, xtr, ytr, xev):
    mk = {"LogReg": lambda: LogisticRegression(max_iter=2000),
          "RF": lambda: RandomForestClassifier(n_estimators=200, random_state=42,
                                               class_weight="balanced", n_jobs=-1)}[kind]
    return mk().fit(xtr, ytr).predict_proba(xev)[:, 1]


rows = []
for k, s in enumerate(subjects):
    mask = subj == s
    cal, tst = split_subject(mask)
    others = np.where(~mask)[0]
    print(f"--- {s}: calibracao {len(cal)} janelas "
          f"({y[cal].sum()} stress), teste {len(tst)} ({y[tst].sum()} stress)")

    # Condicoes A e B: modelo de populacao. Normaliza com os outros 14.
    ev = np.concatenate([cal, tst])
    with contextlib.redirect_stdout(io.StringIO()):
        xtr_A, xev_A = normalize_data(X[others], X[ev])
    # Condicao C: o treino inclui as janelas de calibracao do sujeito.
    trC = np.concatenate([others, cal])
    with contextlib.redirect_stdout(io.StringIO()):
        xtr_C, xtst_C = normalize_data(X[trC], X[tst])

    for name in [m.strip() for m in a.models.split(",")]:
        if name == "D":
            pA = fit_deep(xtr_A.reshape(len(xtr_A), -1, 1), y[others],
                          xev_A.reshape(len(xev_A), -1, 1), 42 + k, a.epochs)
            pC = fit_deep(xtr_C.reshape(len(xtr_C), -1, 1), y[trC],
                          xtst_C.reshape(len(xtst_C), -1, 1), 42 + k, a.epochs)
        else:
            pA = fit_flat(name, xtr_A, y[others], xev_A)
            pC = fit_flat(name, xtr_C, y[trC], xtst_C)

        p_cal, p_tst = pA[:len(cal)], pA[len(cal):]
        thr = youden(y[cal], p_cal)
        for cond, r in (("A_population", metrics(y[tst], p_tst, 0.5)),
                        ("B_threshold", metrics(y[tst], p_tst, thr)),
                        ("C_retrained", metrics(y[tst], pC, 0.5))):
            rows.append(dict(model=name, subject=s, condition=cond,
                             n_cal=len(cal), n_test=len(tst), **r))
        print(f"    {name:6s} A acc {rows[-3]['acc']:5.1f} spec {rows[-3]['spec']:5.1f} | "
              f"B acc {rows[-2]['acc']:5.1f} spec {rows[-2]['spec']:5.1f} (thr {thr:.2f}) | "
              f"C acc {rows[-1]['acc']:5.1f} spec {rows[-1]['spec']:5.1f}")

r = pd.DataFrame(rows)
r.to_csv(a.out, index=False)
print(f"\nGuardado em {a.out}")

for name in r.model.unique():
    g = r[r.model == name]
    print(f"\n===== {name}  (15 sujeitos, media +/- SD entre sujeitos)")
    print(g.groupby("condition")[["acc", "sens", "spec", "auc"]]
          .agg(["mean", "std"]).round(2).to_string())
    piv = g.pivot(index="subject", columns="condition", values="acc")
    for x, z in (("B_threshold", "A_population"), ("C_retrained", "A_population"),
                 ("B_threshold", "C_retrained")):
        for met in ("acc", "spec"):
            pv = g.pivot(index="subject", columns="condition", values=met)
            dif = pv[x] - pv[z]
            t, pval = stats.ttest_rel(pv[x], pv[z])
            print(f"  {x:12s} - {z:12s} {met:4s}: {dif.mean():+6.2f} pp "
                  f"+/- {dif.std(ddof=1):5.2f}  p={pval:.4f}"
                  f"{'  *' if pval < 0.05 else ''}")
