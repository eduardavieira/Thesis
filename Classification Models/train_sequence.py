#!/usr/bin/env python3
"""
Modelo de duas entradas: serie de intervalos NN + features resumo.

    python3 train_sequence.py --csv .../ProcessedECGData_v5.csv \
                              --series .../NNSeries_v5.npz --seeds 5

Motivacao (ponto 3 da supervisora). Os modelos A-D recebem as 8 features
resumo com forma (8, 1), tratando-as como 8 instantes de uma sequencia. Nao
sao: nao existe ordem temporal entre hr_mean, sdnn e pnn50 -- a ordem e a das
colunas do CSV. Um Conv1D(kernel=3) sobre isso impoe partilha de pesos entre
features adjacentes e um prior de localidade arbitrarios, o que e pior do que
nao ter viés nenhum. E por isso que uma Random Forest, que nao assume ordem,
bate as quatro arquiteturas (63,72% vs 61,95% da melhor).

A serie NN e a unica sequencia com ordem real nos dados: batimento 1, 2, 3...
Tipicamente 25-50 por janela de 30 s.

Variantes (--arch):
  seq      so a serie NN            -- a sequencia sozinha chega?
  both     serie + resumo           -- a proposta da supervisora
  summary  so o resumo (MLP)        -- controlo: e o que os modelos A-D fazem
                                       na pratica, mas sem o viés do reshape

O ramo sequencial e Conv1D com pooling medio mascarado. As series tem
comprimentos diferentes (14 a 73 batimentos) e vao com padding a zero; como o
Conv1D nao respeita mascaras do Keras, as posicoes de padding sao zeradas
apos a convolucao e a media e feita sobre o numero de posicoes reais.

Um BiLSTM foi testado primeiro e abandonado: >10 min por seed em CPU contra
~1,5 min do convolucional. O Conv1D e tambem a comparacao cientificamente
mais limpa, por manter a familia de arquitectura dos Modelos C/D e mudar
apenas a representacao de entrada.
"""
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")  # Models.py:176 faz plt.show(); sem isto bloqueia

import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers as L
from tensorflow.keras import Model

FEATURES = ["hr_mean", "nni_mean", "nni_min", "nni_max",
            "sdnn", "rmssd", "nn50", "pnn50"]


def build(arch, seq_len, n_feat):
    inputs, heads = [], []

    if arch in ("seq", "both"):
        si = L.Input(shape=(seq_len, 1), name="nni_series")
        mi = L.Input(shape=(seq_len, 1), name="nni_mask")
        inputs += [si, mi]
        # Conv1D e nao LSTM: em CPU o recorrente sobre 73 passos leva >10 min
        # por seed, o convolucional ~2. E cientificamente e a comparacao certa
        # -- mantem a familia de arquitectura dos Modelos C/D e muda apenas a
        # representacao de entrada, isolando a variavel que interessa.
        x = L.Conv1D(64, 5, padding="same", activation="relu")(si)
        x = L.BatchNormalization()(x)
        x = L.Dropout(0.2)(x)
        x = L.Conv1D(64, 5, padding="same", activation="relu")(x)
        x = L.BatchNormalization()(x)
        # Pooling medio mascarado: Conv1D nao respeita mascaras do Keras, por
        # isso zeramos as posicoes de padding e dividimos pelo numero de
        # posicoes reais, em vez de pelo comprimento com padding.
        x = L.Multiply()([x, mi])
        tot = L.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
        cnt = L.Lambda(lambda t: tf.reduce_sum(t, axis=1))(mi)
        x = L.Lambda(lambda t: t[0] / (t[1] + 1e-6))([tot, cnt])
        heads.append(x)

    if arch in ("summary", "both"):
        fi = L.Input(shape=(n_feat,), name="summary")
        inputs.append(fi)
        f = L.Dense(64, activation="relu")(fi)
        f = L.BatchNormalization()(f)
        f = L.Dropout(0.3)(f)
        f = L.Dense(32, activation="relu")(f)
        heads.append(f)

    h = heads[0] if len(heads) == 1 else L.Concatenate()(heads)
    h = L.Dense(64, activation="relu")(h)
    h = L.BatchNormalization()(h)
    h = L.Dropout(0.4)(h)
    h = L.Dense(32, activation="relu")(h)
    h = L.Dropout(0.3)(h)
    out = L.Dense(1, activation="sigmoid")(h)
    return Model(inputs, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--series", required=True)
    p.add_argument("--arch", default="both", choices=["seq", "both", "summary"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--out", default="results_sequence.csv")
    p.add_argument("--dataset", default="all", choices=["all", "SWELL", "WESAD"],
                   help="restringir a um dataset. Os dois foram gravados a "
                        "frequencias diferentes (WESAD 700 Hz, SWELL 2048 Hz) "
                        "e tem proporcoes de classe muito diferentes (35,6%% vs "
                        "60,7%% stress), pelo que um modelo pode ganhar exatidao "
                        "so por identificar a origem. Treinar dentro de um "
                        "dataset elimina esse atalho.")
    p.add_argument("--cv", default="random", choices=["random", "group", "loso"],
                   help="random = split 80/20 estratificado (o da tese). "
                        "group = GroupKFold k=5 por sujeito, sem janelas do "
                        "mesmo participante em treino e teste.")
    a = p.parse_args()

    raw = pd.read_csv(a.csv)
    z = np.load(a.series)
    M, Lg = z["nni"], z["length"]
    if len(raw) != len(M):
        raise SystemExit(f"ERRO: CSV tem {len(raw)} linhas e o npz tem {len(M)}. "
                         f"Foram gerados na mesma execucao do ETL?")

    keep = raw[FEATURES + ["label"]].notna().all(axis=1).to_numpy()
    if a.dataset != "all":
        is_wesad = raw["file"].str.contains("ECG_values").to_numpy()
        keep &= (is_wesad if a.dataset == "WESAD" else ~is_wesad)
    raw, M, Lg = raw[keep].reset_index(drop=True), M[keep], Lg[keep]
    Xf = raw[FEATURES].to_numpy(float)
    y = raw["label"].to_numpy().astype(int)

    print(f"{len(raw)} janelas, {y.mean()*100:.2f}% stress")
    print(f"serie NN: {M.shape[1]} passos max, comprimento real "
          f"min {Lg.min()} / mediana {int(np.median(Lg))} / max {Lg.max()}")
    print(f"arquitectura: {a.arch}   cv: {a.cv}")

    # Agrupamento por sujeito. Os prefixos S1..S25 COLIDEM entre WESAD e
    # SWELL, por isso o id tem de incluir o dataset -- sem isso juntam-se
    # pessoas diferentes no mesmo grupo e a avaliacao deixa de ser honesta.
    # pd.Series e nao np.array: arrays numpy de strings nao concatenam com '+'.
    ds = pd.Series(np.where(raw["file"].str.contains("ECG_values"),
                            "WESAD", "SWELL"), index=raw.index)
    groups = (ds + "_" + raw["file"].str.extract(r"^(S\d+)")[0]).to_numpy()

    if a.cv == "loso":
        splits = list(LeaveOneGroupOut().split(np.arange(len(y)), y, groups))
        print(f"LOSO: {len(splits)} folds, um por sujeito\n")
    elif a.cv == "group":
        splits = list(GroupKFold(n_splits=5).split(np.arange(len(y)), y, groups))
        print(f"GroupKFold k=5 por sujeito ({len(np.unique(groups))} sujeitos)\n")
    else:
        splits = [train_test_split(np.arange(len(y)), test_size=0.2,
                                   random_state=s, stratify=y)
                  for s in range(42, 42 + a.seeds)]

    res = []
    for k, (itr, ite) in enumerate(splits):
        seed = 42 + k
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Em GroupKFold os indices saem ordenados por ficheiro. O
        # validation_split do Keras corta os ultimos 20% SEM baralhar, o que
        # daria um conjunto de validacao de um so sujeito e possivelmente de
        # uma so classe. Baralhar resolve. (So afecta o early stopping; o
        # conjunto de teste continua disjunto por sujeito.)
        itr = np.random.permutation(itr)

        # Normalizacao com estatisticas so do treino, e so sobre os valores
        # reais (o padding a zero nao entra na media nem no desvio).
        real = M[itr] > 0
        mu, sd = M[itr][real].mean(), M[itr][real].std()
        def prep_seq(m):
            out = (m - mu) / sd
            return np.where(m > 0, out, 0.0)[..., None]  # padding volta a 0
        def prep_mask(m):
            return (m > 0).astype("float32")[..., None]
        Str, Ste = prep_seq(M[itr]), prep_seq(M[ite])
        Ktr, Kte = prep_mask(M[itr]), prep_mask(M[ite])

        fmu, fsd = Xf[itr].mean(0), Xf[itr].std(0)
        Ftr, Fte = (Xf[itr] - fmu) / fsd, (Xf[ite] - fmu) / fsd
        Ftr, Fte = np.clip(Ftr, -5, 5), np.clip(Fte, -5, 5)

        xtr = {"seq": [Str, Ktr], "summary": [Ftr],
               "both": [Str, Ktr, Ftr]}[a.arch]
        xte = {"seq": [Ste, Kte], "summary": [Fte],
               "both": [Ste, Kte, Fte]}[a.arch]
        ytr, yte = y[itr], y[ite]

        model = build(a.arch, M.shape[1], len(FEATURES))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
            loss="binary_crossentropy", metrics=["accuracy"])

        # Mesmo esquema de pesos de classe dos modelos A-D, incluindo o
        # multiplicador de 1,2 na classe stress, para a comparacao ser justa.
        cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
        cw = dict(enumerate(cw))
        cw[1] *= 1.2

        model.fit(xtr, ytr, validation_split=0.2, epochs=a.epochs, batch_size=64,
                  class_weight=cw, verbose=0,
                  callbacks=[
                      tf.keras.callbacks.EarlyStopping(
                          monitor="val_loss", patience=20, restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(
                          monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7)])

        prob = model.predict(xte, verbose=0).flatten()
        pred = (prob > 0.5).astype(int)
        tn, fp, fn_, tp = confusion_matrix(yte, pred).ravel()
        r = dict(arch=a.arch, seed=seed,
                 acc=accuracy_score(yte, pred) * 100,
                 prec=precision_score(yte, pred, zero_division=0) * 100,
                 rec=recall_score(yte, pred) * 100,
                 spec=tn / (tn + fp) * 100,
                 f1=f1_score(yte, pred),
                 auc=roc_auc_score(yte, prob),
                 prauc=average_precision_score(yte, prob),
                 params=model.count_params(),
                 major=max(yte.mean(), 1 - yte.mean()) * 100)
        r["cv"] = a.cv
        r["n_test_subj"] = len(np.unique(groups[ite]))
        res.append(r)
        pd.DataFrame(res).to_csv(a.out, index=False)
        lab = f"fold {k+1}" if a.cv in ("group","loso") else f"seed {seed}"
        print(f"  {lab}: acc {r['acc']:.2f}  prec {r['prec']:.2f}  "
              f"rec {r['rec']:.2f}  spec {r['spec']:.2f}  AUC {r['auc']:.3f}"
              + (f"  ({r['n_test_subj']} sujeitos no teste)" if a.cv in ("group","loso") else ""))

    df = pd.DataFrame(res)
    print(f"\n{'='*66}\n{a.arch.upper()}  ({a.seeds} seeds)\n{'='*66}")
    print(f"  {'metrica':<13}{'media':>9}{'SD':>8}{'min':>9}{'max':>9}")
    for k, lab in [("acc", "accuracy"), ("prec", "precision"), ("rec", "recall"),
                   ("spec", "specificity"), ("auc", "ROC-AUC")]:
        print(f"  {lab:<13}{df[k].mean():9.3f}{df[k].std():8.3f}"
              f"{df[k].min():9.3f}{df[k].max():9.3f}")
    print(f"\n  parametros: {int(df['params'].iloc[0]):,}")
    print(f"  majority  : {df['major'].mean():.2f}%")
    print(f"\n  Referencias (V5, 5 seeds): RandomForest 63.72 | Model C 61.95 | "
          f"Model D 61.18 | LogReg 59.50")


if __name__ == "__main__":
    main()
