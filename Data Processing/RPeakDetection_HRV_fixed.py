#!/usr/bin/env python3
"""
ETL de ECG -> features HRV, com o filtro de artefactos corrigido.

Substitui RPeakDetection_HRV_RemoteMachine.py. Diferencas:

  1. O FILTRO ESTAVA PARTIDO. O original fazia
         for g in rr_interval:
             if g < 0.5 or g > 1.2: rr_interval.remove(g)
     que muta a lista durante a iteracao: o elemento a seguir a cada remocao
     nunca chega a ser testado. Resultado: 24% das janelas ficaram com
     intervalos fora da gama que a tese diz ter filtrado (nni_min ate 14 ms,
     nni_max ate 5465 ms, hr_mean ate 258 bpm).

  2. INTERPOLA EM VEZ DE APAGAR. Apagar um intervalo cria uma diferenca falsa
     entre os batimentos vizinhos, e RMSSD/NN50/pNN50 sao calculados a partir
     dessas diferencas. Os intervalos marcados sao substituidos por
     interpolacao linear a partir dos vizinhos validos.

  3. GAMA MAIS LARGA POR OMISSAO (300-2000 ms = 30-200 bpm, em vez de
     500-1200 ms = 50-120 bpm). O TSST do WESAD leva a frequencia acima de
     120 bpm, por isso a gama antiga apagava dados reais precisamente nas
     janelas mais stressadas. Configuravel em --bounds; o script reporta
     quantos intervalos cairiam na gama antiga, para efeitos de comparacao.

  4. CRITERIO RELATIVO. Alem da gama absoluta, marca intervalos que se afastem
     mais de --rel-thresh da mediana local (janela de 5). Apanha batimentos
     ectopicos que caem dentro da gama absoluta.

  5. DESCARTA JANELAS DEMASIADO CORROMPIDAS em vez de as interpolar quase por
     inteiro (--max-bad). Uma janela com 95% de intervalos marcados nao e um
     estado fisiologico -- e a amostra que alimentava as Figuras 16 e 18.

  6. Corrige o TabError da L213 e o shadowing de `time` (import time vs
     time = ecg_duration/samples), que impediam o original de sequer compilar.

  7. Caminhos por argumento em vez de /home/gsd/ hard-coded; escreve o
     cabecalho; le com pandas (muito mais rapido que np.loadtxt); e recusa
     sobrescrever sem --force (o original abria em modo 'a' e ia acumulando).

Saida: CSV de 12 colunas, na ordem exacta que Classification Models/Data.py
espera via iloc[:, 2:10].

Exemplo:
    python3 RPeakDetection_HRV_fixed.py \
        --datasets "/home/eduarda/Desktop/Tese 2026/Datasets" \
        --durations "/home/eduarda/Desktop/Tese 2026/Data Processing/time_duration.csv" \
        --out /home/eduarda/Desktop/ProcessedECGData_v5.csv
"""
import argparse
import csv
import os
import sys
import time as time_mod  # nunca reatribuir: o original perdia o modulo aqui
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")  # sem display na maquina de calculo
import matplotlib.pyplot as plt

plt.ioff()

import biosppy
from pyhrv.frequency_domain import welch_psd
from pyhrv.hrv import hrv

HEADER = ["file", "label", "hr_mean", "nni_mean", "nni_min", "nni_max",
          "sdnn", "rmssd", "nn50", "pnn50", "LF/HF ratio", "Total power"]

LEGACY_BOUNDS = (0.5, 1.2)  # a gama do script original, so para contagem


def parse_args():
    p = argparse.ArgumentParser(
        description="ETL de ECG para features HRV, com filtro de artefactos corrigido.")
    p.add_argument("--datasets", required=True,
                   help="pasta com os ficheiros de ECG em bruto")
    p.add_argument("--durations", required=True,
                   help="CSV: ficheiro,duracao_s,label,fs")
    p.add_argument("--out", required=True, help="CSV de saida")
    p.add_argument("--window", type=float, default=30.0,
                   help="duracao da janela em segundos (default 30)")
    p.add_argument("--bounds", type=float, nargs=2, default=[0.3, 2.0],
                   metavar=("LO", "HI"),
                   help="gama de plausibilidade dos intervalos RR em segundos "
                        "(default 0.3 2.0 = 30-200 bpm). Usar 0.5 1.2 para "
                        "reproduzir o comportamento pretendido pelo original.")
    p.add_argument("--rel-thresh", type=float, default=0.20,
                   help="desvio relativo maximo face a mediana local (default 0.20)")
    p.add_argument("--max-bad", type=float, default=0.20,
                   help="fraccao maxima de intervalos corrigidos antes de "
                        "descartar a janela (default 0.20)")
    p.add_argument("--min-intervals", type=int, default=10,
                   help="minimo de intervalos validos para aceitar a janela")
    p.add_argument("--use-pyhrv", action="store_true",
                   help="usar pyhrv.hrv() em vez do calculo directo. ~13000x "
                        "mais lento (3h em vez de ~6 min) e da os mesmos "
                        "valores; existe so para verificacao.")
    p.add_argument("--series-out", default=None,
                   help=".npz onde guardar a serie de intervalos NN de cada "
                        "janela, alinhada linha-a-linha com o CSV. E a entrada "
                        "sequencial real (30-50 batimentos por janela); as 8 "
                        "features resumo nao tem ordem temporal entre si.")
    p.add_argument("--force", action="store_true",
                   help="sobrescrever o CSV de saida se ja existir")
    p.add_argument("--limit", type=int, default=None,
                   help="processar apenas os primeiros N ficheiros (para testar)")
    return p.parse_args()


def r_peak_times(segment, fs):
    """Instantes dos picos R, em segundos relativos ao inicio do segmento."""
    idx = biosppy.signals.ecg.ecg(segment, sampling_rate=fs, show=False)[2]
    return np.asarray(idx, dtype=float) / float(fs)


def rr_from_peaks(peak_times):
    """Intervalos RR em segundos."""
    return np.diff(np.asarray(peak_times, dtype=float))


def local_median(x, k=5):
    """Mediana movel, com as pontas preenchidas por reflexao."""
    if len(x) < k:
        return np.full_like(x, np.median(x))
    pad = k // 2
    xp = np.pad(x, pad, mode="reflect")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def clean_rr(rr, lo, hi, rel_thresh):
    """
    Marca e interpola intervalos implausiveis.

    Devolve (rr_corrigido, mascara_marcados). A correccao e por interpolacao
    linear a partir dos vizinhos validos -- nao por remocao, que destruiria as
    diferencas sucessivas de que dependem RMSSD, NN50 e pNN50.
    """
    rr = np.asarray(rr, dtype=float)
    if len(rr) == 0:
        return rr, np.zeros(0, dtype=bool)

    bad = (rr < lo) | (rr > hi)

    # Criterio relativo: apanha ectopicos que caem dentro da gama absoluta.
    # A mediana local e calculada so sobre os intervalos ainda validos.
    if (~bad).sum() >= 3:
        med = local_median(np.where(bad, np.median(rr[~bad]), rr))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(rr - med) / med
        bad |= np.nan_to_num(rel, nan=0.0) > rel_thresh

    good = ~bad
    if good.sum() < 2:
        return rr, bad  # nada em que interpolar; quem chama descarta

    idx = np.arange(len(rr))
    out = rr.copy()
    out[bad] = np.interp(idx[bad], idx[good], rr[good])
    return out, bad


def to_nni_ms(rr_s):
    """
    Converte intervalos RR de segundos para NN em ms exactamente como o pyHRV.

    pyhrv.utils.nn_format faz `[int(x * 1000) for x in nni]` quando max < 10,
    ou seja TRUNCA para milissegundos inteiros. Replicar a truncatura (e nao
    arredondar) e o que garante que as features saem bit-a-bit iguais as da V4.
    """
    nni = np.asarray(rr_s, dtype="float64")
    if len(nni) and np.max(nni) < 10:
        nni = np.asarray([int(x * 1000) for x in nni], dtype="float64")
    return nni


def compute_features(rr_s, fs, use_pyhrv=False):
    """
    As 12 colunas de saida a partir dos intervalos RR (em segundos).

    Por omissao usa o calculo directo: verificado como identico ao
    pyhrv.hrv() em todas as 8 features de tempo (o rmssd difere 1e-14, ruido
    de virgula flutuante) e ~13000x mais rapido -- pyhrv.hrv() gasta 1.7 s por
    janela para calcular uma media e um desvio-padrao sobre ~25 numeros.
    --use-pyhrv forca o caminho lento original, para verificacao.
    """
    if use_pyhrv:
        d = hrv(nni=rr_s, sampling_rate=fs, show=False,
                plot_tachogram=False, plot_ecg=False).as_dict()
        return [d["hr_mean"], d["nni_mean"], d["nni_min"], d["nni_max"],
                d["sdnn"], d["rmssd"], d["nn50"], d["pnn50"],
                d["fft_ratio"], d["fft_total"]]

    nni = to_nni_ms(rr_s)
    diff = np.diff(nni)
    nn50 = int(np.sum(np.abs(diff) > 50))

    # welch_psd(mode='dev') devolve exactamente os mesmos fft_ratio/fft_total
    # que hrv() por ~1 ms. Nao sao usados pelos modelos (Data.py le so as
    # colunas 2:10) mas sao precisos para refazer a comparacao F1 vs F2.
    try:
        w = welch_psd(nni=rr_s, show=False, mode="dev")[0]
        fft_ratio, fft_total = float(w["fft_ratio"]), float(w["fft_total"])
    except Exception:
        fft_ratio = fft_total = float("nan")

    return [float(np.mean(60000.0 / nni)), float(np.mean(nni)),
            float(np.min(nni)), float(np.max(nni)),
            float(np.std(nni, ddof=1)), float(np.sqrt(np.mean(diff ** 2))),
            nn50, float(nn50 / len(diff) * 100.0),
            fft_ratio, fft_total]


def split_ecg(signal, duration_s, window_s):
    """Divide o sinal em segmentos de ~window_s. Mesma logica do original."""
    n = len(signal)
    sec_per_sample = duration_s / n
    samples_per_window = int(window_s // sec_per_sample)
    if samples_per_window <= 0:
        return []
    n_windows = n // samples_per_window
    if n_windows < 1:
        return []
    return np.array_split(signal, n_windows)


def main():
    a = parse_args()
    lo, hi = a.bounds

    if os.path.exists(a.out) and not a.force:
        sys.exit(f"ERRO: {a.out} ja existe. Usar --force para sobrescrever.\n"
                 f"      (o script original abria em modo 'a' e ia acumulando "
                 f"linhas a cada execucao)")

    rows = []
    with open(a.durations, "r") as f:
        for r in csv.reader(f):
            if len(r) < 4 or not r[0].strip():
                continue
            rows.append((r[0].strip(), float(r[1]), int(r[2]), int(r[3])))
    if a.limit:
        rows = rows[:a.limit]

    print(f"Ficheiros a processar : {len(rows)}")
    print(f"Gama de plausibilidade: {lo * 1000:.0f}-{hi * 1000:.0f} ms "
          f"({60 / hi:.0f}-{60 / lo:.0f} bpm)")
    print(f"Desvio relativo max.  : {a.rel_thresh * 100:.0f}% da mediana local")
    print(f"Descartar janela acima: {a.max_bad * 100:.0f}% de intervalos corrigidos")
    print(f"Saida                 : {a.out}\n")

    # Estatisticas por classe, que a supervisora pediu explicitamente
    # ("convem contar quantos intervalos sao removidos em cada classe").
    st = {c: dict(win_ok=0, win_drop_bad=0, win_drop_few=0, win_err=0,
                  rr_total=0, rr_fixed=0, rr_legacy=0) for c in (0, 1)}

    series = []  # uma serie NN por linha do CSV, pela mesma ordem
    t0 = datetime.now()
    with open(a.out, "w", encoding="UTF8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(HEADER)

        for fi, (fname, duration, label, fs) in enumerate(rows, 1):
            path = os.path.join(a.datasets, fname)
            if not os.path.exists(path):
                print(f"[{fi}/{len(rows)}] {fname}: AUSENTE, saltado")
                continue

            # pandas em vez de np.loadtxt: ordens de grandeza mais rapido
            # em ficheiros de milhoes de linhas.
            sig = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=float)
            segments = split_ecg(sig, duration, a.window)
            s = st[label]
            kept = 0

            for seg in segments:
                try:
                    rr = rr_from_peaks(r_peak_times(seg, fs))
                except Exception:
                    s["win_err"] += 1
                    continue

                if len(rr) < a.min_intervals:
                    s["win_drop_few"] += 1
                    continue

                rr_c, bad = clean_rr(rr, lo, hi, a.rel_thresh)
                s["rr_total"] += len(rr)
                s["rr_fixed"] += int(bad.sum())
                s["rr_legacy"] += int(((rr < LEGACY_BOUNDS[0]) |
                                       (rr > LEGACY_BOUNDS[1])).sum())

                if bad.mean() > a.max_bad:
                    s["win_drop_bad"] += 1
                    continue
                if (~bad).sum() < a.min_intervals:
                    s["win_drop_few"] += 1
                    continue

                try:
                    feats = compute_features(rr_c, fs, a.use_pyhrv)
                except Exception:
                    s["win_err"] += 1
                    continue

                w.writerow([fname, label] + feats)
                # A serie NN em si, alinhada linha-a-linha com o CSV. E esta a
                # unica sequencia com ordem temporal real nos dados: batimento
                # 1, 2, 3... As 8 features resumo nao tem ordem nenhuma entre si.
                series.append(to_nni_ms(rr_c).astype(np.float32))
                s["win_ok"] += 1
                kept += 1

            # welch_psd/pyhrv deixam figuras matplotlib abertas a cada chamada;
            # sem isto o processo cresce sem limite ao longo dos 96 ficheiros.
            plt.close("all")
            fout.flush()
            print(f"[{fi}/{len(rows)}] {fname:<32} label={label} "
                  f"{kept}/{len(segments)} janelas mantidas")

    print(f"\nTempo total: {datetime.now() - t0}")

    # Series NN em matriz (N, maxlen) com padding a zero, mais o comprimento
    # real de cada uma. O zero e seguro como padding porque nenhum intervalo NN
    # valido e zero -- permite usar Masking(mask_value=0.0) no modelo.
    if a.series_out and series:
        maxlen = max(len(x) for x in series)
        M = np.zeros((len(series), maxlen), dtype=np.float32)
        L = np.array([len(x) for x in series], dtype=np.int32)
        for i, x in enumerate(series):
            M[i, :len(x)] = x
        np.savez_compressed(a.series_out, nni=M, length=L)
        print(f"\nSeries NN: {a.series_out}")
        print(f"  matriz {M.shape} (padding a zero)   comprimentos: "
              f"min {L.min()}, mediana {int(np.median(L))}, max {L.max()}")
        print(f"  NOTA: a ordem das linhas coincide com a do CSV.")

    print(f"\n{'=' * 66}\nESTATISTICAS POR CLASSE\n{'=' * 66}")
    print(f"{'':<34}{'nao-stress':>15}{'stress':>15}")
    labels = [("janelas mantidas", "win_ok"),
              ("descartadas (>%d%% corrigido)" % (a.max_bad * 100), "win_drop_bad"),
              ("descartadas (poucos intervalos)", "win_drop_few"),
              ("descartadas (erro)", "win_err"),
              ("intervalos RR analisados", "rr_total"),
              ("intervalos RR corrigidos", "rr_fixed")]
    for lab, k in labels:
        print(f"  {lab:<32}{st[0][k]:>15}{st[1][k]:>15}")
    for c, nm in ((0, "nao-stress"), (1, "stress")):
        t, f_ = st[c]["rr_total"], st[c]["rr_fixed"]
        lg = st[c]["rr_legacy"]
        print(f"\n  {nm}:")
        print(f"    taxa de correccao          {f_ / t * 100:.2f}%" if t else "    sem dados")
        if t:
            print(f"    cairiam na gama 500-1200   {lg / t * 100:.2f}%  "
                  f"<- o que a tese diz ter filtrado")

    total_ok = st[0]["win_ok"] + st[1]["win_ok"]
    print(f"\n  TOTAL de janelas no CSV: {total_ok}  (a V4 tinha 6596)")
    print(f"\nEscrito: {a.out}")
    print("Apontar Classification Models/main.py:13 para este ficheiro.")


if __name__ == "__main__":
    main()
