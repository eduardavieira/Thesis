#!/usr/bin/env python3
"""Reescreve o time_duration.csv com a duracao real de cada ficheiro (amostras / fs).

O split_ecg() do ETL calcula o tamanho da janela a partir da duracao listada:
    samples_per_window = 30 / (duracao_listada / n_amostras)
Se a duracao listada nao bate com o ficheiro, as "janelas de 30 s" ficam com
outra duracao real. Casos encontrados em 2026-09-26 (duracao listada / real):

    S2_baseline_ECG_values.csv  1518,3 / 1144,0 s  -> janelas de 22,9 s
    S5_stress_I.csv             1595,0 / 2940,8 s  -> janelas de 55,3 s  (trocada com a do stress_T)
    S5_stress_T.csv             2940,0 / 1595,4 s  -> janelas de 16,3 s
    S12_stress_T.csv            2126,0 /  918,8 s  -> janelas de 13,0 s
    S3_stress_T.csv             2142,0 / 2322,7 s  -> janelas de 32,5 s

Os restantes ficheiros do WESAD tem desvios de -2% a +7% (janelas de 28 a
30,7 s); os do SWELL batem ao milissegundo.

Uso:
    python3 fix_durations.py --datasets ".../Datasets" \
        --durations ".../time_duration.csv" --out time_duration_v6.csv
"""
import argparse
import csv
import os
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--datasets", required=True)
p.add_argument("--durations", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

rows = []
with open(a.durations) as f:
    for r in csv.reader(f):
        if len(r) < 4 or not r[0].strip():
            continue
        name, listed, label, fs = r[0].strip(), float(r[1]), int(r[2]), int(r[3])
        path = os.path.join(a.datasets, name)
        # wc -l: os ficheiros do SWELL tem centenas de MB; uma amostra por linha
        n = int(subprocess.run(["wc", "-l", path], capture_output=True,
                               text=True, check=True).stdout.split()[0])
        real = n / fs
        rows.append((name, real, label, fs))
        if abs(listed / real - 1) > 0.05:
            print(f"{name:<30} listada {listed:9.3f}  real {real:9.3f}  "
                  f"janela efectiva {30 * real / listed:5.1f} s")

with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    for name, real, label, fs in rows:
        w.writerow([name, f"{real:.6f}", label, fs])
print(f"{len(rows)} ficheiros -> {a.out}")
