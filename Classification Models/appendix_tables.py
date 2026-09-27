#!/usr/bin/env python3
"""
Regenera as Tabelas A.1 e A.2 do apendice (duracao do ECG por participante e
condicao) com as duracoes reais (amostras / fs) do time_duration_v6.csv.

    python3 appendix_tables.py --tex .../appendices/SupportWork.tex \
        --durations "../Data Processing/time_duration_v6.csv" --out ../drafts/appendix_tables_v6.tex

Le as duas tabelas do SupportWork.tex tal como estao e substitui so os valores
de duracao e os totais, mantendo a formatacao. As duracoes antigas vinham do
time_duration.csv, que tinha quatro erros (S2 baseline, S5 I/T trocados,
S12 T) e desvios de -2% a +7% no WESAD.
"""
import argparse
import re

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--tex", required=True)
p.add_argument("--durations", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

dur = pd.read_csv(a.durations, header=None, names=["f", "dur", "label", "fs"])
D = dict(zip(dur.f.str.strip(), dur.dur))
tex = open(a.tex, encoding="utf-8").read()

t1 = tex[tex.index(r"\begin{table}"):tex.index(r"\label{neutral_state_dataset}")]
rest = tex[tex.index(r"\label{neutral_state_dataset}"):]
t2 = rest[rest.index(r"\begin{table}"):rest.index(r"\label{stress_state_dataset}")]

fmt = lambda x: f"{x:.3f}"
used = {"neutral": [], "stress": []}

# ---------------------------------------------------------------- Tabela A.1
out1, ds = [], None
for line in t1.split("\n"):
    if "{WESAD}" in line:
        ds = "WESAD"
    elif "{SWELL-KW}" in line:
        ds = "SWELL"
    m = re.search(r"&\s*(\d+)\s*&\s*([\d.]+)", line)
    if m and ds and not line.startswith("Total"):
        n = int(m.group(1))
        f = f"S{n}_baseline_ECG_values.csv" if ds == "WESAD" else f"S{n}_neutral.csv"
        v = D[f]
        used["neutral"].append(v)
        line = line[:m.start(2)] + fmt(v) + line[m.end(2):]
    out1.append(line)
t1n = "\n".join(out1)
tot1 = sum(used["neutral"])
t1n = re.sub(r"(Total\s*&\s*\d+\s*&\s*)[\d.]+", lambda m: m.group(1) + fmt(tot1), t1n)

# ---------------------------------------------------------------- Tabela A.2
cond = {"Interruptions": "stress_I", "Time Pressure": "stress_T"}


def fname(n, stressor, ds):
    if ds == "WESAD":
        return f"S{n}_stress_ECG_values.csv"
    return f"S{n}_{cond[stressor]}.csv"


out2, left_ds = [], None
for line in t2.split("\n"):
    if "{WESAD}}" in line:
        left_ds = "WESAD"
    elif "{SWELL-KW}}" in line and line.startswith(r"\multicolumn{1}{|c|}{\multirow"):
        left_ds = "SWELL"
    # coluna da esquerda: \multicolumn{1}{c|}{N[ - Stressor]} & \multicolumn{1}{c|}{valor}
    m = re.search(r"\\multicolumn\{1\}\{c\|\}\{(\d+)(?: - (Interruptions|Time Pressure))?\}\s*&\s*"
                  r"\\multicolumn\{1\}\{c\|\}\{([\d.]+)\}", line)
    if m and left_ds:
        v = D[fname(int(m.group(1)), m.group(2), left_ds)]
        used["stress"].append(v)
        line = line[:m.start(3)] + fmt(v) + line[m.end(3):]
    # coluna da direita (sempre SWELL): N - Stressor & valor
    m = re.search(r"&\s*(\d+) - (Interruptions|Time Pressure)\s*&\s*([\d.]+)", line)
    if m:
        v = D[fname(int(m.group(1)), m.group(2), "SWELL")]
        used["stress"].append(v)
        line = line[:m.start(3)] + fmt(v) + line[m.end(3):]
    out2.append(line)
t2n = "\n".join(out2)
tot2 = sum(used["stress"])
t2n = re.sub(r"(\\multicolumn\{4\}\{\|c\|\}\{Total\}\s*&\s*\d+\s*&\s*)[\d.]+",
             lambda m: m.group(1) + fmt(tot2), t2n)

w = dur[dur.f.str.contains("ECG_values")]
wb, ws = w[w.label == 0].dur, w[w.label == 1].dur
with open(a.out, "w", encoding="utf-8") as fo:
    fo.write("% ===== Tabela A.1: substitui desde \\begin{table} ate antes de \\label{neutral_state_dataset}\n")
    fo.write(t1n + "\n\n")
    fo.write("% ===== Tabela A.2: substitui desde \\begin{table} ate antes de \\label{stress_state_dataset}\n")
    fo.write(t2n + "\n")
print(f"neutral: {len(used['neutral'])} gravacoes, {tot1:.3f} s = {tot1/3600:.1f} h")
print(f"stress : {len(used['stress'])} gravacoes, {tot2:.3f} s = {tot2/3600:.1f} h  "
      f"(+{(tot2/tot1-1)*100:.1f}% face ao neutral)")
print(f"WESAD baseline {wb.min():.0f}-{wb.max():.0f} s, stress {ws.min():.0f}-{ws.max():.0f} s")
print(f"-> {a.out}")
