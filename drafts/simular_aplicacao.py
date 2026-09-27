#!/usr/bin/env python3
"""
Aplica um documento de correcoes a uma copia do LaTeX e verifica o resultado.

    python3 simular_aplicacao.py <pasta do zip extraido> <correcoes.md> [--out <pasta>]

Em cada item verifica que o termo de pesquisa aparece exactamente uma vez no
momento em que e aplicado (ou seja, depois dos itens anteriores), e acrescenta ao
.bib as entradas marcadas com "acrescenta ao `dissertation.bib`:". No fim verifica
chavetas, \\ref sem \\label, \\gls sem acronimo, \\cite sem entrada no .bib e
\\label duplicados. Nao compila (nao ha LaTeX nesta maquina).
Com /dev/null no lugar do .md serve so para verificar um zip.
"""
import glob
import os
import re
import shutil
import sys

src, MD = sys.argv[1], sys.argv[2]
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "/tmp/tese_simulada"
if os.path.exists(out):
    shutil.rmtree(out)
shutil.copytree(src, out)
files = sorted(glob.glob(out + "/**/*.tex", recursive=True)) + [out + "/dissertation.bib"]
T = {f: open(f, encoding="utf-8").read() for f in files}
bib = out + "/dissertation.bib"
problems = []


def find_unique(term):
    hits = [(f, t.count(term)) for f, t in T.items() if term in t]
    n = sum(c for _, c in hits)
    return (hits[0][0] if n == 1 else None), n


md = open(MD, encoding="utf-8").read()
for it in re.split(r"\n(?=## )", md):
    title = it.split("\n", 1)[0][:70]
    # blocos: Procura [Seleciona até] Substitui por ```...``` (ou "apaga")
    for m in re.finditer(
            r"\*\*Procura:\*\* `([^`]+)`\s*(?:\n\*\*Seleciona até:\*\* `([^`]+)`)?\s*"
            r"\*\*Substitui por:\*\*\s*(?:```\w*\n(.*?)\n```|\*\(apaga[^)]*\)\*)", it, re.S):
        start, end, new = m.group(1), m.group(2), m.group(3) or ""
        f, n = find_unique(start)
        if not f:
            problems.append(f"{title} | termo {n}x: {start[:60]}")
            continue
        t = T[f]
        a = t.index(start)
        if end:
            b = t.find(end, a)
            if b < 0:
                problems.append(f"{title} | 'seleciona ate' nao encontrado")
                continue
            b += len(end)
        else:
            b = a + len(start)
        T[f] = t[:a] + new + t[b:]
    # entradas novas para o .bib
    for m in re.finditer(r"acrescenta ao `dissertation\.bib`:\s*```bibtex\n(.*?)\n```", it, re.S):
        T[bib] += "\n\n" + m.group(1) + "\n"

for f, t in T.items():
    open(f, "w", encoding="utf-8").write(t)

# ------------------------------------------------------------ verificacoes
alltex = "\n".join(t for f, t in T.items() if f.endswith(".tex"))
nocom = re.sub(r"(?<!\\)%.*", "", alltex)
labels = set(re.findall(r"\\label\{([^}]*)\}", nocom))
refs = set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]*)\}", nocom))
gls_def = set(re.findall(r"\\newacronym(?:\[[^\]]*\])?\{([^}]*)\}", alltex))
gls_use = set(re.findall(r"\\[gG]ls(?:pl)?\{([^}]*)\}", nocom))
bibkeys = set(re.findall(r"@\w+\{([^,\s]+),", T[bib]))
cites = set(k.strip() for grp in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", nocom)
            for k in grp.split(","))
dup = [l for l in labels if len(re.findall(r"\\label\{" + re.escape(l) + r"\}", nocom)) > 1]
for f, t in T.items():
    if f.endswith(".tex"):
        c = re.sub(r"(?<!\\)%.*", "", t).replace("\\{", "").replace("\\}", "")
        if c.count("{") != c.count("}"):
            problems.append(f"chavetas desequilibradas em {os.path.relpath(f, out)}: {c.count('{') - c.count('}')}")
problems += [f"\\ref sem \\label: {r}" for r in sorted(refs - labels)]
problems += [f"\\gls sem acronimo: {g}" for g in sorted(gls_use - gls_def)]
problems += [f"\\cite sem entrada no .bib: {k}" for k in sorted(cites - bibkeys)]
problems += [f"\\label duplicado: {l}" for l in dup]
print(f"copia simulada em {out}")
print("\n".join(problems) if problems else "sem problemas")
