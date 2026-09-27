#!/usr/bin/env python3
"""
Estado de cada correcao num zip do Overleaf: aplicada, pendente ou em conflito.

    python3 estado_aplicacao.py <pasta do zip extraido> <correcoes.md> [...]

Para cada item (Procura / Seleciona ate / Substitui por):
  APLICADO  o texto novo aparece no LaTeX (espacos normalizados)
  PENDENTE  o termo de pesquisa aparece exactamente uma vez
  CONFLITO  nem uma coisa nem outra (o texto foi editado de outra forma)
Compara tambem as seccoes sec_4_*_v6.tex com o Applications.tex.
"""
import glob
import os
import re
import sys

root = sys.argv[1]
files = glob.glob(root + "/**/*.tex", recursive=True) + glob.glob(root + "/*.bib")
T = {f: open(f, encoding="utf-8", errors="replace").read() for f in files}
norm = lambda s: re.sub(r"\s+", " ", s).strip()
ALL = norm("\n".join(T.values()))
here = os.path.dirname(os.path.abspath(__file__))


def status(start, end, new):
    new_n = norm(new) if new else ""
    # texto novo presente (primeiros ~160 caracteres com conteudo, para tolerar
    # pequenas diferencas no fim do bloco)
    probe = new_n[:160]
    if new_n and probe and probe in ALL:
        return "APLICADO"
    n = sum(t.count(start) for t in T.values())
    if n == 1 and (not end or any(start in t and end in t[t.index(start):] for t in T.values())):
        return "PENDENTE"
    return f"CONFLITO ({n}x)"


for md in sys.argv[2:]:
    txt = open(md, encoding="utf-8").read()
    print(f"\n#### {os.path.basename(md)}")
    for it in re.split(r"\n(?=## )", txt):
        title = it.split("\n", 1)[0].strip()
        res = []
        for m in re.finditer(
                r"\*\*Procura:\*\* `([^`]+)`\s*(?:\n\*\*Seleciona até:\*\* `([^`]+)`)?\s*"
                r"\*\*Substitui por:\*\*\s*(?:```\w*\n(.*?)\n```|\*\(apaga[^)]*\)\*)", it, re.S):
            res.append(status(m.group(1), m.group(2), m.group(3)))
        if res:
            print(f"  {title[:60]:60s} {', '.join(res)}")

# seccoes do Cap. 4
app = norm(T.get(root + "/chapters/Applications.tex", ""))
print("\n#### seccoes do Cap. 4")
for d in sorted(glob.glob(os.path.join(here, "sec_4_*_v6.tex"))):
    body = re.sub(r"\A(?:%.*\n)+", "", open(d, encoding="utf-8").read())
    paras = [norm(p) for p in re.split(r"\n\s*\n", body) if len(norm(p)) > 200 and not p.lstrip().startswith("\\")]
    hit = sum(p[:150] in app for p in paras)
    print(f"  {os.path.basename(d):20s} paragrafos iguais aos do zip: {hit}/{len(paras)}")
