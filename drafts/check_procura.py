#!/usr/bin/env python3
"""Valida os termos de procura de um .md de correcoes contra o LaTeX extraido.
Cada termo entre crases a seguir a **Procura:** ou **Seleciona até:** tem de
aparecer exactamente uma vez no conjunto dos .tex e .bib."""
import re, glob, sys
root = sys.argv[2] if len(sys.argv) > 2 else '.'
files = glob.glob(root + '/**/*.tex', recursive=True) + glob.glob(root + '/*.bib')
txt = {f: open(f, encoding='utf-8', errors='replace').read() for f in files}
md = open(sys.argv[1], encoding='utf-8').read()
bad = 0
for kind, term in re.findall(r'\*\*(Procura|Seleciona até):\*\* `([^`]+)`', md):
    hits = [(f, t.count(term)) for f, t in txt.items() if term in t]
    n = sum(c for _, c in hits)
    if n != 1:
        bad += 1
        print(f"[{n}x] {kind}: {term[:90]}  {[h[0].split('/')[-1] for h in hits]}")
print("termos com problema:", bad)
