---
name: thesis-corrections
description: Contexto e plano de correções da tese de mestrado "HRV30 - Stress Detection from Ultra-Short HRV using Deep Learning" (Eduarda Vieira, UMinho). Use quando o trabalho envolver a tese ou este repositório - correções pedidas pela supervisora, edição do LaTeX/Overleaf, re-treino dos modelos, baselines, LOSO, explicabilidade (XAI), o pipeline de ETL de ECG/HRV, os datasets WESAD e SWELL-KW, ou qualquer pergunta sobre o que já foi validado e o que continua por resolver.
---

# Tese HRV30 — contexto e correções

Deteção binária de stress a partir de HRV ultra-curto (janelas de 30 s) com deep
learning, mais explicabilidade e um agente de feedback via LLM. Datasets:
SWELL-KW + WESAD. Iteração atual: **V5** (correções à V4). Classificador do
sistema: **Model G** (Dual-Input CNN).

## Antes de fazer seja o que for

1. **Nunca editar o zip do Overleaf.** Extrair para o scratchpad para consultar;
   entregar correções no formato **procurar / selecionar até / substituir por**,
   com cada termo de pesquisa validado (aparece uma só vez) contra o zip mais
   recente do repositório. É o formato que a autora usa no Overleaf.
2. **Ler `references/facts.md` antes de afirmar qualquer número.** Ler **de
   baixo para cima**: as secções finais corrigem as iniciais.
3. **Nunca usar resultados do conjunto combinado** como resultado principal.
4. Responder em **português europeu**. A tese é escrita em inglês.
5. **O scratchpad (`/tmp`) é apagado quando o PC reinicia.** Scripts que
   sustentem números da tese vão para `Classification Models/`, nunca só para o
   scratchpad.

## Ficheiros

| Ficheiro | Conteúdo |
|---|---|
| `references/facts.md` | Todos os números validados, em ordem cronológica de descoberta |
| `references/codebase.md` | Mapa do código, scripts, armadilhas |
| `references/corrections-v5.md` | Cada pedido da supervisora e onde foi resolvido (topo: estado final) |
| `references/etl-v5.md` | O ETL corrigido |
| `scripts/validate.py` | Re-corre as validações de base |
| `drafts/` (raiz do repo) | Rascunhos e documentos de procurar/substituir entregues à autora |

## O achado central

**Os dois datasets têm rótulos fisiologicamente opostos.** Na tese assenta em
três linhas de evidência:

1. **Efeito por dataset**: os 8 features mudam em sentidos opostos (hr_mean
   +25,7 bpm no WESAD, −4,6 bpm no SWELL).
2. **Transferência entre datasets**: AUC 0,178 a 0,441 em 10 de 10 combinações.
3. **O desenho do SWELL** (paper deles): a condição neutra é sempre o primeiro
   bloco. 22/22 participantes têm frequência mais baixa no "stress".

O autorrelato dos participantes do SWELL também o confirma (arousal −1,28,
p<0,05), mas **não entra na tese** — decisão da autora. Não repropor.

| | WESAD (TSST) | SWELL (carga de trabalho) |
|---|---|---|
| Efeito no hr_mean | **+25,7 bpm** | −4,6 bpm (invertido) |
| d de Cohen médio | 1,13 | 0,37 |
| Melhor AUC por sujeito | **0,951** (LogReg, LOSO) | 0,612 (LogReg, GroupKFold, p=0,058, n.s.) |

## Os números principais (corrida de 2026-09-26, a que está na tese)

1. **WESAD, por sujeito: todos os modelos treinados com AUC 0,874–0,951**
   (GroupKFold e LOSO). Nenhuma das 36 comparações par a par sobrevive a Holm,
   em nenhum protocolo nem métrica. Não há vencedor por sujeito.
2. **WESAD, split aleatório: Model G 89,95% · AUC 0,959 · PR-AUC 0,933.** Bate os
   cinco modelos neuronais das features resumo com significância após Holm (D por
   6,46 pp). **Contra a LogReg (+4,97 pp) e a RF (+3,18 pp) não sobrevive** em
   exatidão. Em AUC sobrevive contra todos exceto a RF (três no limite, 0,0499).
   Em PR-AUC nada sobrevive.
3. **Transferência WESAD↔SWELL: AUC 0,18–0,44.**
4. **Regra "só sei o dataset"**: 62,58% (aleatório) supera as quatro arquiteturas
   A–D; 61,35% (por sujeito) supera todos os modelos treinados.

## Model G e a explicabilidade (Secção 4.4)

- A série NN conta: achatar a série para a média da janela custa 12,17 pp e
  0,078 de AUC. O ramo resumo só desloca o limiar (AUC inalterada sem ele).
- **A máscara transporta a frequência cardíaca** (nº de batimentos, r=0,98 com
  hr_mean, AUC 0,882 sozinha) e nenhum método de atribuição a cobre.
- **As atribuições por feature resumo não se leem uma a uma**: hr_mean, rmssd e
  nn50 têm efeito condicional contrário à fisiologia (duas vias,
  `model_g_directions.py`). O modelo inteiro vai no sentido certo (intervalos
  ×0,9 → P(stress) sobe em 87% das janelas).
- Por isso o agente **não** deve receber "features mais importantes" do G.

## Armadilhas

**Duas CSVs.** `/home/eduarda/Desktop/ProcessedECGData_v5.csv` (5430 janelas,
ETL corrigido) é a boa; as séries NN estão em `/home/eduarda/Desktop/NNSeries_v5.npz`.
A `hrv_data_final/ProcessedECGData.csv` tem `sdnn_index` toda NaN que parte a
seleção posicional do `Data.py`. Selecionar features por nome.

**Prefixos de sujeito colidem.** `S1..S25` existem nos dois datasets. Agrupar
por `dataset + sujeito`. No LOSO do WESAD o fold 0 é o **S10** (ordem
lexicográfica).

**`Models.py:176` faz `plt.show()`**: definir `matplotlib.use("Agg")` antes de
importar `Models`.

**TensorFlow não é determinista em CPU.** Mesmo código e mesmas seeds, duas
invocações: Model B +2,86 pp, D −1,59 pp; F e G < 0,2 pp. Diferenças de poucos
pp entre os Models A–D não são interpretáveis. O C e o D trocaram de ordem.

**As primeiras 16 janelas de cada ficheiro do SWELL** são 8 min de filme
(relaxação). **Não foram excluídas na CSV V5** (ex.: `S1_neutral` tem as 117
janelas possíveis) e têm o rótulo do bloco. Não afeta o WESAD; excluí-las não
muda a inversão (mediana −5,44 vs −5,26 bpm, 18/20 sujeitos). Documentado no
apêndice A.1.1.

**Neste sistema o `grep` é o ugrep**, que recusa expressões longas ("exceeds
complexity limits"). Para procurar muitos padrões no LaTeX, usar Python.

**`stress_feedback_agent.py` só carrega modelos de entrada (8,1)** (os A–D). A
3.8 descreve o G carregado dos pesos, o que é verdade via
`explain_model_g.py --load`, mas a função de integração do agente não suporta o G.

## Estado (2026-09-26)

**Todos os pedidos da supervisora estão cumpridos** — validado contra o zip de
26/09 14:35: os 12 pontos das notas, os erros de conteúdo, atenção/gradientes,
figuras, escrita e as 15 anotações do PDF. Tabela de correspondência no topo de
`references/corrections-v5.md`.

Falta:
1. Gralha **"valuation metrics included"** (`Contribution.tex:307`) → "Evaluation".
2. **Reescrever o Capítulo 4 na voz da autora antes de submeter** (4.4, 4.6 e
   muitas substituições foram redigidas pelo Claude).
3. Opcionais sugeridos e não feitos: gramática do 1.º parágrafo da 4.3,
   "degrades entirely" na 4.1.1, citação de Lazarus ("taxes or exceeds" vs
   "tax or exceed"), apagar figuras antigas `attention_*` de `images/`.

## Decisões da autora (não repropor)

- Resultado principal é o WESAD; o combinado é controlo negativo (acordado com a
  orientadora a 2026-09-23).
- Classificador do sistema: **Model G** (com a orientadora, 2026-09-24).
- Os questionários/autorrelato do SWELL não entram (2026-09-26).
- O protocolo personalizado (limiar por sujeito) fica fora da tese (2026-09-23);
  medido e válido, ver `facts.md`, "Protocolo personalizado".
- ROC/AUC definida na 3.6.1, não no Capítulo 2. Não há acrónimos para API, HTTP, CPU.
- A ablação F1/F2 fica com o Model D (feita antes da escolha do G) e sem PR-AUC.

## O achado para o trabalho futuro

Em LOSO a AUC é 0,90–0,95 mas a exatidão por sujeito vai de 37,5% a 100% e a
especificidade de 0% a 100%. No S10, a LogReg (AUC 0,901) e o Model F (0,923) têm
especificidade 0: **ordenação boa, decisão inútil**. O limiar fixo de 0,5 não
serve pessoas com basais diferentes — é o argumento da calibração por sujeito
no trabalho futuro.
