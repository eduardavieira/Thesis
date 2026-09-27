---
name: thesis-corrections
description: Contexto e plano de correções da tese de mestrado "HRV30 - Stress Detection from Ultra-Short HRV using Deep Learning" (Eduarda Vieira, UMinho). Use quando o trabalho envolver a tese ou este repositório - correções pedidas pela supervisora, edição do LaTeX/Overleaf, re-treino dos modelos, baselines, LOSO, explicabilidade (XAI), o pipeline de ETL de ECG/HRV, os datasets WESAD e SWELL-KW, ou qualquer pergunta sobre o que já foi validado e o que continua por resolver.
---

# Tese HRV30 — contexto e correções

Deteção binária de stress a partir de HRV ultra-curto (janelas de 30 s) com deep
learning, mais calibração do limiar por pessoa, explicabilidade e um agente de
feedback via LLM. Datasets: SWELL-KW + WESAD. Iteração atual: **V6** (durações
reais das gravações). Classificador do sistema: **Model G** (Dual-Input CNN).

## Antes de fazer seja o que for

1. **Nunca editar o zip do Overleaf.** Extrair para o scratchpad para consultar;
   entregar correções no formato **procurar / selecionar até / substituir por**,
   com cada termo de pesquisa validado (aparece uma só vez) contra o zip mais
   recente do repositório, e com a linha **Ficheiro:** de cada alteração. É o
   formato que a autora usa no Overleaf.
2. **Ler `references/facts.md` antes de afirmar qualquer número.** Ler **de
   baixo para cima**: as secções finais corrigem as iniciais. Os números da V6
   estão no fim.
3. **Nunca usar resultados do conjunto combinado** como resultado principal.
4. Responder em **português europeu**. A tese é escrita em inglês.
5. **O scratchpad (`/tmp`) é apagado quando o PC reinicia.** Scripts que
   sustentem números da tese vão para `Classification Models/`, nunca só para o
   scratchpad.
6. **Antes de apagar ficheiros do repositório, enviar a lista à autora** e
   esperar aprovação.

## Ficheiros

| Ficheiro | Conteúdo |
|---|---|
| `references/facts.md` | Todos os números validados, em ordem cronológica (V6 no fim) |
| `references/codebase.md` | Mapa do código, scripts, armadilhas |
| `references/corrections-v5.md` | Cada pedido da supervisora e onde foi resolvido |
| `references/etl-v5.md` | O ETL corrigido (a V6 só muda as durações) |
| `scripts/validate.py` | Re-corre as validações de base |
| `drafts/correcoes_finais.md` | O documento de correções aplicado (secção H: o que ainda falta) |
| `drafts/estado_aplicacao.py <zip> <md>` | Diz, item a item, o que já está aplicado num zip |
| `drafts/simular_aplicacao.py <zip> <md>` | Aplica um `.md` a uma cópia e verifica refs, cites, gls e chavetas |
| `drafts/check_procura.py <md> <zip>` | Valida que cada termo de pesquisa é único |
| `README.md` (raiz) | Atualizado para a V6 a 2026-09-27 |

## O achado central

**Os dois datasets têm rótulos fisiologicamente opostos.** Na tese assenta em
três linhas de evidência (4.1):

1. **Efeito por dataset**: os 8 features mudam em sentidos opostos (hr_mean
   +25,8 bpm no WESAD, −4,6 bpm no SWELL; 15/15 e 22/22 sujeitos).
2. **Transferência entre datasets**: AUC 0,170 a 0,440 em 10 de 10 combinações.
3. **O desenho do SWELL** (artigo deles): a condição neutra é sempre o primeiro bloco.

| | WESAD (TSST) | SWELL (carga de trabalho) |
|---|---|---|
| Efeito no hr_mean | **+25,8 bpm** | −4,6 bpm (invertido) |
| d de Cohen médio | 1,14 | 0,36 |
| AUC por sujeito | 0,878–0,919 (GKF) · 0,913–0,970 (LOSO) | 0,523–0,612 (GKF; LogReg p=0,015, Holm 0,075) |

## Os números principais (V6)

- **WESAD aleatório:** G 89,78% · AUC 0,955 · PR 0,940; **F 90,44 · 0,965 · 0,952**;
  G≈F (p=0,65). Com Holm, a vantagem do G sobrevive em AUC (vs todos menos RF) e
  PR-AUC (vs os 7); com **Nadeau–Bengio** só o F mantém vantagem robusta.
- **Por sujeito:** nada sobrevive a Holm nas 36 comparações. O G tem a média mais
  alta nos dois protocolos. Os modelos da série são os que mais perdem do aleatório
  para o GroupKFold (F −11,4 pp, G −9,6 pp).
- **Calibração por pessoa (4.6.3):** limiar próprio em ~15 min de cada sujeito →
  G 82,1% → 91,1% (p Holm 0,028), pior sujeito 48,4% → 73,3%. LogReg n.s.
- **Combinado (controlo negativo):** a regra "só sei o dataset" (62,44% / 61,44%)
  bate todos os treinados por sujeito.
- **Corpus:** 5409 janelas (WESAD 907, SWELL 4502), 37 sujeitos.

## Model G e a explicabilidade (Secção 4.4)

- **Usa a ordem dos batimentos:** baralhar os intervalos dentro da janela (features
  recalculadas) custa 13,19 pp. O ramo resumo mexe sobretudo no limiar (−12,64 pp,
  AUC só −0,016).
- **A máscara transporta a frequência cardíaca** (r=0,998 com hr_mean, AUC 0,889)
  e nenhum método de atribuição a cobre.
- **As atribuições por feature não se leem uma a uma:** hr_mean, nni_mean, nn50 e
  pnn50 têm efeito condicional contrário à fisiologia. O modelo inteiro vai no
  sentido certo (intervalos ×0,9 → P(stress) sobe em 98% das janelas).
- Por isso o agente **não** deve receber "features mais importantes" do G.

## Armadilhas

**CSVs.** A boa é a **V6**: `/home/eduarda/Desktop/ProcessedECGData_v6.csv` +
`NNSeries_v6.npz` (5409 janelas; `_v6_wesad` = só as 907 do WESAD, mesmas linhas).
A V5 (`..._v5.*`, 5430) tinha janelas de 13–55 s em 5 ficheiros por causa das
durações erradas; usar `Data Processing/time_duration_v6.csv`. Vários scripts
têm os caminhos da V5 por omissão: passar `--csv/--series/--weights` da V6 (ver
`run_v6.sh` e o README). Selecionar features por nome.

**Prefixos de sujeito colidem.** `S1..S25` existem nos dois datasets. Agrupar
por `dataset + sujeito`. No LOSO do WESAD o fold 0 é o **S10**.

**`Models.py:176` faz `plt.show()`**: definir `matplotlib.use("Agg")` antes de
importar `Models`.

**TensorFlow não é determinista em CPU.** Mesmo código e mesmas seeds: Model B
+2,86 pp, D −1,59 pp; F e G < 0,2 pp. Um só fold de LOSO pode oscilar muito
(S5 do G: AUC 0,45 numa corrida, 0,98 noutra).

**Não editar o `run_v6.sh` enquanto uma corrida está ativa** (o bash lê o script aos
poucos e rebenta). **`pgrep -f` num ciclo de espera apanha-se a si próprio**:
esperar pela CSV de saída ou pelo PID, não pelo nome do script.

**O *Replace all* da ortografia (-ize → -ise) estraga identificadores**: aconteceu
com a chave `articleOptimizers`. Verificar `\cite` sem entrada no `.bib` depois.

**As primeiras 16 janelas de cada ficheiro do SWELL** são 8 min de filme; não foram
excluídas (apêndice A.1.1: −5,76 vs −5,26 bpm, 18/20).

**Neste sistema o `grep` é o ugrep**, que recusa expressões longas. Usar Python.

**`stress_feedback_agent.py` só carrega modelos de entrada (8,1)** (os A–D); o
modelo por omissão (`used_model.h5`) foi apagado na limpeza.

## Estado (2026-09-27, zip das 18:37)

Aplicado: todas as correções da supervisora, das duas avaliações externas e a
V6 inteira, exceto o que a autora decidiu não aplicar (abaixo). **Falta a secção
H do `correcoes_finais.md`:** gralha "a a system", a chave de citação
`articleoptimisers` → `articleOptimizers`, e a figura `gradient_by_class_mean_G.png`
com "N=189" (substituir pela de `results/model_G_v6/`).

A autora vai **reescrever o Cap. 4 na sua voz**. Depois, verificar números e
afirmações contra o fim do `facts.md` (qualificadores de significância, G≈F,
vantagem da série só no aleatório, limites da calibração).

## Decisões da autora (não repropor)

- Resultado principal é o WESAD; o combinado é controlo negativo (2026-09-23).
- Classificador do sistema: **Model G** (com a orientadora, 2026-09-24). Na V6 o F
  iguala/ultrapassa o G; o texto diz que o F seria igualmente defensável.
- Os questionários/autorrelato do SWELL não entram (2026-09-26).
- **A calibração por pessoa entra na 4.6.3** (2026-09-27; reverte a decisão de
  2026-09-23 de a deixar de fora).
- ROC/AUC definida na 3.6.1, não no Capítulo 2. Não há acrónimos para API, HTTP, CPU.
- A ablação F1/F2 fica com o Model D (feita antes da escolha do G) e sem PR-AUC.
- **V6 adotada** (2026-09-26).
- Os nomes dos apêndices ficam; o índice e o glossário ficam (o índice tem entradas
  das macros do `dissertation.sty`); a Fig. 3.2 fica como está (2026-09-27).
- **SWELL vs WESAD "com a mesma bitola"** (baseline do WESAD também primeiro; fala e
  postura no TSST; "validated stress response"; "the cause is traced"): **não entra
  no texto**; é tema para a defesa (2026-09-27).
- **Não aplicados por decisão (2026-09-27):** B7 (LF/HF com a crítica de Billman),
  D2 (cortar a subsecção "Deep Learning Frameworks" — a orientadora não pediu), e a
  Fig. 3.1: a autora tirou o texto da `proposed_approach_new.png` para não voltar a
  ficar desatualizado.
- Ficam no repositório os resultados da V5 e o `V5_Tese_..._old.zip`.
