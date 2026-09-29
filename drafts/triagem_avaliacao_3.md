# Triagem da 3.ª avaliação externa (2026-09-29)

> **Atualização (29/09):** a autora decidiu parar de escrever código e pôr no texto a
> comparação SWELL/WESAD e a discussão da escolha do G. As correções estão em
> `drafts/correcoes_avaliacao3.md`. As experiências da secção 5 **não se correm**: ficam
> como limitações ou trabalho futuro. A secção 4 (⊘) deixou de ser só para a defesa.

Avaliador independente que só leu o zip de 29/09 (15:00). Cada ponto foi verificado no
zip e, quando dependia de números, nos dados V6. Os números novos estão no fim do
`facts.md` (secção "Terceira avaliação externa"). As linhas citadas pelo avaliador batem
com o zip.

Legenda: **✔** confirmado · **◐** em parte · **✘** errado · **≡** já dito no texto ·
**⊘** decisão tua, não se reabre (só notas para a defesa).

---

## 1. O que muda conclusões (prioridade alta)

### 1.1 A FC sozinha explica quase tudo em sujeitos novos (2.2) — ✔, e é mais forte do que o avaliador disse
`Classification Models/hr_only_baseline.py`: regressão logística **só com hr_mean**:

| | aleatório | GroupKFold | LOSO |
|---|---|---|---|
| hr_mean | 0,882 | **0,889** | **0,925** |
| nº de batimentos | 0,889 | 0,899 | 0,942 |
| 8 features (LogReg da tese) | 0,922 | 0,913 | 0,939 |
| gama dos 9 modelos treinados | — | 0,878–0,919 | 0,913–0,970 |

Emparelhado por fold: G − hr_mean = +0,030 (GKF, p = 0,36) e +0,045 (LOSO, p = 0,079).
**Em sujeitos novos nenhum modelo tem AUC significativamente acima da FC sozinha.**

- **Texto:** acrescentar a linha "hr_mean only (logistic regression)" à Tabela
  `tab:wesad_subject` e uma frase na 4.2. Na Contribuição 2 e no Resumo dizer que em
  sujeitos novos a informação discriminante é sobretudo a FC.
- **Título:** a FC média é um índice do domínio temporal da HRV (Task Force), por isso
  o título defende-se. Mas o texto tem de dizer o que a tabela mostra.
- **Contradição Shaffer:** a Intro l.74/l.92 e o SotA l.422 dizem que a 30 s o domínio
  temporal é fiável; a Discussão l.452 cita o Shaffer (RMSSD, SDNN e pNN50 precisam de
  60 s). Moderar a Intro e o SotA: só a FC é válida a 10 s; os outros índices temporais
  estão abaixo do mínimo validado.

### 1.2 A calibração sem janelas de stress funciona (2.7) — ✔
`Classification Models/rest_normalisation.py`: mesmo LOSO e mesmas janelas de teste da
4.6.3. Cada sujeito é centrado na média do seu repouso (1.ª metade do baseline) e **não
se usa nenhuma janela de stress**:

| | exatidão | pior sujeito | < 70% |
|---|---|---|---|
| LogReg, 8 features, centrada no repouso | **88,2 ± 14,5** | 62,1 | 3 |
| LogReg, só hr_mean, centrada no repouso | 87,3 ± 14,2 | 62,1 | 3 |
| G, limiar próprio (tese, precisa de stress) | 91,1 ± 7,4 | 73,3 | 0 |
| LogReg, limiar 0,5 (tese) | 81,4 ± 18,3 | 38,7 | 3 |

Centrada − LogReg a 0,5: +6,8 pp (t p = 0,056; Wilcoxon 0,047). Centrada − G com limiar
próprio: −2,9 pp (p = 0,42). Isto resolve o limite "a calibração exige um episódio de
stress", que a tese declara (l.445) e o avaliador ataca.

- **Experiência que falta:** o G com a mesma normalização pelo repouso (~30–40 min).
- **Texto:** entra na 4.6.3 como quarta condição. Citar Prajod 2024 e Byun 2025, que já
  estão no SotA l.496.
- **Limiar "1,00"** — ✘ **o avaliador leu mal.** É **0,997** (S10 e S11), não "nunca prever
  stress". Com esse limiar, o S10 tem especificidade 100% e sensibilidade 66,7%. Mas o
  "1.00" do texto (l.443) engana: escrever "0.04 to 0.997" e dizer que, nesses dois
  sujeitos, a saída está saturada perto de 1, o que torna o limiar frágil entre dias.

### 1.3 O baseline dos integrated gradients (2.6) — ✔
`Classification Models/model_g_baseline_check.py`, com os pesos V6 da 4.4:

- Nas janelas reais, os pesos de classe só levam a P(stress) média de 0,36 para 0,40.
  **Não explicam o 0,916.**
- Série plana (a baseline da tese): 0,916. A mesma série com a variabilidade real,
  centrada na média do treino: 0,765. A série plana conta ~0,15.
- O resto vem de a janela "média" (~82 bpm) ficar do lado do stress da fronteira do G.

**Texto:**
- l.330: trocar "consistent with the class weighting" por esta explicação.
- l.319: "The error is a consequence of the offset of the baseline" confunde o método com o
  modelo. O 0,628 é do modelo; a baseline só muda a forma como é repartido.
- l.252 critica a série plana como entrada fora da distribuição, e a baseline dos IG é
  exatamente isso. Dizer os dois de forma coerente.
- A justificação para não usar SHAP está comentada (l.317). Pô-la de volta.

O sub-argumento "sensibilidade < especificidade contradiz os pesos" é ✘ fraco: os pesos
só deslocam a probabilidade média ~4 pp e não garantem sens > espec. Em sujeitos novos
as redes até têm sens ≥ espec. (G LOSO 82,9 / 83,9).

### 1.4 Contribuição 3 e escolha do G (2.3, 2.4) — ✔ no texto, ⊘ na decisão
A escolha do G é tua (com a orientadora), por isso não se reabre. Mas o texto que a
justifica contradiz-se:

- l.402 diz que a escolha "rests on the one difference ... measured with statistical
  support" e cita G − E "after correction". A l.148 diz que nenhuma sobrevive ao teste
  corrigido. Reescrever a l.402 com o qualificador (Holm habitual sim, Nadeau–Bengio não).
- Conclusões l.21, "Most of these differences survive": começar pelo teste corrigido.
- Contribuição 3 (Intro l.147): acrescentar "under a random split" ao título da
  contribuição. Em GroupKFold, o F (0,903) fica abaixo da LogReg (0,913).
- Contribution l.199, "which isolates the variable under test": ✔ falso. F vs C/D mudam
  também a receita de treino; a comparação limpa é G − E. Reescrever.
- Nested CV: seria viável, mas é uma experiência grande. A l.309 e a l.463 já declaram a
  seleção nas partições de teste (≡).
- Para a defesa: o melhor argumento a favor do G agora é a calibração. Com limiar próprio
  o G sobe 9 pp (p Holm 0,028), a LogReg não (S2 AUC 0,32). Mas a centragem pelo repouso
  (1.2) põe a LogReg perto do G.

---

## 2. Erros e incongruências de texto (correções rápidas)

| # | Onde | Problema | Correção |
|---|---|---|---|
| T1 | SotA l.21 | "adrenaline and noradrenaline, which releases cortisol" ✔ erro fisiológico | O simpático e a medula suprarrenal libertam adrenalina e noradrenalina; o cortisol vem do eixo hipotálamo-hipófise-suprarrenal (córtex) |
| T2 | SotA l.39 | "there is not a treatment", "proved" ✔ | "no single treatment", "shown to reduce" |
| T3 | SotA l.251 | GD "gets the lowest error value possible" ✔ | Desce o gradiente até um mínimo local, sem garantia de mínimo global |
| T4 | SotA l.283 | "RNNs have the deepest structures" ✔ | Tirar; RNNs são profundas no tempo (desdobradas), não as mais profundas |
| T5 | SotA l.168 | cita `ml12` (redes spiking para avatares) para "mais camadas → mais exatidão"; e "Research proved" ✔ | Tirar `ml12`, moderar o "proved" |
| T6 | SotA l.403 | "ultra-short (≤ 5 min)" ✔ | "< 5 min" (5 min é short-term) |
| T7 | Intro l.94 | "must therefore" vs SotA l.398 "not settled" ✔ | "would have to, if it falls under those rules (Section …)" |
| T8 | Intro l.118 | "common preprocessing and training protocol" ✔ falso (A–D vs E–G, pesos da LogReg/RF) | "the same data, partitions and evaluation protocol" |
| T9 | Intro l.106 | objetivo "durations at which they were validated"; as tabelas do SotA não têm durações ✔ | Reescrever o objetivo para a secção `soa:reliability`, ou pôr a duração nas tabelas |
| T10 | Contribution l.193 | "All four models use … batch normalisation" ✔ o A não tem | "Models B to D also use batch normalisation" |
| T11 | Contribution l.172, l.184 | "temporal dependencies in the HRV time series data" vs l.165/l.195 ✔ | Tirar "time series" / "temporal patterns from the HRV features" |
| T12 | Contribution l.73 | filtro sem tipo nem ordem ✔ | Butterworth de ordem 2, `filtfilt` (fase zero), 0,5–40 Hz, na gravação inteira; o BioSPPy aplica ainda um FIR 3–45 Hz em cada segmento |
| T13 | Contribution l.63/l.75 | picos por segmento ✔ | Declarar: o Hamilton reinicializa em cada segmento (8 s) e o intervalo entre segmentos perde-se (≈1 em ~35). Limitação: a deteção não foi validada contra anotações |
| T14 | Applications l.67 | "All 22 participants" ✔ | Acrescentar "(18 of the 20 with windows of both conditions once the relaxation phase is excluded, Appendix …)" |
| T15 | Resumo | "5,409 windows from 37 subjects" ✔ | Acrescentar "the main evaluation uses the 907 windows of the 15 WESAD subjects" |
| T16 | Applications l.61/l.65 | ICs da transferência sobre seeds ✔ | "across training runs"; não refletem a amostragem de sujeitos |
| T17 | `tab:wesad_subject` | falta a sensibilidade ✔ | Acrescentar coluna (números no `facts.md`) |
| T18 | Intro l.74 e l.88 | article6/9/20 não são sobre smartwatches; article10 é, mas é PPG ◐ | Citar `velmovitsky2022apple` (já no .bib) para o ECG de 30 s |
| T19 | SotA l.61 | features não lineares nomeadas e nunca tratadas ✔ | Uma frase: SD1 é RMSSD/√2 e SD2 depende do SDNN (Brennan 2001, **verificar no original**); as entropias precisam de janelas mais longas |
| T20 | SotA l.573–579 | o resumo repete a Intro ✔ | Resumir a literatura: partições aleatórias, datasets juntos, 60 s nos estudos por sujeito, lacuna |
| T21 | Conclusões, trabalho futuro | janelas mais longas comentadas (l.40); o texto antigo promete 70–80% sem base ✔ | Reescrever sem o número: curva 10/30/60/120 s no mesmo pipeline, e o BVP do WESAD |
| T22 | Contribuição 1 | não diz que estende Prajod 2024 ◐ | Uma oração: "extending the cross-dataset loss reported by Prajod et al." (a l.456 já diferencia) |

**Opcionais:**
- PyTorch (SotA l.366): a secção fica (decisão D2). Só atualizar o facto: desde 2022 está
  na PyTorch Foundation (Linux Foundation).
- Epidemiologia centrada nos EUA (l.27–34): juntar dados da Europa e de Portugal
  (EU-OSHA, Ordem dos Psicólogos). É preciso pesquisar as fontes.
- Figura do neurónio (fonte Healthline, `neuronSite`) e anatomia do neurónio: material de
  manual. A decisão sobre cortes é tua.

---

## 3. Já está no texto (≡): preparar para a defesa, sem mudar nada

- 30 s nunca comparado com 60/120 s: l.465. O BVP do WESAD não foi usado: l.465.
- 49 ms medidos em Keras num portátil: l.406.
- Determinismo do TensorFlow: l.467 (variação medida em vez de corrigida).
- LogReg sem pesos e RF "balanced": l.117. Sem procura de hiperparâmetros: só implícito
  (l.174, "default hyperparameters"). Vale uma frase nas limitações.
- Tarefa binária padrão do WESAD (com diversão) não corrida: l.50, l.454.
- Agente não avaliado: l.391. A pergunta "uma lista fixa fazia o mesmo?" é boa para a
  defesa. A resposta honesta é que o agente só acrescenta variedade (l.385), e isso não
  foi testado.
- Potência: a l.445 diz "little power". Melhor do que uma análise de potência: dar o IC95
  da diferença G − LogReg em LOSO, que mostra que também não se pode afirmar equivalência.
- Contribuição 6 sobrevalorizada: já diz "proof of concept"; basta não prometer mais.
- Sensibilidade < especificidade (l.242 vs l.447): ✘ não é incongruência, a l.447
  reconhece-o. Acrescentar que em sujeitos novos a relação se inverte (T17).

## 4. Decisões tuas (⊘): não se reabrem

- **2.1 SWELL vs WESAD com a mesma bitola** (decidido a 27/09: tema para a defesa).
  Notas novas para a defesa:
  - **Qualidade do sinal:** as exclusões no SWELL são iguais nas duas classes
    (20,0% / 19,7%). Não podem criar uma inversão que depende da classe.
  - **Stressores fracos** dariam d ≈ 0, não uma inversão em 22 de 22. É preciso um fator
    sistemático, e o Koldijk reconhece "order or fatigue effects".
  - **No WESAD a ordem joga contra o efeito:** uma primeira condição nova subiria a FC do
    baseline. A fala e a postura do TSST jogam a favor e não se podem excluir.
  - **p = 0,78 no bloco neutro:** não há decaimento dentro do bloco. Um efeito de ordem
    pode atuar entre blocos (fadiga ao longo da sessão), não dentro de cada um.
- Model G como classificador. Ablação F1/F2 só com o D. A secção "Deep Learning
  Frameworks" fica.

## 5. Experiências possíveis, por custo

| Experiência | Custo | Responde a |
|---|---|---|
| G com normalização pelo repouso (LOSO) | ~30–40 min | 2.7; melhora a 4.6.3 |
| IG com baseline alternativa (média de janelas reais / expected gradients) | ~10 min | 2.6 |
| Deteção de picos: Hamilton vs outro detetor (concordância nos intervalos) | ~1 h | 2.9 |
| Curva 10/30/60/120 s com LogReg e G | ETL por duração (~20 min cada no WESAD) + treino | 2.8 |
| Tarefa padrão do WESAD (baseline + diversão vs TSST) | extrair a diversão + treino | 2.10, Schmidt 2018 |
| Nested CV para a escolha do modelo | horas | 2.4 |
| Re-treino determinista | horas | 2.10 |
