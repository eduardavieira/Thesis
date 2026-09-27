# Mapa do código

Repo: `/home/eduarda/Desktop/repositorios/Thesis`

## Pipeline completo

```
ECG bruto (S00-files SWELL / pickle WESAD)
  → Data Processing/          extração, band-pass, deteção de picos R, features HRV
  → ProcessedECGData.csv      6596 janelas × 8 features + label
  → Classification Models/    split, normalização, treino, avaliação
  → Explainability.py         saliência de gradientes + integrated gradients
  → stress_feedback_agent.py  classificação → Claude API → mensagem ao utilizador
```

## Data Processing/

| Ficheiro | Função |
|---|---|
| `RPeakDetection_HRV_RemoteMachine.py` | ETL principal. É aqui que está o bug do filtro. |
| `RPeakDetection_HRV.ipynb` | Versão notebook do mesmo |
| `filtering_wesad.ipynb` | Band-pass 0,5–40 Hz, só aplicado ao WESAD |
| `Obtaining data/Obtaining WESAD data.ipynb` | Desserialização dos pickles |

Fluxo em `RPeakDetection_HRV_RemoteMachine.py`:

- `rPeak()` (L67) — biosppy `ecg()`, devolve tempos dos picos R **em segundos**
- `rPeaks_final()` (L85) — aplica a cada segmento de 30 s
- `detectrr()` (L99) — diferenças consecutivas → intervalos RR **em segundos**
- L174–176 — filtro de artefactos. **PARTIDO**, ver abaixo
- L179 — `pyhrv.hrv(nni=...)` converte s→ms automaticamente (média < 10 ⇒ assume s).
  As unidades estão corretas; o problema não é de unidades.
- L197 — escreve 12 valores por linha

### O bug do filtro (`RPeakDetection_HRV_RemoteMachine.py:174`)

```python
for g in rr_interval:
    if (g < 0.5 or g > 1.2):
        rr_interval.remove(g)      # muta a lista durante a iteração
```

Remover durante a iteração faz o índice interno saltar posições, por isso o
elemento imediatamente a seguir a cada remoção nunca é testado.

```
input:  [0.4, 0.45, 0.8, 1.5, 1.6, 0.9]
output: [0.45, 0.8, 1.6, 0.9]        # 0.45 e 1.6 sobreviveram
```

Correção mínima: `for g in list(rr_interval):`. Correção certa (pedida pela
supervisora): **interpolar** em vez de remover — apagar um intervalo cria uma
diferença falsa entre batimentos vizinhos, e RMSSD/NN50/pNN50 são calculados
precisamente a partir dessas diferenças.

Nota de âmbito: 500–1200 ms corresponde a 50–120 bpm. O TSST do WESAD leva a
frequência acima de 120 bpm, por isso mesmo um filtro *funcional* apagaria dados
reais nas janelas mais stressadas. Vale a pena contar remoções por classe.

## Classification Models/

| Ficheiro | Conteúdo |
|---|---|
| `main.py` | Ponto de entrada. Caminho da CSV em L13, escolha do modelo em L49 |
| `Data.py` | `load_dataset()`, `normalize_data()` |
| `Models.py` | Todas as arquiteturas + `train_evaluate_model()` |
| `Explainability.py` | `AttentionExtractor` (nome enganador), `IntegratedGradientsExplainer`, `create_case_studies()` |
| `run_explainability.py` | Gera todas as figuras de XAI |
| `stress_feedback_agent.py` | Agente LLM, com fallback local |
| `measure_inference.py` | Latência de inferência (63 ms/amostra) |
| `saved_models/used_model.h5` | Model D treinado |

### Mapeamento Modelo A–D (`Models.py:178`)

A tese usa letras, o código usa nomes. Não coincidem em ordem:

| Tese | Função em `Models.py` | Arquitetura |
|---|---|---|
| Model A | `model2_simple` | LSTM simples |
| Model B | `model2_new` | Deep LSTM (3 camadas) |
| Model C | `model4_cnn_lstm_hybrid` | CNN-LSTM |
| **Model D** | `model5_cnn_transformer_hybrid_v2` | CNN-Transformer — **o selecionado** |

### Pré-processamento (`Data.py`)

`load_dataset()` — seleção **posicional** `iloc[:, 2:10]`. Ver a armadilha das duas
CSVs no `SKILL.md`. `dropna` sobre labels e features.

`normalize_data()` — três fases, todos os parâmetros só do treino (sem fuga):
1. clip por IQR a `Q1 − 3·IQR` / `Q3 + 3·IQR`
2. `StandardScaler`
3. clip de segurança a ±5 SD

### Reshape crítico (`main.py:41`)

```python
x_train_reshape = x_train_norm.reshape(nrows_train, ncols, 1)   # (N, 8, 1)
```

As 8 features passam a ser tratadas como **8 timesteps**. Não existe ordem temporal
entre `hr_mean`, `nni_mean`, `sdnn`, etc. — a ordem é a das colunas do CSV.

Consequência técnica (importante para a discussão com a supervisora): isto **não**
reduz os modelos a MLPs. Um `Conv1D(kernel=3)` sobre 8 features impõe partilha de
pesos entre features adjacentes e um prior de localidade — `hr_mean`/`nni_mean`/
`nni_min` partilham um filtro apenas por estarem lado a lado no CSV. É um viés
indutivo arbitrário, o que é *pior* do que não ter nenhum. É por isso que a Random
Forest, que não assume ordem, extrai ~4 pontos a mais.

### Treino (`Models.py:58`)

- `validation_split=0.2` sobre o treino (L110) — é daqui que vem o conjunto de
  validação que a Secção 3.6 menciona mas não descreve
- `compute_class_weight('balanced')` (L76) e depois `class_weight_dict[1] *= 1.2` (L86)
  — o multiplicador é deliberado, para empurrar a fronteira para a classe positiva
- Adam `lr=0.0001`; `clipnorm=0.5` (LSTM/CNN-LSTM), `clipnorm=1.0` (Transformer)
- 100 épocas (recorrentes) / 150 (Transformer); early stopping + ReduceLROnPlateau

### Explicabilidade — o nome está errado

`Explainability.py:18` define `AttentionExtractor`, mas L40–41 diz-se explicitamente:

> *"TensorFlow's MultiHeadAttention doesn't expose weights directly, so we'll use
> gradient-based attention as a proxy"*

**Nunca foi extraído um peso de atenção.** Tudo o que existe são gradientes
(`|∂y/∂x|`, L57), que funcionam identicamente nos Modelos A, B e C. Os eixos das
figuras dizem "Attention Score" e estão errados.

Dois métodos distintos, que medem coisas diferentes:

| Método | Fórmula | Sinal | Depende da baseline |
|---|---|---|---|
| Saliência (L47–60) | `\|∂y/∂x\|` | não | não |
| Integrated Gradients (L311) | `(x − baseline) × grad_médio` | sim | sim |

A baseline é zero **em dados estandardizados**, ou seja, a média do treino. Uma
feature exatamente na média recebe IG = 0 independentemente do gradiente. **A
discordância entre os dois métodos é matematicamente esperada**, não é um bug —
o erro está no texto da tese afirmar que convergem.

O "atributo exatamente zero" nas figuras também não é bug: é a normalização
min-max em `Explainability.py:60`, que mapeia sempre o mínimo para 0 exato.
Trocar por `scores / scores.max()` resolve a leitura enganadora.

### Seleção das amostras das figuras (`run_explainability.py`)

| Figura | Origem | Critério |
|---|---|---|
| Fig. 14 | `feature_importance_global.png` | média de 200 amostras |
| Fig. 15 | `attention_stress_vs_nostress.png` | `tp_indices[0]` — **primeiro TP de qualquer confiança** |
| Fig. 16 | `true_positive_high_conf_attention.png` | `create_case_studies`, primeiro TP com `proba > 0.8` |
| Fig. 17 | `true_negative_high_conf_attention.png` | primeiro TN com `proba < 0.2` |
| Fig. 18 | `true_positive_high_conf_ig.png` | **mesma amostra da Fig. 16** |
| Fig. 19 | `true_negative_high_conf_ig.png` | mesma amostra da Fig. 17 |

Daqui saem duas críticas da supervisora, ambas corretas: a Fig. 15 usa uma amostra
**diferente** das Figs. 16/17 mas o texto lê-se como se fossem a mesma; e as
Figs. 16 e 18 são a mesma amostra a dar rankings contraditórios.

## Fonte LaTeX

No zip (`V4_Tese_Eduarda_Vieira_Detecao_Stress_HRV_ECG.zip`), extrair para `/tmp`:

| Ficheiro | Capítulo |
|---|---|
| `chapters/Introduction.tex` | 1 — Contribuições em L135 |
| `chapters/StateOfTheArt.tex` | 2 — Tabelas 1 e 2, typos, parágrafos a cortar |
| `chapters/Contribution.tex` | 3 — Desenvolvimento, métricas em L218 |
| `chapters/Applications.tex` | 4 — Resultados, XAI, discussão |
| `chapters/ConclusionsAndFutureWork.tex` | 5 |
| `chapters/PlannedSchedule.tex` | 6 — **resto da proposta, a remover** |
| `preamble/Abstract.tex` | Abstract + Resumo (keywords comentadas L15/L37) |
| `dissertation.tex` | `\sysname` = HRV30 (L6); título (L48); `\part` (L99,105,120) |

### Bibliografia — porque é que as citações aparecem sem parênteses

`dissertation.sty:10` carrega `natbib`; `dissertation.tex:113` usa `plainnat`.
Em natbib, `\cite` comporta-se como **`\citet`** → "Hickey et al. (2021)" sem
parênteses, colado ao texto.

Estado atual: **296 `\cite{}`, zero `\citet`, zero `\citep`**. O diagnóstico da
supervisora está certo quanto ao efeito, mas o comando a substituir é `\cite`.
Substituição global `\cite{` → `\citep{`, depois rever manualmente os ~40 casos
"Author et al. \cite{}" que devem passar a `\citet{}` sem repetir o nome.

---

# Scripts criados na V5

Todos em `Classification Models/`. Nenhum toca no `main.py` nem nos ficheiros
originais. Todos definem `matplotlib.use("Agg")` antes de importar `Models`.

## `retrain_v5.py` — Modelos A–D

```bash
python3 retrain_v5.py <csv> [--models A,B,C,D] [--seeds 5]
                      [--features F1|F2] [--cv random|group]
                      [--dataset all|SWELL|WESAD] [--out results.csv]
```

Mesmo pipeline do `main.py` (split estratificado, `normalize_data()`, reshape
para `(N, 8, 1)`, `train_evaluate_model()`), com hiperparâmetros por arquitetura
da Secção 3.6: clipnorm 0,5 / 100 épocas (A, B, C); 1,0 / 150 (D).
Grava o CSV a cada corrida, por isso uma falha a meio não perde tudo.

Seleciona features **por nome** — o `iloc[:, 2:10]` do `Data.py` parte-se em
qualquer CSV com `sdnn_index`.

## `train_sequence.py` — arquiteturas novas

```bash
python3 train_sequence.py --csv <csv> --series <npz>
                          [--arch summary|seq|both] [--seeds 5]
                          [--cv random|group] [--dataset all|SWELL|WESAD]
```

| `--arch` | Entrada | O que isola |
|---|---|---|
| `summary` | 8 features, vetor plano `(8,)` | tirar o reshape falso |
| `seq` | série NN `(73, 1)` + máscara | dar a sequência real |
| `both` | as duas, em ramos paralelos | a proposta da supervisora |

O ramo sequencial é Conv1D(64,5) ×2 + **pooling médio mascarado**: as séries
têm comprimentos diferentes (14 a 73) e vão com padding a zero; como o Conv1D
não respeita máscaras do Keras, as posições de padding são zeradas após a
convolução e a média é feita sobre o número de posições reais.

Um BiLSTM foi testado e abandonado — >10 min/seed em CPU contra ~1,5 min.
O Conv1D é também a comparação mais limpa: mantém a família dos Modelos C/D e
muda só a representação de entrada.

Em `--cv group` os índices de treino são permutados antes do `fit`, porque o
`validation_split` do Keras corta os últimos 20% **sem baralhar** e daria um
conjunto de validação de um só sujeito. Só afeta o early stopping.

## `transfer_test.py` — transferência entre datasets

```bash
python3 transfer_test.py --csv <csv> --series <npz> [--seeds 3]
```

Treina num dataset e testa no outro, nas duas direções, com LogReg,
RandomForest e as três arquiteturas novas. Normalização com estatísticas só do
dataset de treino. Leitura: AUC = 0,5 é ausência de sinal transferível;
**AUC < 0,5 é sinal transferível mas invertido**.

## Dados gerados

| Ficheiro | Conteúdo |
|---|---|
| `/home/eduarda/Desktop/ProcessedECGData_v5.csv` | 5430 janelas × 12 colunas |
| `/home/eduarda/Desktop/NNSeries_v5.npz` | `nni` (5430, 73) + `length` (5430,) |

O `.npz` está alinhado **linha a linha** com o CSV — verificado comparando
`mean(série)` com `nni_mean` (erro máximo 6e-5). Gerado com
`RPeakDetection_HRV_fixed.py --series-out`. Se o CSV for regenerado sem o
`--series-out`, o alinhamento perde-se; gerar sempre os dois na mesma execução.
O `train_sequence.py` verifica o número de linhas e aborta se não bater.

## Ficheiros de resultados

`results_v5.csv` (A–D, F2, aleatório) · `results_group_ABCD.csv` (A–D, por
sujeito) · `results_F1_D.csv` / `results_F2_D.csv` (Contribuição 1) ·
`results_seq_{summary,seq,both}.csv` (aleatório) ·
`results_group_{summary,seq,both}.csv` (por sujeito) ·
`results_swell_{random,group}_*.csv` · `results_wesad_{random,group}_*.csv` ·
`results_wesad_ABCD_{random,group}.csv` · `results_transfer.csv`

## Fontes externas consultadas

| Ficheiro | Para quê |
|---|---|
| `~/Downloads/ICMI 2014 paper_final_cr.pdf` | Paper do SWELL. Secção 3: "the neutral condition was always the first condition"; "each block started with a relaxation phase of about 8 minutes" |
| `~/Downloads/Questionnaire Blok Results_final - performance added.ods` | Autorrelato por bloco (folha `Data`): Stress, Arousal, MentalEffort, TemporalDemand, Valence |
| `Tese 2026/Data Processing/SWELL-KW - overview available data.xlsx` | Folha `Sheet3`: ordem das condições por participante. Bloco 1 = N nos 25 |

Nota: `Tese 2026/Data Processing/S2_quest.csv` está mal arrumado — é um
questionário do **WESAD** (a linha ORDER diz `Base;TSST;Medi 1;...`), não do SWELL.

## Erros meus nesta sessão, para não se repetirem

- `for g in rr_interval: rr_interval.remove(g)` — mutação durante iteração (era
  o bug original do ETL, e a causa de 24% de janelas contaminadas)
- `tn, fp, fn, tp = confusion_matrix(...)` sobrescreveu uma variável `fn` que
  guardava o nome da arquitetura → `Unknown model: 94` na 2ª seed
- `np.where(...)` devolve array numpy de strings, que **não** concatena com `+`.
  Usar `pd.Series` para construir ids de sujeito
- Afirmei que a série NN era "a única representação a discriminar acima do
  acaso" no conjunto combinado; a LogReg também passava (AUC 0,567, p=0,032),
  e o resultado da série estava inflacionado por separação de datasets

---

# Alterações à explicabilidade (2026-09-17)

## `explain_wesad.py` — novo

```bash
python3 explain_wesad.py [--dataset WESAD] [--seed 42] [--load]
```

Treina o Model D nos dados V5 filtrados a um dataset, guarda em
`saved_models/wesad_model.h5`, imprime a matriz de confusão (nova Tabela 8) e
depois faz `exec()` do `run_explainability.py`. O script original testa se
`model`, `x_test_reshape` e `y_test` existem no scope e, existindo, salta o
carregamento do `.h5` da V4 — foi desenhado para isto.

**`--load`** carrega o modelo já guardado em vez de treinar: ~2 min em vez de
~15. Usar sempre que se mexe em rótulos ou no código de visualização. O split é
determinista (mesma seed, mesmo CSV, mesmo filtro), por isso o conjunto de teste
é exatamente o mesmo do treino original.

Seleciona features **por nome** e força `matplotlib.use("Agg")` antes de
importar `Models` (senão o `plt.show()` da L176 bloqueia).

Nota: o `run_explainability.py` escreve para `results/` com caminhos fixos.
As figuras da V4 foram copiadas para `results_v4_backup/` antes da primeira
corrida. **Voltar a correr sobrescreve `results/`.**

## `Explainability.py` — duas correções

Backup do original em `Explainability.py.bak`.

### 1. Rótulos: "Attention" → "Gradient Magnitude"

| Linha | Antes | Agora |
|---|---|---|
| ~76 | `Attention Score (Importance)` | `Gradient Magnitude (Normalised)` |
| ~95 | `bubble size = attention` | `bubble size = gradient magnitude` |
| ~101 | `Attention Score` (colorbar) | `Gradient Magnitude` |
| ~145 | `Attention Score` (eixo y) | `Gradient Magnitude (Normalised)` |
| ~146 | `Attention Pattern Comparison` | `Gradient Attribution` |

A docstring da classe `AttentionExtractor` passou a dizer explicitamente que não
extrai pesos de atenção e porquê. **O nome da classe e os métodos mantêm-se**,
para não partir o `run_explainability.py`. Os nomes dos ficheiros também
(`attention_*.png`) — renomeá-los obrigaria a mexer nas referências do LaTeX.

### 2. Normalização: min-max → divisão pelo máximo

Três sítios. O min-max mapeia sempre a feature de menor gradiente para **0
exacto**, dando a impressão falsa de que não contribui — era o ponto da
supervisora sobre "atribuição exatamente zero".

```python
# antes
scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
# agora
scores = scores / (scores.max() + 1e-12)
```

Em `compare_attention_patterns` a normalização é feita com **escala comum aos
dois perfis** (`max(grad_stress.max(), grad_no_stress.max())`). Normalizar cada
um pelo seu próprio máximo poria ambos os picos a 1,00 e esconderia as
diferenças de magnitude entre a amostra de stress e a de não-stress, que é
precisamente o que a figura serve para comparar.

## `--cv loso` (2026-09-17)

Acrescentado ao `retrain_v5.py` e ao `train_sequence.py`. `LeaveOneGroupOut`,
um fold por sujeito. No WESAD são 15 folds em vez dos 5 do GroupKFold, o que
triplica os pontos de teste.

Pré-requisito verificado: os 15 sujeitos do WESAD têm ambas as classes
(~22 stress, ~40 não-stress cada), por isso a AUC é calculável em cada fold.

**Atenção à interpretação.** Com um só sujeito no fold de teste, a AUC mede
ordenação *intra-sujeito*; com GroupKFold k=5 são 3 sujeitos por fold e a
métrica também exige ordenar *entre* pessoas com basais diferentes. Daí a AUC
em LOSO (0,951) ser mais alta que em GroupKFold (0,907) sem que nenhuma esteja
errada. Não misturar os dois números na mesma tabela sem dizer isto.

---

# Scripts de 2026-09-24/26 (Model G, PR-AUC, intervalos de confiança)

Todos em `Classification Models/`. Sustentam os números da versão final do Cap. 4.

## `explain_model_g.py` — explicabilidade do Model G
```bash
python3 explain_model_g.py          # treina (~2 min), guarda pesos e analisa
python3 explain_model_g.py --load   # reutiliza saved_models/wesad_model_G.weights.h5
```
Mesma partição do Model D (seed 42, as mesmas 189 janelas). Três entradas
(série, máscara, resumo); a máscara fica fixa e nunca é atribuída. Calcula matriz
de confusão, |dy/dF| global e por classe, IG sobre as duas entradas (baseline =
média do treino), ablação por ramo, fidelidade, estabilidade, estudos de caso e
latência. Saída em `results/model_G/` (6 figuras usadas na tese +
`summary_G.csv`, `case_studies_G.csv`). Guarda **pesos** (`save_weights`), não o
modelo: as Lambdas do `build()` não serializam bem; recarregar com
`build("both", 73, 8).load_weights(...)`.

## `model_g_directions.py` — sentido dos efeitos aprendidos
Spearman IG~valor e perturbação +0,5 SD por feature resumo; série −0,5 SD;
intervalos ×0,9 coerentes; IG por batimento. Sustenta a subsecção "Summary
Features" da 4.4 (hr_mean, rmssd, nn50 invertidos; 87%; rho −0,12).

## PR-AUC — alterações a scripts existentes (backups `*.bak_prauc`)
- `retrain_v5.py`, `train_sequence.py`: coluna `prauc` (`average_precision_score`).
- `wesad_baselines.py`: `prauc`, `--out`, especificidade do majority (=100 no WESAD).

`run_pr.sh {abcd|ef|g}` corre o WESAD nos três protocolos (logs em `logs_pr/`).
Saídas: `results_wesad_pr_{random,group,loso}_{ABCD,summary,seq,both}.csv`,
`results_wesad_pr_baselines.csv`. O LOSO do Model D foi refeito inteiro
(`results_wesad_pr_loso_D.csv`) e fundido no `_ABCD`.

## `pr_report.py` — números do texto
Médias, rankings, testes emparelhados com Holm (família planeada: G ou F contra
os 7 modelos de resumo; por sujeito: 36 pares), quedas de exatidão, LOSO por
sujeito (S10). Tudo o que a 4.2, 4.6, Cap. 5, Abstract e Intro citam.

## `confidence_intervals.py` — ICs e tabelas
IC 95% com a t (`x̄ ± t·s/√n`, truncado aos limites da métrica). Escreve
`results/ci_tables.tex`: as tabelas `tab:wesad_random` e `tab:wesad_subject` com
PR-AUC e o apêndice `ap:confidence_intervals` (8 tabelas: WESAD ×3, combinado ×2,
SWELL, transferência, ablação). Recalcula as baselines do SWELL por sujeito →
`results_swell_group_baselines.csv` (LogReg 0,612, p=0,058).

## Método de entrega das correções de texto
Scripts `patch_*.py` (no scratchpad, perdidos no reinício — reconstruir se
preciso): lista de (ficheiro, termo de pesquisa, fim da seleção, texto novo);
validam que cada termo aparece uma só vez no zip, aplicam a uma cópia, verificam
chavetas e que todos os `\ref`/`\cite` resolvem, e geram o `.md` em `drafts/`.
Para verificar se a autora aplicou: comparar texto normalizado (espaços colapsados,
comentários `%` removidos) do zip novo com o texto novo de cada item.

---

# Scripts da V6 e da avaliação externa (2026-09-26, noite)

Todos em `Classification Models/` salvo indicação. `run_v6.sh {etl_wesad|etl_all|base|abcd|ef|g|pooled_abcd|pooled_efg|swell_transfer|f1f2_xai}`.
**Não editar o `run_v6.sh` enquanto uma corrida está ativa** (o bash lê o script aos
poucos e rebenta a meio). O combinado A–D por sujeito foi corrido à parte, um
processo por modelo: `results_v6_pooled_group_{A,B,C,D}.csv` (fundir antes do `--v6`).

| Script | O quê |
|---|---|
| `Data Processing/fix_durations.py` | duração real (amostras/fs) → `time_duration_v6.csv` |
| `dataset_analysis.py` | todos os números da 4.1/4.1.1 que não são redes (d de Cohen, sujeitos, blocos do SWELL, filme, baselines do combinado, regra "só sei o dataset", classificador de domínio, RF só no SWELL) |
| `corrected_ttest.py [--prefix]` | Nadeau–Bengio + Holm no split aleatório; IC do F2−F1 |
| `model_g_permutation.py` | permutação dos batimentos dentro da janela (teste certo da ordem) |
| `model_g_faithfulness_ref.py` | referência da fidelidade: queda nos 28 pares de features |
| `appendix_tables.py` | regenera as Tabelas A.1/A.2 com as durações reais |
| `compare_v5_v6.py` | V5 vs V6 por protocolo |
| `confidence_intervals.py --v6`, `pr_report.py --v6` | mesmos relatórios sobre as CSVs V6 (sem `--v6` reproduzem a V5 ao byte) |
| `drafts/simular_aplicacao.py <zip> <md>` | aplica um documento de correções a uma cópia e verifica refs/cites/gls/chavetas (os rascunhos das Partes 1 e 2 foram apagados a 2026-09-27) |

Receitas reconstruídas (os originais estavam no scratchpad): declives dos blocos e
"2.ª metade" do SWELL e classificador de domínio não reproduzem ao decimal os
valores V5 da tese (ver comentários no `dataset_analysis.py`); o resto sim.

# Limpeza do repositório (2026-09-27, aprovada pela autora)

Apagados (os que estavam no git continuam no histórico): cópias `*.bak*`,
`__pycache__/`, `.ipynb_checkpoints/`, `training_history.png`, o backup parcial do
LOSO V5; `logs_pr/` e os logs grandes de `logs_v6/` (ficam os de resumo);
`hrv_data_final/` (CSV V4), `results_v4_backup/`, `saved_models/used_model.h5`;
`main.py`, `Classification.ipynb`, `Explainability.py`, `run_explainability.py`,
`explain_wesad.py`, `gradient_class_means.py`, `measure_inference.py`, `run_pr.sh`;
`RPeakDetection_HRV_RemoteMachine.py`, `RPeakDetection_HRV.ipynb`,
`time_duration.csv`, `Obtaining data/S2_stress_ECG_values(Wesad).csv`; rascunhos
V5 já aplicados em `drafts/`. **Ficam por decisão da autora:** todos os resultados
V5 (`results_*.csv` sem v6, `results/` V5, pesos V5), o `..._old.zip` e o
`compare_v5_v6.py`. Novo `.gitignore` (pycache, checkpoints, *.bak*).
Consequência: o `stress_feedback_agent.py` refere `used_model.h5` no exemplo de uso,
que já não existe (a tese não depende disso: o agente só recebe o rótulo).

# Após a V6 (2026-09-27)

- `personalised_protocol.py` aceita `--series`, `--models G,LogReg` e
  `--scaler standard` (StandardScaler, como as baselines da tese). A corrida da tese:
  `--csv ProcessedECGData_v6_wesad.csv --series NNSeries_v6_wesad.npz --models
  G,LogReg --scaler standard --out results_v6_personalised.csv`. Sem argumentos
  reproduz a V5 (Model D, `normalize_data`).
- `explain_model_g.py`: título das figuras com `N={len(yte)}` (antes fixo em 189).
- `drafts/estado_aplicacao.py` e `drafts/simular_aplicacao.py <zip> <md>`: verificação
  de um documento de correções contra um zip.
- O README foi reescrito para a V6 (números, calibração, comandos `run_v6.sh`).
