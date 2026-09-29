# Correções da 3.ª avaliação externa — contra o zip de 29/09 (15:00)

Cada **Procura** e cada **Seleciona até** foi validado contra o zip de 29/09 (15:00) e
aparece uma só vez. Os itens são independentes, por isso a ordem de aplicação não
importa. Verificação: `drafts/simular_aplicacao.py <zip extraído> drafts/correcoes_avaliacao3.md`.

**Não há experiências novas.** Os números vêm de três verificações já feitas sobre os
resultados V6, sem treinar nenhuma rede:
- FC sozinha: `hr_only_baseline.py`;
- calibração só com repouso: `rest_normalisation.py`;
- baseline dos IG: `model_g_baseline_check.py`.

Estão no fim do `facts.md`. O que não se testou fica declarado como limitação.

**Decisões refletidas aqui (29/09):**
- A comparação SWELL/WESAD com a mesma bitola **entra no texto** (secção W). Isto
  reverte a decisão de 27/09.
- A escolha do G passa a ser discutida com as alternativas (secção G).
- O autorrelato **continua fora**, nos dois datasets (decisão de 26/09): o argumento
  assenta só na fisiologia e no protocolo.

Os itens **[verificar]** dependem de uma fonte que convém confirmar antes de aplicar.
A secção T20 é um rascunho, para reescreveres na tua voz.

## Índice por ficheiro

| Ficheiro | Itens |
|---|---|
| `preamble/Abstract.tex` | W8, F5, F6 |
| `chapters/Introduction.tex` | W11, G4, F4, F7, T7, T8, T9, T18, S3 |
| `chapters/StateOfTheArt.tex` | F8, T1, T2, T3, T4, T5, T6, T19, T20, O1 |
| `chapters/Contribution.tex` | W1, G5, X3, X4, T10, T11, T12, T13 |
| `chapters/Applications.tex` | W2–W7, W9, W12, G1, G3, G7, F1, F2, F3, F9, C1, C2, C3, X1, X2, X5, T14, T16, S1 |
| `chapters/ConclusionsAndFutureWork.tex` | W10, G6, C4, C5, T21 |
| `dissertation.bib` | T19 (entrada nova) |

---

# W. SWELL-KW e WESAD com a mesma bitola

O avaliador tem razão num ponto que já tinha sido levantado: o WESAD tem o mesmo
defeito que usamos para pôr o SWELL de parte (baseline sempre primeiro) e mais um
(fala no TSST).

Verificado no artigo original (`Tese 2026/artigos_teses/wesad/ubi_icmi2018.pdf`):
- baseline sempre primeiro; as versões A e B só trocam a ordem entre diversão e stress (Fig. 2);
- "to induce variance in the subjects' posture, the baseline, amusement and stress
  conditions were conducted either standing or sitting. For each condition,
  approximately half of the subjects were standing and the other half were sitting";
- "the study participants spoke during the TSST. Hence, the classifiers might have
  partially learned to distinguish between speaking (stress condition) and
  non-speaking episodes".

A postura, portanto, **não** é um confundidor ao nível do dataset. A ordem e a fala são.

O argumento que sobra, e que o texto passa a usar, é o **sentido** da resposta:
- no WESAD a resposta vai no sentido do stress agudo em 15 de 15 participantes;
- um efeito de ordem como o proposto para o SWELL reduziria a diferença no WESAD, em vez
  de a criar;
- a fala não se separa do stressor, por isso o que se deteta é a condição TSST.

## W1. 3.3 — o protocolo do WESAD
**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `The stressed state was induced with the \gls{TSST}, which consists of a public speaking and a mental arithmetic task performed in front of a panel \citep{WESAD}.`

**Substitui por:**
```latex
The stressed state was induced with the \gls{TSST}, which consists of a public speaking and a mental arithmetic task performed in front of a panel \citep{WESAD}. The baseline, 20 minutes of reading neutral magazines, was recorded first in every participant, and the two versions of the protocol differ only in the order of the amusement and stress conditions. To vary posture, approximately half of the participants stood and half sat in each condition; during the \gls{TSST} they spoke, and the authors of the dataset note that classifiers may partly learn to distinguish speaking from not speaking \citep{WESAD}.
```

## W2. 4.1 — "The cause is documented"
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The cause is documented in the design of SWELL-KW by its own authors, who state that`

**Substitui por:**
```latex
The most likely explanation is documented in the design of SWELL-KW by its own authors, who state that
```

## W3. 4.1 — o bloco neutro não decai: o mecanismo não fica identificado
O avaliador nota que o p = 0,78 não apoia o efeito de ordem: se o bloco neutro estivesse
elevado pela novidade, devia descer. O texto atual lê-o como "elevação sustentada".

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The neutral block does not decay over its duration:`
**Seleciona até:** `preserves the inversion, at $-4.46$ bpm.`

**Substitui por:**
```latex
The data do not identify the mechanism. Heart rate does not decline over the neutral block: a linear fit of heart rate against the position of the window within the block, centred for each recording, gives a change of $+0.13$ bpm from start to end ($p = 0.78$), so there is no sign of habituation within it, and restricting the comparison to the second half of the neutral block preserves the inversion, at $-4.46$ bpm. The stress blocks, in contrast, rise over their duration, by 2.24 bpm under interruptions and 6.45 bpm under time pressure ($p < 0.001$); because windows within a block are not independent, these p-values are indicative. The stressors therefore have an effect, but it starts from a lower level than that of the neutral block. The authors of SWELL-KW note that, because every session started with the neutral condition, participants might have experienced order or fatigue effects \citep{SWELL}; the data are consistent with that explanation but cannot confirm it, since no participant recorded the neutral condition after a stressor.
```

## W4. 4.1 — as outras explicações (qualidade do sinal, stressores fracos)
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `A classifier trained on these labels therefore learns the order in which the blocks were recorded rather than the participant's state.`

**Substitui por:**
```latex
Two other explanations can be set against the data. Signal quality does not account for the reversal: the windows excluded in SWELL-KW are 20.0\% of the no-stress and 19.7\% of the stress windows (Section \ref{etl}), a rate too similar between the classes to reverse the direction of every feature. Weak stressors would explain a small effect, which SWELL-KW shows, but not a reversal of the same sign in almost every participant. That requires a factor that differs systematically between the neutral and the stress blocks, and the fixed order is the one that the design documents. A classifier trained on these labels can therefore learn the order in which the blocks were recorded rather than the participant's state.
```

## W5. 4.1 — o WESAD sob o mesmo escrutínio
Parágrafo novo, antes da frase que liga ao Prajod e ao Vos.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `These findings align with the evidence found by \citet{prajod2024stressor} and \citet{vos2023ensemble}, as established in Section \ref{major_findings}.`

**Substitui por:**
```latex
The same scrutiny applies to WESAD. Its baseline was also recorded first in every participant, and its stress condition differs from the baseline in more than the stressor: during the \gls{TSST} the participants speak, which alters breathing, and breathing modulates the intervals between beats (Section \ref{data}). Posture is not a confounder at the level of the dataset, since approximately half of the participants stood and half sat in each condition \citep{WESAD}. What separates the two datasets is the direction of the response. In WESAD every feature moves as the physiology of acute stress predicts and mean heart rate rises in all 15 participants, and an order effect of the kind proposed for SWELL-KW, a first condition recorded at a raised level, would reduce that difference rather than create it. The contribution of speech, however, cannot be separated from that of the stressor: what the models of this chapter detect is the \gls{TSST} condition, which includes speaking. WESAD is used as the primary evaluation not because its labels are free of confounders, but because its response has the direction and the size expected of acute stress.

These findings align with the evidence found by \citet{prajod2024stressor} and \citet{vos2023ensemble}, as established in Section \ref{major_findings}.
```

## W6. 4.1.1 — "validated physiological state"
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `on the dataset whose labels correspond to a validated physiological state, the same features`

**Substitui por:**
```latex
on WESAD, whose stress condition produces the response expected of acute stress, the same features
```

## W7. 4.2 — resumo da secção
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `is feasible when the label corresponds to a validated physiological state:`

**Substitui por:**
```latex
is feasible for a standardised laboratory stressor whose response has the direction expected of acute stress:
```

## W8. Abstract e Resumo — "validated stress response"
**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `On WESAD, whose labels correspond to a validated stress response,`

**Substitui por:**
```latex
On WESAD, whose stress condition is a standardised laboratory stressor,
```

**Procura:** `No WESAD, cujos rótulos correspondem a uma resposta de stress validada,`

**Substitui por:**
```latex
No WESAD, cuja condição de stress é um stressor laboratorial padronizado,
```

## W9. 4.6 — abertura da discussão
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `Stress can be detected from a single 30-second \gls{ECG} window when the label corresponds to a physiological state:`

**Substitui por:**
```latex
Stress can be detected from a single 30-second \gls{ECG} window when the label marks a stressor that produces the expected physiological response:
```

**Procura:** `shows that the cause is the labels rather than the models.`

**Substitui por:**
```latex
points to the labels rather than the models.
```

## W10. Conclusões — "the cause lies in"
**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `and the cause lies in the design of \gls{SWELL-KW}, where the neutral condition was always recorded first.`

**Substitui por:**
```latex
and the most likely explanation is the design of \gls{SWELL-KW}, where the neutral condition was always recorded first. The baseline of WESAD was also recorded first, but there the response has the direction expected of acute stress; its stress condition also involves speaking, whose contribution cannot be separated from that of the stressor.
```

**Procura:** `when the label corresponds to a physiological state,`

**Substitui por:**
```latex
when the label marks a stressor that produces the expected physiological response,
```

## W11. Introdução, Contribuição 1 — "traced to" e o Prajod
Também responde à crítica 2.11: a contribuição passa a dizer o que estende.

**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `and the cause is traced to the design of \gls{SWELL-KW}.`

**Substitui por:**
```latex
and the most likely explanation is the design of \gls{SWELL-KW}, in which the neutral condition was always recorded first. This extends the loss of performance between the two datasets reported by \citet{prajod2024stressor}: an \gls{AUC} below 0.5 shows that their labels disagree, not only that models fail to generalise.
```

## W12. Limitações — os confundidores do WESAD
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `and the results do not establish that the same discrimination is possible for the more diffuse stressors of daily life.`

**Substitui por:**
```latex
and the results do not establish that the same discrimination is possible for the more diffuse stressors of daily life. Within WESAD, the baseline was always recorded first and the participants speak during the stress condition; the contribution of speech to the detected response was not separated from that of the stressor.
```

---

# G. A escolha do Model G

A escolha não muda, mas o texto deixa de a apoiar num resultado que não sobrevive ao
teste corrigido, e passa a dizer quais são as alternativas.

## G1. 4.6.1 — primeiro parágrafo
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The choice rests on the one difference between models that this chapter measured with statistical support:`
**Seleciona até:** `shows that Model G uses that order: shuffling the intervals within each window costs 13.19 pp of accuracy.`

**Substitui por:**
```latex
The data do not single out one model, and the choice of Model G rests on a design argument rather than on a measured superiority. Under the random split, the two models that read the \gls{NN} interval series exceed the models built on the summary features, and with the paired t-test several of these differences survive Holm correction, including the gain of Model G over Model E, which shares its training procedure and differs from it only in the series (0.020 of \gls{AUC} and 0.029 of PR-AUC). With the corrected resampled test, which accounts for the overlap between random partitions, that gain does not survive, and the advantage of the series is robust only for Model F. Models A to D receive the eight summary features reshaped as a sequence of length eight, an ordering that has no temporal meaning, while Models F and G receive the only input in the data that is ordered in time, and the permutation test of Section \ref{sec:branch_ablation} shows that Model G uses that order: shuffling the intervals within each window costs 13.19 pp of accuracy. Model G was retained because it reads the series and, through its summary branch, also the statistics that the literature reports, and because it has the highest mean \gls{AUC} under both subject-wise protocols; neither reason is a significant difference.
```

## G2. (sem alteração)
A frase "Model F, which is smaller and has a single input, would be an equally defensible
choice" (l.404) fica como está.

## G3. 4.6.1 — as alternativas mais simples
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `What the neural models add is the ability to read the series, which a linear model on eight statistics cannot do; it pays off under the random split, and whether it pays off on new subjects is not established by 15 of them.`

**Substitui por:**
```latex
What the neural models add is the ability to read the series, which a linear model on eight statistics cannot do; it pays off under the random split, and whether it pays off on new subjects is not established by 15 of them. On new subjects, a logistic regression on the mean heart rate alone reaches an \gls{AUC} of 0.889 under GroupKFold and 0.925 under leave-one-subject-out, and Model G, which has the highest mean, does not exceed it significantly (Section \ref{sec:wesad_results}). A logistic regression whose inputs are centred on each subject's rest reaches, without any stress window of that subject, an accuracy that does not differ significantly from that of Model G with its own threshold (Section \ref{sec:discussion_threshold}). For a system whose motivation includes explainability, that simpler configuration is a serious alternative: its coefficients can be read one by one, which Section \ref{sec:discussion_xai} shows is not the case for Model G.
```

## G4. Introdução, Contribuição 3 — "under a random split"
**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `\item Evidence that the input representation matters more than the architecture. Under a random split,`

**Substitui por:**
```latex
\item Evidence that, under a random split, the input representation matters more than the architecture. Under that protocol,
```

**Procura:** `Shuffling the beats within each window shows that the model uses their order. Under subject-wise partitioning, no difference between models is measurable.`

**Substitui por:**
```latex
Shuffling the beats within each window shows that the model uses their order. Under subject-wise partitioning, no difference between models is measurable, and Model G does not significantly exceed a logistic regression on the mean heart rate alone.
```

## G5. 3.5 — "which isolates the variable under test"
F e C/D mudam também a receita de treino. A comparação limpa é G − E.

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `in order to preserve the architecture family of Models C and D and change only the input representation, which isolates the variable under test, and because`

**Substitui por:**
```latex
in order to preserve the architecture family of Models C and D, and because
```

**Procura:** `so that the model sees both the interval series and the statistics computed from it.`

**Substitui por:**
```latex
so that the model sees both the interval series and the statistics computed from it. Models E and G share the summary branch and the training procedure, so their comparison isolates the contribution of the series; Models F and D differ in the training procedure as well as in the input (Section \ref{traintest}).
```

## G6. Conclusões — "Most of these differences survive"
**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `The two exceed the models built on the summary features. Most of these differences survive correction for multiple comparisons, and with a test corrected for the overlap between random partitions the most robust is the advantage of Model F in PR-AUC.`

**Substitui por:**
```latex
The two exceed the models built on the summary features under that split, but once the overlap between random partitions is accounted for, the advantage is robust mainly for Model F, in PR-AUC, and that of Model G survives only against Model A.
```

**Procura:** `Under subject-wise partitioning, no difference between models is measurable on 15 subjects.`

**Substitui por:**
```latex
Under subject-wise partitioning, no difference between models is measurable on 15 subjects, and Model G does not significantly exceed a logistic regression on the mean heart rate alone, which reaches an \gls{AUC} of 0.889 under GroupKFold and 0.925 under leave-one-subject-out.
```

## G7. Limitações — nested CV e procura de hiperparâmetros
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `and its reported performance is therefore an optimistic estimate.`

**Substitui por:**
```latex
and its reported performance is therefore an optimistic estimate. A nested cross-validation, in which the choice is repeated inside each training partition, would remove this bias and is feasible at about two minutes per training run; it was not carried out. No hyperparameter search was carried out for any model, and the neural models were trained with two recipes, one for Models A to D and one for Models E to G (Section \ref{traintest}).
```

---

# F. A frequência cardíaca sozinha (crítica 2.2)

Números de `hr_only_baseline.py`: regressão logística sem pesos, só com `hr_mean`,
nos mesmos folds da tese. A LogReg de 8 features reproduz a da tese ao decimal.

## F1. Tabela `tab:wesad_subject` — linha nova e legenda
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `Majority classifier & 64.06 $\pm$ 0.54 & 0.500 & 0.359 $\pm$ 0.005 & 64.08 $\pm$ 1.06 & 0.500 & 0.359 $\pm$ 0.011 \\ \hline`

**Substitui por:**
```latex
Majority classifier & 64.06 $\pm$ 0.54 & 0.500 & 0.359 $\pm$ 0.005 & 64.08 $\pm$ 1.06 & 0.500 & 0.359 $\pm$ 0.011 \\ \hline
Heart rate only (logistic regression) & 78.48 $\pm$ 11.27 & 0.889 $\pm$ 0.099 & 0.848 $\pm$ 0.139 & 78.25 $\pm$ 18.00 & 0.925 $\pm$ 0.116 & 0.902 $\pm$ 0.142 \\ \hline
```

**Procura:** `Performance on WESAD under subject-wise partitioning: GroupKFold with five folds and leave-one-subject-out over the 15 subjects.`

**Substitui por:**
```latex
Performance on WESAD under subject-wise partitioning: GroupKFold with five folds and leave-one-subject-out over the 15 subjects. The heart-rate-only row is a logistic regression whose only input is hr\_mean, a reference for the discrimination that the mean heart rate provides on its own.
```

## F2. 4.2 — parágrafo novo depois de "not measurable at this sample size"
Responde também à crítica 2.10 (potência): em vez de uma análise de potência, dá o
intervalo da diferença, que não permite afirmar nem superioridade nem equivalência.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `and that the differences between them are not measurable at this sample size.`

**Substitui por:**
```latex
and that the differences between them are not measurable at this sample size.

Most of that discrimination is carried by the mean heart rate. A logistic regression whose only input is hr\_mean reaches an \gls{AUC} of 0.889 under GroupKFold and 0.925 under leave-one-subject-out (Table \ref{tab:wesad_subject}), within the range of the trained configurations. Model G, which has the highest mean under both protocols, exceeds it by 0.030 ($p = 0.36$) and 0.045 ($p = 0.079$). Under leave-one-subject-out, the 95\% confidence interval of the difference in \gls{AUC} between Model G and logistic regression on the eight features, $-0.024$ to $+0.086$, is compatible both with no difference and with an advantage of 0.086 for Model G, so the data establish neither superiority nor equivalence. On unseen subjects, 15 subjects do not establish that the interval series or the variability features add to the mean heart rate.
```

## F3. 4.2 — sensibilidade por sujeito (tabela nova)
O avaliador nota que a sensibilidade, que a 3.6.1 chama a métrica que mais importa, não
aparece nas tabelas por sujeito. Os valores vêm das CSVs da V6 (média ± DP entre folds).

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `This dissociation between ranking and decision is the empirical basis for the per-subject calibration tested in Section \ref{sec:discussion_threshold}.`

**Substitui por:**
```latex
This dissociation between ranking and decision is the empirical basis for the per-subject calibration tested in Section \ref{sec:discussion_threshold}.

Table \ref{tab:wesad_subject_sens} reports sensitivity and specificity under the two subject-wise protocols. On unseen subjects the neural models detect a larger share of the stress windows than logistic regression and the random forest, at the cost of specificity, as the class weights of Section \ref{traintest} intend, and Model G reaches similar values for both, 82.85\% and 83.91\% under leave-one-subject-out, whereas under the random split its sensitivity is well below its specificity (Table \ref{tab:wesad_random}). The standard deviations across folds, between 6 and 41 pp, show that for some subjects most of the stress windows are missed.

\begin{table}[h!]
\centering
\small
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Model} & \multicolumn{2}{c|}{\textbf{GroupKFold}} & \multicolumn{2}{c|}{\textbf{LOSO}} \\ \cline{2-5}
 & \textbf{Sensitivity (\%)} & \textbf{Specificity (\%)} & \textbf{Sensitivity (\%)} & \textbf{Specificity (\%)} \\ \hline
Heart rate only (logistic regression) & 57.84 $\pm$ 21.56 & 90.03 $\pm$ 15.52 & 57.13 $\pm$ 41.42 & 89.40 $\pm$ 29.02 \\ \hline
Logistic regression & 73.78 $\pm$ 11.38 & 89.31 $\pm$ 15.36 & 71.59 $\pm$ 32.74 & 88.18 $\pm$ 26.11 \\ \hline
Random forest & 74.55 $\pm$ 6.17 & 84.87 $\pm$ 14.03 & 72.68 $\pm$ 30.23 & 85.26 $\pm$ 25.56 \\ \hline
Model E (Summary MLP) & 82.79 $\pm$ 10.60 & 79.37 $\pm$ 16.75 & 81.76 $\pm$ 26.50 & 78.23 $\pm$ 28.80 \\ \hline
Model A (LSTM) & 81.55 $\pm$ 10.25 & 76.09 $\pm$ 15.89 & 83.01 $\pm$ 26.59 & 69.69 $\pm$ 38.70 \\ \hline
Model B (Deep LSTM) & 76.95 $\pm$ 11.92 & 74.20 $\pm$ 18.68 & 77.01 $\pm$ 28.91 & 73.98 $\pm$ 33.56 \\ \hline
Model C (CNN-LSTM) & 82.48 $\pm$ 10.10 & 78.32 $\pm$ 15.65 & 83.01 $\pm$ 27.61 & 79.27 $\pm$ 26.63 \\ \hline
Model D (CNN-Transformer) & 77.25 $\pm$ 6.98 & 84.34 $\pm$ 14.17 & 78.86 $\pm$ 31.75 & 80.13 $\pm$ 26.36 \\ \hline
Model F (Sequence CNN) & 82.23 $\pm$ 11.55 & 77.34 $\pm$ 22.67 & 84.08 $\pm$ 27.02 & 81.51 $\pm$ 29.54 \\ \hline
Model G (Dual-Input CNN) & 81.05 $\pm$ 12.20 & 79.75 $\pm$ 21.15 & 82.85 $\pm$ 26.92 & 83.91 $\pm$ 26.12 \\ \hline
\end{tabular}
\caption{Sensitivity and specificity on WESAD under subject-wise partitioning, at the threshold of 0.5. Mean $\pm$ standard deviation across folds.}
\label{tab:wesad_subject_sens}
\end{table}
```

## F4. Introdução, Contribuição 2
**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `and between 0.913 and 0.970 under leave-one-subject-out, while per-subject accuracy ranges from 36.1\% to 100\%,`

**Substitui por:**
```latex
and between 0.913 and 0.970 under leave-one-subject-out, values that a logistic regression on the mean heart rate alone approaches (0.889 and 0.925), while per-subject accuracy ranges from 36.1\% to 100\%,
```

## F5. Abstract e Resumo — a FC sozinha
**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `and between 0.91 and 0.97 when ranking the windows of each unseen subject separately.`

**Substitui por:**
```latex
and between 0.91 and 0.97 when ranking the windows of each unseen subject separately; the mean heart rate alone reaches 0.89 and 0.93.
```

**Procura:** `e entre 0,91 e 0,97 ao ordenar as janelas de cada sujeito não visto em separado.`

**Substitui por:**
```latex
e entre 0,91 e 0,97 ao ordenar as janelas de cada sujeito não visto em separado; a frequência cardíaca média, sozinha, atinge 0,89 e 0,93.
```

## F6. Abstract e Resumo — as 907 janelas da avaliação principal
**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `SWELL-KW and WESAD, were processed into 5,409 windows from 37 subjects,`

**Substitui por:**
```latex
SWELL-KW and WESAD, were processed into 5,409 windows from 37 subjects, of which the 907 windows of the 15 WESAD subjects form the main evaluation,
```

**Procura:** `foram processados em 5409 janelas de 37 sujeitos,`

**Substitui por:**
```latex
foram processados em 5409 janelas de 37 sujeitos, das quais as 907 janelas dos 15 sujeitos do WESAD constituem a avaliação principal,
```

## F7. Introdução — o domínio temporal a 30 s (contradição com a Discussão)
**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `Therefore, at thirty seconds a substantial part of the \gls{HRV} feature set is structurally unreliable, and any stress detection system operating at this timescale must be built on the time-domain subset alone.`

**Substitui por:**
```latex
Therefore, at thirty seconds a substantial part of the \gls{HRV} feature set is structurally unreliable, and a stress detection system operating at this timescale must be built on the time-domain subset. Even within that subset the minimum durations differ: in one of the studies compiled by \citet{shaffer2020ultrashort}, heart rate was valid from 10 seconds, whereas RMSSD, SDNN and pNN50 required 60 seconds.
```

## F8. SotA — "a 30 s duration is suitable ... using time domain features"
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `And thus a 30 s duration is suitable for calculating \gls{HRV} metrics using time domain features but not necessarily frequency domain features.`

**Substitui por:**
```latex
At 30 s the frequency-domain features are therefore excluded, and the time-domain features are the usable subset, although not all of them have been validated at that length: in one of the studies compiled by \citet{shaffer2020ultrashort}, heart rate was valid from 10 s, whereas SDNN, RMSSD and pNN50 required 60 s.
```

## F9. 4.6.4 — Relation to Previous Work
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `a window below the one-minute floor is sufficient, particularly when the model reads the interval series directly instead of statistics whose estimation variance is high on 30 to 50 intervals.`

**Substitui por:**
```latex
a window below the one-minute floor is sufficient to detect the stress condition.
```

**Procura:** `In this work, the mean heart rate of the window alone separates the two classes of the WESAD test partition with an \gls{AUC} of 0.882 (Section \ref{sec:explainability}).`

**Substitui por:**
```latex
In this work, the mean heart rate of the window alone separates the two classes with an \gls{AUC} of 0.882 under the random split (Section \ref{sec:explainability}), and a logistic regression on it reaches 0.889 and 0.925 on unseen subjects, not significantly below Model G (Section \ref{sec:wesad_results}). At 30 seconds the detection therefore rests mainly on the measure that the ultra-short literature validates at the shortest durations, and the contribution of the variability features on unseen subjects remains to be shown.
```

---

# C. Calibração por pessoa (crítica 2.7)

Números de `rest_normalisation.py`: mesmo LOSO e mesmas janelas de teste da 4.6.3.
Cada sujeito é centrado na média das suas janelas de não-stress da 1.ª metade (~10 min
de repouso), e não se usa nenhuma janela de stress do sujeito de teste. Não foi
corrido com o G, e o texto di-lo.

## C1. Tabela `tab:personalised` — linha nova e legenda
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `\multirow{3}{*}{Logistic regression} & Threshold 0.5 & 81.40 $\pm$ 18.32 & 38.7 & 0.0 & 3 \\ \cline{2-6}`
**Seleciona até:** `& Retrained & 81.84 $\pm$ 18.19 & 38.7 & 0.0 & 3 \\ \hline`

**Substitui por:**
```latex
\multirow{4}{*}{Logistic regression} & Threshold 0.5 & 81.40 $\pm$ 18.32 & 38.7 & 0.0 & 3 \\ \cline{2-6}
 & Own threshold & 85.92 $\pm$ 16.46 & 37.9 & 31.6 & 2 \\ \cline{2-6}
 & Retrained & 81.84 $\pm$ 18.19 & 38.7 & 0.0 & 3 \\ \cline{2-6}
 & Centred on own rest & 88.22 $\pm$ 14.52 & 62.1 & 73.7 & 3 \\ \hline
```

**Procura:** `Each subject's windows are split in time: the first half calibrates, the second half is the test set of all three conditions.`

**Substitui por:**
```latex
Each subject's windows are split in time: the first half calibrates, the second half is the test set of all conditions. In the last condition the inputs of every subject, in training and in testing, are centred on the mean of that subject's no-stress calibration windows, and no stress window of the test subject is used.
```

## C2. 4.6.3 — o limiar "1,00" e a calibração só com repouso
O "1.00" é 0,997 (S10 e S11). O avaliador leu-o como "nunca prever stress", o que é
falso: o S10 fica com especificidade 100% e sensibilidade 66,7%.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The thresholds chosen for Model G range from 0.04 to 1.00, with a median of 0.65, which is a direct measurement of the problem described above.`

**Substitui por:**
```latex
The thresholds chosen for Model G range from 0.04 to 0.997, with a median of 0.65, which is a direct measurement of the problem described above; for subjects S10 and S11 the outputs of the model lie so close to 1 that the chosen threshold is 0.997, a value that small changes in the signal between sessions could move.

The threshold requires stress windows of the subject. A calibration that needs only rest is the one used by \citet{prajod2024stressor} and \citet{byun2025automated}, who normalised the features of each participant by that participant's own data (Chapter \ref{cha:estado_de_arte}). Applied here to logistic regression, centring the eight features of each subject on the mean of their no-stress calibration windows, about ten minutes at rest, raises the accuracy from 81.40\% to 88.22\% ($p = 0.056$; Wilcoxon signed-rank $p = 0.047$) without any stress window of the test subject, and the result does not differ significantly from Model G with its own threshold ($-2.91$ pp, $p = 0.42$). Centring the mean heart rate alone gives 87.32\%. The worst subject, at 62.1\%, remains below that of Model G with its own threshold. This condition was not tested with Model G.
```

## C3. 4.6.3 — limites
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The calibration requires labelled windows of both classes, including a stress episode, which a real user would have to provide.`

**Substitui por:**
```latex
The threshold calibration requires labelled windows of both classes, including a stress episode, which a real user would have to provide; centring on rest removes that requirement, but was tested only with logistic regression.
```

## C4. Conclusões — a calibração só com repouso
**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `raises the accuracy of Model G on the second half from 82.1\% to 91.1\%, without retraining.`

**Substitui por:**
```latex
raises the accuracy of Model G on the second half from 82.1\% to 91.1\%, without retraining, and centring the inputs of a logistic regression on about ten minutes of each subject's rest, without any stress window, raises its accuracy from 81.4\% to 88.2\%.
```

## C5. Trabalho futuro — calibração
**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `should be measured on unseen recording sessions \citep{haque2023stateofart}.`

**Substitui por:**
```latex
should be measured on unseen recording sessions \citep{haque2023stateofart}, together with the calibration that needs only a recording at rest, which was tested here only with logistic regression.
```

---

# X. Explicabilidade: o baseline dos integrated gradients (crítica 2.6)

Números de `model_g_baseline_check.py`, com os pesos V6 da 4.4 e sem treino:
- nas janelas reais de treino, os pesos de classe só levam a P(stress) média da
  prevalência, 36,0%, para 39,9%;
- a baseline da tese dá 0,916;
- a mesma série com a variabilidade real de cada janela, centrada na média do treino, dá 0,765;
- o resto vem de a janela média (~82 bpm) ficar do lado do stress da fronteira.

## X1. 4.4.4 — a origem do 0,916
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The offset is consistent with the class weighting described in Section \ref{traintest}, although this was not tested by training without weights; the weighting gives the stress class the larger weight in a dataset where it accounts for 35.94\% of windows.`

**Substitui por:**
```latex
The class weighting of Section \ref{traintest} does not explain this offset: on the real training windows it raises the mean predicted probability only from the 36.0\% of stress windows to 39.9\%. The offset comes from the baseline itself. Its series is flat, an input that does not occur in training, and a window without beat-to-beat variability resembles the stress windows, whose variability is lower; restoring the variability of each window while keeping its mean at the training mean lowers the prediction from 0.916 to 0.765. The remainder is the position of the mean window, at a heart rate of about 82 bpm, on the stress side of the decision boundary of Model G.
```

## X2. 4.4.3 — o falso positivo: o erro é do modelo, não da baseline
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The error is a consequence of the offset of the baseline more than of any feature of the window.`

**Substitui por:**
```latex
The error is the model's: a window whose heart rate is at the training mean lies on the stress side of its boundary, as the baseline itself does (Section \ref{sec:xai_disagreement}), and the attributions describe how far the other properties of the window move it back, not why it starts there.
```

## X3. 3.7 — descomentar a justificação para não usar SHAP
**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `%SHAP, the most widely used post-hoc method`

**Substitui por:**
```latex
SHAP, the most widely used post-hoc method
```

## X4. 3.7 — a baseline é uma série plana
**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `The baseline for the integrated gradients sets every real interval of the series to the mean interval of the training set and the eight summary features to their training means, which after the standardisation described in Section \ref{models} is zero for both inputs.`

**Substitui por:**
```latex
The baseline for the integrated gradients sets every real interval of the series to the mean interval of the training set and the eight summary features to their training means, which after the standardisation described in Section \ref{models} is zero for both inputs. The resulting series is flat, an input that does not occur in training; its consequences are examined in Section \ref{sec:xai_disagreement}.
```

## X5. Limitações — a baseline
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `and the integrated gradients depend on a baseline to which the model assigns a probability of 0.916.`

**Substitui por:**
```latex
and the integrated gradients depend on a flat baseline, outside the training distribution, to which the model assigns a probability of 0.916; a baseline drawn from real windows was not tested.
```

---

# T. Erros e incongruências de texto

## T1. SotA — a adrenalina não liberta cortisol
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `the body activates the sympathetic branch of the ANS, which releases adrenaline and noradrenaline, which releases cortisol.`

**Substitui por:**
```latex
the body activates the sympathetic branch of the ANS, which releases noradrenaline and, through the adrenal medulla, adrenaline, and the hypothalamic-pituitary-adrenal axis, which releases cortisol from the adrenal cortex.
```

## T2. SotA — "not a treatment", "proved"
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Although there is not a treatment for stress, there are stress reduction strategies`

**Substitui por:**
```latex
Stress has no single treatment, but there are stress reduction strategies
```

**Procura:** `These are all methods proved to reduce general stress`

**Substitui por:**
```latex
These methods have been shown to reduce general stress
```

## T3. SotA — gradient descent não garante o erro mínimo
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `This optimiser gets the lowest error value possible or increases the accuracy of the model by changing the weights of the model while iterating the training data.`

**Substitui por:**
```latex
This optimiser changes the weights of the model in the direction that decreases the loss, iterating over the training data until it reaches a minimum, which is not guaranteed to be the global one.
```

## T4. SotA — "RNNs have the deepest structures"
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `RNNs have the deepest structures amongst the \gls{DL} algorithms, which are able to map sequential input data to their output.`

**Substitui por:**
```latex
RNNs are \gls{DL} algorithms that map sequential input data to their output.
```

## T5. SotA — "Research proved" e a fonte sobre avatares (`ml12`)
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Research proved that if the network had more hidden layers,`
**Seleciona até:** `These types of neural networks are known as \gls{DL} \citep{ml17, ml15,ml12}.`

**Substitui por:**
```latex
Networks with more hidden layers, that is, more layers between the input and the output layer, can represent more complex functions of their input, at a higher computational cost, and are known as \gls{DL} networks \citep{ml17, ml15}.
```

## T6. SotA — ultra-curto é < 5 min
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `the use of ultra-short \gls{HRV} signals ($\leq$ 5 minutes)`

**Substitui por:**
```latex
the use of ultra-short \gls{HRV} signals (shorter than 5 minutes)
```

## T7. Introdução — "must" vs "not settled" (AI Act)
**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `A stress detection system intended for deployment must therefore be able to account for its own predictions,`

**Substitui por:**
```latex
If a stress detection system falls under these rules, which is not settled (Section \ref{soa:EU_rules}), it will have to be able to account for its own predictions,
```

## T8. Introdução, objetivos — "common ... training protocol"
**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `- Assess all architectures on a common dataset with a common preprocessing and training protocol.`

**Substitui por:**
```latex
- Assess all architectures on the same data, partitions and evaluation protocol.
```

## T9. Introdução, objetivos — durações de validação
As tabelas de features do SotA não têm durações. O que a tese revê é a fiabilidade a
durações ultra-curtas (secção `soa:reliability`).

**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `- Review the features employed in the stress detection literature and the durations at which they were validated.`

**Substitui por:**
```latex
- Review the features employed in the stress detection literature and the evidence on their reliability at ultra-short durations.
```

## T10. 3.5 — o Model A não tem batch normalisation
**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `All four models use dropout regularisation (ranging from 0.2 to 0.4), batch normalisation for training stability, ReLU activation functions in hidden layers, and Sigmoid activation in the output layer for binary stress classification.`

**Substitui por:**
```latex
All four models use dropout regularisation (ranging from 0.2 to 0.4), ReLU activation functions in hidden layers and Sigmoid activation in the output layer for binary stress classification; Models B to D also use batch normalisation for training stability.
```

## T11. 3.5 — "temporal dependencies in the HRV time series" nos A–D
Contradiz a l.165 e a l.195 (os A–D não recebem séries temporais).

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `This deeper architecture with multiple LSTM layers allows the model to capture complex temporal dependencies in the \gls{HRV} time series data, at the cost of increased training complexity and computational requirements.`

**Substitui por:**
```latex
This deeper architecture with multiple LSTM layers was designed to capture complex temporal dependencies in sequential data, at the cost of increased training complexity and computational requirements.
```

**Procura:** `This progressive architecture enables the extraction of multi-scale temporal patterns from the \gls{HRV} features, with each layer capturing increasingly complex relationships.`

**Substitui por:**
```latex
This progressive architecture was designed to extract patterns at several scales from sequential data, with each layer capturing increasingly complex relationships.
```

## T12. 3.4 — o filtro: tipo e ordem
Verificado em `Data Processing/filtering_wesad.ipynb`: `scipy.signal.butter(2, ...)` +
`filtfilt`, na gravação inteira.

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `a band-pass filter at 0.5 Hz and 40 Hz was applied to remove those interferences.`

**Substitui por:**
```latex
a second-order Butterworth band-pass filter between 0.5 Hz and 40 Hz was applied forward and backward, so without phase distortion, to each complete recording to remove those interferences.
```

## T13. 3.4 — deteção dos picos por segmento
Verificado no BioSPPy 0.8.0:
- o `ecg()` aplica um FIR 3–45 Hz de ordem int(0,3·fs);
- o Hamilton inicializa com os primeiros 8 s (`init_ecg = 8`);
- o ETL corre-o em cada segmento.

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `Secondly, the R peaks of each window were detected with the Hamilton segmenter, the default method of the \texttt{ecg} function of BioSPPy \citep{r_peaks_biosppy}.`

**Substitui por:**
```latex
Secondly, the R peaks of each window were detected with the Hamilton segmenter, the default method of the \texttt{ecg} function of BioSPPy \citep{r_peaks_biosppy}, which first applies its own finite impulse response band-pass filter between 3 Hz and 45 Hz. Because detection runs on each 30-second segment separately, the segmenter initialises its thresholds on the first seconds of every segment, and the interval between the last beat of one segment and the first beat of the next is not used. The detected peaks were not validated against manual annotations.
```

## T14. 4.1 — "All 22 participants" sem a ressalva do apêndice
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `whereas all 15 WESAD participants show a higher one, at $+20.78$ bpm.`

**Substitui por:**
```latex
whereas all 15 WESAD participants show a higher one, at $+20.78$ bpm. Excluding the windows of the relaxation phase that precedes each block of SWELL-KW, which carry the label of the block (Appendix \ref{ap:datasets}), the reversal holds in 18 of the 20 participants who retain windows of both conditions.
```

## T15. (incluído no F6)

## T16. 4.1 — ICs da transferência: sobre seeds, não sobre sujeitos
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `for the six that come from five runs the upper limit of the 95\% confidence interval is also below 0.5, at most 0.494 (Table \ref{tab:ci_transfer}).`

**Substitui por:**
```latex
for the six that come from five runs the upper limit of the 95\% confidence interval is also below 0.5, at most 0.494 (Table \ref{tab:ci_transfer}). These intervals describe the variation between training runs, not between subjects.
```

## T17. (incluído no F3)

## T18. Introdução — citações dos smartwatches
O `article6`, o `article9` e o `article20` não são sobre smartwatches. O `article10`
(Maritsch 2019) é, mas usa PPG. O `velmovitsky2022apple`, já no .bib e citado no SotA,
usa o ECG de 30 s do Apple Watch. Corrige também a gralha "the the".

**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `a constraint imposed by the the device rather than by the physiology being measured \citep{article10, article6, article9, article20}.`

**Substitui por:**
```latex
a constraint imposed by the device rather than by the physiology being measured \citep{velmovitsky2022apple}.
```

**Procura:** `and with recent developments smartwatches can now measure with high precision ultra short-term ECGs - with a length of thirty seconds.`

**Substitui por:**
```latex
and some smartwatches can now record a single-lead ECG of thirty seconds on demand \citep{velmovitsky2022apple}.
```

**Procura:** `Thus, the use of wearables for calculating \gls{HRV} data benefits individuals by providing continuous feedback about their stress levels in real time, anytime, anywhere \citep{article6, article20, article10, article9}.`

**Substitui por:**
```latex
Wearables could therefore give individuals feedback about their stress levels during daily life, provided that stress can be detected from recordings of that length \citep{article10, velmovitsky2022apple}.
```

## T19. SotA — features geométricas e não lineares nunca tratadas **[verificar]**
Confirmar no original a relação SD1/SD2 com SDNN e SDSD (Brennan et al. 2001) e, no
Task Force, os 20 minutos mínimos para os métodos geométricos.

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `geometric, and nonlinear \citep{article3} and recent work has shown that`

**Substitui por:**
```latex
geometric, and nonlinear \citep{article3}. Geometric measures require recordings of at least 20 minutes \citep{task1996heart}, and the Poincar\'e descriptors SD1 and SD2 are functions of SDNN and of the standard deviation of successive differences \citep{brennan2001poincare}, so neither group is used in this work. Recent work has shown that
```

acrescenta ao `dissertation.bib`:
```bibtex
@article{brennan2001poincare,
  author  = {Brennan, Michael and Palaniswami, Marimuthu and Kamen, Peter},
  title   = {Do existing measures of {Poincar{\'e}} plot geometry reflect nonlinear features of heart rate variability?},
  journal = {IEEE Transactions on Biomedical Engineering},
  volume  = {48},
  number  = {11},
  pages   = {1342--1347},
  year    = {2001},
  doi     = {10.1109/10.959330}
}
```

## T20. SotA — o resumo repete a Introdução (rascunho, reescrever na tua voz)
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Stress is now part of the modern lifestyle and it has become one of the most prevalent health concerns in society.`
**Seleciona até:** `stress detection can be performed by analysing \gls{HRV} data derived from ECGs signals collected from these Wearables and fitness bracelets.`

**Substitui por:**
```latex
Stress activates the autonomic nervous system, and \gls{HRV} is the most studied marker of that activation. The literature reviewed in this chapter leaves four points that shape the rest of this work. First, the measures of \gls{HRV} were standardised for five-minute recordings; at thirty seconds the frequency-domain measures cannot be estimated, and the time-domain measures have been validated at different minimum durations, from 10 seconds for heart rate to 60 seconds for SDNN, RMSSD and pNN50. Second, the highest accuracies reported on SWELL-KW and WESAD come from pre-computed features, longer windows and random partitions that place windows of the same participant in the training and test sets; the studies that validate by subject report lower values, mostly on windows of 60 seconds or longer. Third, models trained on one stress dataset lose much of their performance on another, and where datasets are combined the agreement of their labels is assumed rather than tested. Fourth, normalising the features of each person by that person's own data, or choosing the threshold for each person, improves detection on new subjects. The closest precedents operate on the raw \gls{ECG} of 10 to 60 seconds \citep{article6, article9} or on \gls{HRV} over 60 seconds or more \citep{article8, prajod2024stressor, lee2025deep}; the evaluation of \gls{HRV} from 30-second windows alone, on subjects the model has not seen, is the configuration of a consumer wearable and the one this work addresses.
```

## T21. Trabalho futuro — janelas mais longas e o BVP
O texto antigo (comentado) prometia "70–80%" sem base. Fica a proposta sem o número.

**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `%\textbf{Longer analysis windows.} Moving from 30-second to 60-second or 2-minute windows`
**Seleciona até:** `would quantify the practical cost of real-time responsiveness.`

**Substitui por:**
```latex
\textbf{Window length and signal.} Only 30-second windows of chest \gls{ECG} were evaluated. Repeating the evaluation with windows of 10, 60 and 120 seconds in the same pipeline would measure what the ultra-short window costs, and whether the variability features add to the mean heart rate at longer windows; the pulse signal of the wrist device of WESAD would test the same models on the signal of a consumer wrist device.
```

## T22. (incluído no W11)

---

# S. Outros pontos da avaliação

## S1. 4.6.4 — a referência dos autores do WESAD (só ECG)
Verificado na Tabela 4 de Schmidt et al. 2018. A tarefa binária inclui a diversão no
não-stress; janelas de 60 s; LOSO. Só ECG do peito: exatidão DT 80,17 · RF 82,78 ·
AB 83,37 · LDA 85,44 · kNN 79,19.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The results on WESAD are not directly comparable with those of its authors either, because the binary task used here excludes the amusement condition (Section \ref{data}).`

**Substitui por:**
```latex
The results on WESAD are not directly comparable with those of its authors either. With 60-second windows, the amusement condition in the no-stress class and leave-one-subject-out validation, \citet{WESAD} report accuracies between 79.19\% and 85.44\% from the chest \gls{ECG} alone, depending on the classifier; the corresponding figures here, with 30-second windows and without the amusement condition (Section \ref{data}), are 74.75\% to 83.74\%, and 82.60\% for logistic regression.
```

## S2. (incluído no G7)

## S3. Contribuição 6 — o agente não foi comparado com uma lista fixa
**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `in a design that keeps physiological data on the device and falls back to local content when the service is unavailable.`

**Substitui por:**
```latex
in a design that keeps physiological data on the device and falls back to local content when the service is unavailable. Whether generated suggestions serve the user better than a fixed list was not evaluated.
```

---

# O. Opcional

## O1. SotA — PyTorch: o facto desatualizado **[verificar]**
A subsecção fica (decisão D2). Só o facto está desatualizado: desde setembro de 2022 o
PyTorch é gerido pela PyTorch Foundation, na Linux Foundation.

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `PyTorch is an open source Python package. Facebook, the Idiap Research Institute, New York University (NYU), and NEC Labs America hold the copyrights for PyTorch.`

**Substitui por:**
```latex
PyTorch is an open source Python package, developed at Facebook and governed since 2022 by the PyTorch Foundation, part of the Linux Foundation.
```

## Fica por fazer, fora deste documento
- **Epidemiologia centrada nos EUA** (SotA l.27–34): juntar dados da Europa e de
  Portugal. É preciso pesquisar as fontes; não pus números sem as ver.
- **Figura e anatomia do neurónio** (fonte Healthline): a decisão sobre cortes no Cap. 2 é tua.

---

# Estado no zip de 29/09 (16:57)

Aplicado tudo, com o texto igual ao deste documento: 56 itens verificados palavra a
palavra, mais T8 e T9 com `\item`. A secção H do `correcoes_finais.md` também está
aplicada. Falta só a T19, que entrou a meio: sem o SD1/SD2 e com uma gralha ("so is not
used").

## T19b. SotA — gralha e as medidas não lineares
Versão sem o Brennan. Se confirmares o Brennan 2001, usa antes a T19 original.

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Geometric measures require recordings of at least 20 minutes \citep{task1996heart}, so is not used in this work.`

**Substitui por:**
```latex
Geometric measures require recordings of at least 20 minutes \citep{task1996heart}, so they are not used in this work, and neither are the nonlinear measures.
```
