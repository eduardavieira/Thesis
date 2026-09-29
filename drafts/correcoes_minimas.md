# Correções mínimas — contra o zip de 29/09 (16:57)

São 11 correções. Cada uma tira uma afirmação que a própria tese ou os dados
contradizem. Não há números novos para calcular, nem experiências. Cada termo de
pesquisa aparece uma só vez no zip de 16:57; a ordem de aplicação não importa.

Ficam para a reescrita: notas a vermelho, copyright, bibliografia, gralhas e o ×100
duplicado da equação do F1. O título mantém-se.

| Ficheiro | Itens |
|---|---|
| `preamble/Abstract.tex` | 1 |
| `chapters/Introduction.tex` | 2, 3 |
| `chapters/StateOfTheArt.tex` | 4 |
| `chapters/Contribution.tex` | 5 |
| `chapters/Applications.tex` | 6, 7, 8, 9, 10, 11 |
| `chapters/ConclusionsAndFutureWork.tex` | 6, 10 |

---

## 1. Resumo — primeiro os sujeitos novos, depois a partição aleatória
É a primeira pergunta que o júri faz: porque é que os números de destaque vêm do
protocolo com fuga? A mudança é só de ordem, com os mesmos números.

**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `reaches 89.78\% accuracy and an AUC of 0.955 under a random split, and 80.14\% and 0.919 on unseen subjects under GroupKFold,`

**Substitui por:**
```latex
reaches an AUC of 0.919 and 80.14\% accuracy on unseen subjects under GroupKFold, and 0.955 and 89.78\% under a random split, in which windows of the same subject appear in training and test,
```

**Procura:** `atinge 89,78\% de exatidão e uma AUC de 0,955 com partição aleatória, e 80,14\% e 0,919 em sujeitos não vistos com GroupKFold,`

**Substitui por:**
```latex
atinge uma AUC de 0,919 e 80,14\% de exatidão em sujeitos não vistos com GroupKFold, e 0,955 e 89,78\% com partição aleatória, em que janelas do mesmo sujeito aparecem no treino e no teste,
```

## 2. Introdução — "DL is scarce… high and promising results"
A tese mostra depois que esses resultados altos vêm de partições com fuga.

**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `The use of deep learning techniques to detect stress is scarce in the literature. However the few studies that have used deep learning have achieved high and promising results, which indicates the importance of`

**Substitui por:**
```latex
Deep learning studies of stress detection report high accuracies, but mostly under conditions examined critically in Chapter \ref{cha:estado_de_arte}, which motivates evaluating
```

## 3. Introdução — a interpretabilidade da HRV
A explicabilidade conclui o contrário para o Model G.

**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `and the physiological interpretability of \gls{HRV} features makes such an account feasible in a way that is not available for less structured signals.`

**Substitui por:**
```latex
and the physiological meaning of \gls{HRV} features makes such an account possible in principle, although Chapter \ref{cha:experiencias} shows that it is limited when a model receives redundant inputs.
```

## 4. Cap. 2 — AI Act: "necessary instruments" vs "does not prescribe"
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `are now not merely academic tools but necessary instruments for demonstrating legal compliance`

**Substitui por:**
```latex
are now also among the tools with which developers can demonstrate legal compliance
```

## 5. 3.6 — a validação também é aleatória nos protocolos por sujeito
O código baralha os índices de treino antes de separar a validação, em todos os
protocolos (`train_sequence.py` l.170–175, `retrain_v5.py` l.133–136). O texto dava a
entender que não.

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `which takes the last 20\% of the training indices, in random order under the random split;`

**Substitui por:**
```latex
drawn at random from the training windows in every protocol, so that under the subject-wise protocols it is not a single subject or a single class;
```

## 6. Nadeau–Bengio — a vantagem do F em PR-AUC depende da razão usada
A razão 0,25 conta as janelas de validação como treino. Descontando-as (20/64 = 0,31),
a PR-AUC do F só sobrevive contra o Model A. As conclusões sobre o G não mudam.
Verificado com `corrected_ttest.py --ratio 0.3125`, que só lê as CSVs.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `For Model F, the PR-AUC differences over all seven models, the accuracy differences over logistic regression and the random forest, and the \gls{AUC} difference over Model A remain significant ($p_{\mathrm{adj}} \leq 0.047$).`

**Substitui por:**
```latex
For Model F, the PR-AUC differences over all seven models, the accuracy differences over logistic regression and the random forest, and the \gls{AUC} difference over Model A remain significant ($p_{\mathrm{adj}} \leq 0.047$). The correction uses a ratio of 0.25, which counts the validation windows as training windows; counting only the 64\% of the data used to fit the weights, a ratio of 0.31, the PR-AUC advantage of Model F survives only against Model A, and the results for Model G do not change.
```

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `and the advantage of the series is robust only for Model F.`

**Substitui por:**
```latex
and the advantage of the series is more robust for Model F, although part of it depends on how the correction is applied.
```

**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `the advantage is robust mainly for Model F, in PR-AUC, and that of Model G survives only against Model A.`

**Substitui por:**
```latex
few of these differences survive, and that of Model G survives only against Model A.
```

## 7. 4.4 — o falso positivo do S17 pode ser um artefacto
O texto atribuía o erro ao modelo sem ver o ECG. Um patamar de ~12 intervalos longos
arrasta a mediana local consigo, por isso o critério de correção não o apanha.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The error is the model's: a window whose heart rate is at the training mean lies on the stress side of its boundary, as the baseline itself does (Section \ref{sec:xai_disagreement}), and the attributions describe how far the other properties of the window move it back, not why it starts there.`

**Substitui por:**
```latex
The run of long intervals may also be an artefact that the correction of Section \ref{etl} does not detect, because a sustained plateau shifts the local median with it; the \gls{ECG} of this window was not inspected. The attributions describe how the output departs from the baseline, not whether the input is physiological.
```

## 8. 4.6.1 — a razão para o G não pode contradizer a explicabilidade
"Lê também as estatísticas da literatura" choca com a 4.4.2: essas atribuições não
se leem.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `Model G was retained because it reads the series and, through its summary branch, also the statistics that the literature reports, and because it has the highest mean \gls{AUC} under both subject-wise protocols; neither reason is a significant difference.`

**Substitui por:**
```latex
Model G was retained because it reads the series and has the highest mean \gls{AUC} under both subject-wise protocols; neither is a significant difference, and its summary branch does not add interpretable information (Section \ref{sec:summary_attribution}).
```

## 9. 4.6.1 — tirar a comparação com o D numa só partição
É o tipo de evidência que a tese rejeita noutros sítios.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `Three further properties favour Model G over the architectures it replaces.`

**Substitui por:**
```latex
Two further properties favour Model G over the architectures it replaces.
```

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `On the partition used for the explainability analysis it produces 5 false alarms in 117 no-stress windows, against 12 for Model D, and misses 11 of 65 stress windows, against 14.`

**Substitui por:** *(apaga a frase)*

## 10. 4.6.3 e Conclusões — a centragem no repouso não é significativa
Dar o t e o Wilcoxon quando só o Wilcoxon passa parece escolha do teste. O resto da
tese usa o t (p = 0,056).

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `raises the accuracy from 81.40\% to 88.22\% ($p = 0.056$; Wilcoxon signed-rank $p = 0.047$) without any stress window of the test subject, and the result`

**Substitui por:**
```latex
raises the accuracy from 81.40\% to 88.22\% without any stress window of the test subject, a gain that is not significant on 15 subjects ($p = 0.056$), and the result
```

**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `without any stress window, raises its accuracy from 81.4\% to 88.2\%.`

**Substitui por:**
```latex
without any stress window, reaches 88.2\% against 81.4\%, a gain that is not significant on 15 subjects.
```

## 11. 4.6.3 — porque é que os números da calibração diferem dos da 4.2
O pior sujeito aparece com 36,1% (todos os modelos, LOSO), 39,7% (G, LOSO) e 48,4%
(calibração). O S10 aparece com especificidade ≤ 7,7% na 4.2 e com 21,1% aqui. Estão
todos certos, mas vêm de conjuntos diferentes.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `the second half is the test set of all conditions.`

**Substitui por:**
```latex
the second half is the test set of all conditions. Because only the second half is tested, the values at the threshold of 0.5 differ from the full leave-one-subject-out results of Section \ref{sec:wesad_results}.
```
