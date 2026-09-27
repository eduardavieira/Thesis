# Correções finais — contra o zip de 27/09 (13:47)

Documento único com **tudo o que ainda falta alterar**. Substitui as Partes 1 e 2
para efeitos de aplicação: o que delas já está no zip não se repete aqui.
Cada **Procura** e cada **Seleciona até** foi validado contra o zip de 27/09 e
aparece uma só vez; os itens são independentes, a ordem não importa.

**O que o zip já tem:** toda a Parte 2 (números da V6), as secções 4.1–4.4 e 4.6
novas, e cerca de dois terços da Parte 1. Verificado com
`drafts/estado_aplicacao.py`.

Os itens **[verificar]** dependem de uma fonte que convém confirmar antes de
aplicar.

## Índice por ficheiro

| Ficheiro | Itens |
|---|---|
| `preamble/Abstract.tex` | B12, B13, C12, G5 |
| `chapters/Introduction.tex` | B13, B14, C15 |
| `chapters/StateOfTheArt.tex` | B1, B2, B3, B4, B5, B6, C13, C14, C16, D1, D2, D3, D4, D6 |
| `chapters/Contribution.tex` | A7, B7, B8, C1, C2, C3, C4, C16 |
| `chapters/Applications.tex` | A1, A2, A3, A4, A5, A6, B9, B10, B11, C5, C6, C7, C8, C9, C10, D5, G1, G2 |
| `chapters/ConclusionsAndFutureWork.tex` | B13, B15, C11, G3, G4 |
| `dissertation.bib` | B16, B7 (entrada nova), B14 (entrada nova), C1 (entrada nova) |
| todos os `.tex` | C17 (ortografia, *Replace all*) |

Figuras (secção E): substituir as duas imagens no Overleaf e editar a Fig. 3.1.

---

# A. Decisões tuas que o zip ainda não reflete

O zip foi exportado antes de decidires que a comparação SWELL/WESAD fica fora
do texto. Estes sete itens repõem a redação original nesses pontos.

## A1. 4.1 — "the evidence points to"
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The evidence points to the design of SWELL-KW, as documented by its own authors, who state that`

**Substitui por:**
```latex
The cause is documented in the design of SWELL-KW by its own authors, who state that
```

## A2. 4.1 — frase sobre não se poder separar ordem e estado
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The evidence is therefore consistent with a classifier trained on these labels learning the order`
**Seleciona até:** `since no participant recorded the neutral condition after a stressor.`

**Substitui por:**
```latex
A classifier trained on these labels therefore learns the order in which the blocks were recorded rather than the participant's state.
```

## A3. 4.1.1 — "on WESAD"
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `Under the same protocol, on WESAD, the same features`

**Substitui por:**
```latex
Under the same protocol, on the dataset whose labels correspond to a validated physiological state, the same features
```

## A4. 4.2 — "established laboratory stressor"
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `when the label corresponds to an established laboratory stressor:`

**Substitui por:**
```latex
when the label corresponds to a validated physiological state:
```

## A5. 4.6 — abertura
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `shows that the evidence points to the labels rather than the models.`

**Substitui por:**
```latex
shows that the cause is the labels rather than the models.
```

## A6. Limitações — apagar o parágrafo dos confundidores do WESAD
O parágrafo seguinte ("Model G was chosen after…") fica.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The WESAD label is also confounded with the protocol.`
**Seleciona até:** `not established by a counterbalanced comparison.`

**Substitui por:** *(apaga o parágrafo, incluindo a linha em branco a seguir)*

## A7. 3.3 — frase da baseline sempre primeiro
**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `performed in front of a panel. The baseline was recorded first in every participant, and the two versions of the protocol differ only in the order of the amusement and stress conditions \citep{WESAD}.`

**Substitui por:**
```latex
performed in front of a panel \citep{WESAD}.
```

---

# B. Parte 1 — itens que ainda faltam

## B1. "Artificial learning… billions of neurons" descreve o cérebro — Cap. 2
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Artificial learning is done by a highly interconnected neural network, composed by billions of distinct types of neurons,`

**Substitui por:**
```latex
Biological learning is carried out by a highly interconnected neural network, composed of billions of neurons of distinct types,
```

## B2. Softmax nunca descrita; "liner" — Cap. 2
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `ReLU stands for rectified liner unit`

**Substitui por:**
```latex
The softmax function generalises the sigmoid to $K$ classes: it maps a vector of $K$ scores to $K$ probabilities that sum to one, and is used in the output layer of multiclass classifiers.

ReLU stands for rectified linear unit
```

## B3. A cross-entropy não é limitada a 1 — Cap. 2
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `If this difference is large, then the function will have result close to one and if the difference is almost none, then the result will be close to zero`

**Substitui por:**
```latex
It is zero when the predicted distribution matches the true one and grows without bound as the probability assigned to the true class approaches zero
```

## B4. Multiclasse descrita como multilabel — Cap. 2
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Multiclass classification allows the output to be labeled simultaneously by several of the K possible labels.`
**Seleciona até:** `will be the output label for the respective input`

**Substitui por:**
```latex
In multiclass classification each input belongs to one of $K$ possible classes; each class is scored, and the class with the highest score is the output for that input
```

## B5. Gradientes em ordem aos parâmetros vs à entrada — Cap. 2
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Gradient-based feature importance methods begin with the output of the network`
**Seleciona até:** `with respect to each intermediate parameter`

**Substitui por:**
```latex
Gradient-based attribution methods differentiate the output of the trained network with respect to its input rather than its parameters: the gradient measures how sensitive the prediction is to a small change in each input value, while the weights of the network are held fixed
```

## B6. Cho et al.: "(0–1) using z-score" é contraditório — Cap. 2
A contradição vem do próprio artigo (Cho et al., 2019, Secção 2.2): os autores
escrevem "scaled them to the same range (0–1) using z-score normalization", mas a
fórmula que dão é o z-score, (x − µ)/σ, com a média e o desvio-padrão de cada
dataset. A tese passa a descrever o que a fórmula faz.

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `they scaled the \gls{ECG} signals to the same range (0–1) using z-score normalisation.`

**Substitui por:**
```latex
they normalised the \gls{ECG} signals of each data set with the z-score, using the mean and standard deviation of that data set.
```

## B7. LF/HF como "sympathovagal balance" — Tabela do Cap. 3
**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `Index of sympathovagal balance. Higher LF/HF indicates higher stress.`

**Substitui por:**
```latex
Traditionally read as sympathovagal balance, an interpretation now disputed \citep{billman2013lfhf}.
```
E acrescenta ao `dissertation.bib`:
```bibtex
@article{billman2013lfhf,
  author  = {Billman, George E.},
  title   = {The {LF/HF} ratio does not accurately measure cardiac sympatho-vagal balance},
  journal = {Frontiers in Physiology},
  volume  = {4},
  pages   = {26},
  year    = {2013},
  doi     = {10.3389/fphys.2013.00026}
}
```

## B8. "No physiological data leaves the device" — 3.8
**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `This has a privacy benefit, in that no physiological data leaves the device, but it also means`

**Substitui por:**
```latex
This limits what leaves the device, since no \gls{HRV} values are transmitted. The predicted label is nonetheless information about the user's health inferred from physiological signals, and would likely be treated as data concerning health under the General Data Protection Regulation, so a deployed version would still need an explicit legal basis for sending it to a third-party service. The design also means
```

## B9. "Validates" com dois exemplos — 4.5
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `This demonstration validates that the complete pipeline`

**Substitui por:**
```latex
This demonstration shows, on two windows, that the complete pipeline
```

## B10. "Naturally varied… repetition fatigue" — 4.5
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The integration of a large language model as the feedback layer offers responses that are naturally varied across sessions, avoiding the repetition fatigue associated with fixed notification messages. They align with`

**Substitui por:**
```latex
Because the messages are generated rather than fixed, they can vary between sessions; whether this reduces the habituation associated with repeated notifications was not evaluated. The suggestions align with
```

## B11. O que a demonstração do agente não avalia — 4.5
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `the current design informs the user of a state without offering any account of why it was inferred.`

**Substitui por:**
```latex
the current design informs the user of a state without offering any account of why it was inferred.

The demonstration does not evaluate the agent. Two windows establish that the pipeline runs end to end; they do not measure the latency of the language model, which depends on the network and was not recorded, nor the safety and appropriateness of the generated advice, which would require a systematic review of responses before any deployment.
```

## B12. "Continuous, real-time" — Abstract e Resumo
**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `Consumer wearable devices already allow continuous, real-time stress monitoring, which makes early intervention possible in everyday life.`

**Substitui por:**
```latex
Consumer wearable devices already record cardiac signals in everyday life, which makes stress monitoring and early intervention possible outside the laboratory.
```

**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `Os dispositivos \textit{wearable} já permitem monitorizar o stress de forma contínua e em tempo real, o que torna possível a intervenção precoce no dia a dia.`

**Substitui por:**
```latex
Os dispositivos \textit{wearable} já registam sinais cardíacos no dia a dia, o que torna possível monitorizar o stress e intervir precocemente fora do laboratório.
```

## B13. "Identifies" → "points to", com o resultado da calibração — Abstract, Resumo, Introdução, Conclusões
Com o teste da secção G, as frases passam a citar o resultado em vez de só o
anunciar.

**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `which identifies the per-person decision threshold, rather than the signal, as the limiting factor`

**Substitui por:**
```latex
which points to the per-person decision threshold, rather than the signal, as the limiting factor: choosing the threshold of the selected model on about 15 minutes of each subject's recording raises its accuracy on the rest of the session from 82.1\% to 91.1\%
```

**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `o que identifica o limiar de decisão de cada pessoa, e não o sinal, como o fator limitante`

**Substitui por:**
```latex
o que aponta o limiar de decisão de cada pessoa, e não o sinal, como o fator limitante: escolher o limiar do modelo selecionado em cerca de 15 minutos de gravação de cada sujeito sobe a sua exatidão no resto da sessão de 82,1\% para 91,1\%
```

**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `which identifies the placement of the decision threshold, and not the discriminability of the signal, as the limiting factor`

**Substitui por:**
```latex
which points to the placement of the decision threshold, and not the discriminability of the signal, as the limiting factor; choosing the threshold on about 15 minutes of each subject's recording raises the accuracy of Model G from 82.1\% to 91.1\%, and that of the worst subject from 48.4\% to 73.3\%
```

**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `What limits the system is the placement of the decision threshold across individuals:`

**Substitui por:**
```latex
The results point to the placement of the decision threshold across individuals as what limits the system:
```

## B14. AI Act — datas (Introdução)
As datas mudaram: o Regulamento (UE) 2026/1744 ("Digital Omnibus on AI"), em vigor
desde 27/07/2026, adiou as obrigações de risco elevado para 2/12/2027 (Anexo III) e
2/8/2028 (Anexo I, que inclui os dispositivos médicos). Verificado no texto da EUR-Lex.

**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `with full compliance obligations due by August 2026 \citep{euaiact2024}.`

**Substitui por:**
```latex
with the obligations for the high-risk systems listed in its Annex III applicable from December 2027 and those for regulated products, including medical devices, from August 2028, after their postponement by Regulation (EU) 2026/1744 \citep{euaiact2024, aiomnibus2026}.
```
E acrescenta ao `dissertation.bib`:
```bibtex
@misc{aiomnibus2026,
  author       = {{European Parliament and Council of the European Union}},
  title        = {Regulation ({EU}) 2026/1744 of 8 July 2026 amending Regulation ({EU}) 2024/1689 (Digital Omnibus on {AI})},
  howpublished = {Official Journal of the European Union, L series, 24 July 2026},
  year         = {2026},
  url          = {https://eur-lex.europa.eu/eli/reg/2026/1744/oj}
}
```
Confirma o título exato na página da EUR-Lex indicada no `url`.

## B15. Conclusões — dados fisiológicos
**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `Only the predicted label crosses the API boundary, so no physiological data leaves the device, and the agent`

**Substitui por:**
```latex
Only the predicted label crosses the API boundary, so no \gls{HRV} values leave the device, although the label is itself inferred health information; and the agent
```

## B16. Bibliografia — `jadav2024lstm`
**Ficheiro:** `dissertation.bib`
**Procura:** `title = {International Journal of INTELLIGENT SYSTEMS AND APPLICATIONS IN ENGINEERING Heart Rate Variability Based LSTM Model for Stress Detection with Explainable AI Insights}`

**Substitui por:**
```bibtex
title = {Heart Rate Variability Based {LSTM} Model for Stress Detection with Explainable {AI} Insights},
journal = {International Journal of Intelligent Systems and Applications in Engineering}
```

---

# C. Segunda avaliação — incoerências e erros

## C1. Participantes excluídos do SWELL-KW — 3.3
O artigo do SWELL-KW (Koldijk et al., 2014) não regista falhas por participante:
diz que o conjunto tem os 25 e que, nos seus dados agregados por minuto, os autores
excluíram minutos em que a deteção de picos falhou. As falhas dos participantes 8,
11 e 23 estão na folha de disponibilidade que acompanha os dados
(`SWELL-KW - overview available data.xlsx`, coluna do ECG). Por isso a frase cita o
dataset publicado no DANS e não o artigo. **Confirma que essa folha veio com os
dados do DANS**; se veio de outro sítio, a citação muda.

O critério "sem nenhum bloco com frequência cardíaca válida, ou sem bloco neutro"
separa exatamente os três excluídos dos 22 incluídos: os incluídos com blocos
assinalados (3, 15, 20, 22 e 25) têm sempre pelo menos um bloco válido.

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `Participants 11 and 23 have a documented failure of the cardiac recording, in the second and the first block respectively, and participant 8 was excluded without a recorded reason.`

**Substitui por:**
```latex
The overview of available data distributed with the dataset \citep{swell_dans} records a problem with the cardiac recording of the three participants excluded: for participant 23 the recording of the first block, the neutral condition, is too short; participant 11 has no cardiac recording of the second block; and for participants 8 and 11 no valid heart rate could be extracted from any of the remaining blocks, the third block of participant 8 being marked as containing very strange data.
```
E acrescenta ao `dissertation.bib`:
```bibtex
@misc{swell_dans,
  author       = {Koldijk, Saskia and Sappelli, Maya and Verberne, Suzan and Neerincx, Mark A. and Kraaij, Wessel},
  title        = {The {SWELL} Knowledge Work Dataset for Stress and User Modeling Research},
  howpublished = {DANS Data Station Social Sciences and Humanities},
  year         = {2014},
  doi          = {10.17026/dans-x55-69zp}
}
```

## C2. Descrições dos Modelos A–D falam de dependências temporais — 3.5
Uma frase antes das descrições resolve a contradição com o parágrafo que as
segue, sem reescrever cada modelo; "raw HRV signals" é falso para todos.

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `The Model A (Simple LSTM) architecture consists of`

**Substitui por:**
```latex
The architectures of Models A to D were designed for sequential data. As discussed after their description, the input they receive in this work is not a time series, so the references to temporal modelling below describe what each architecture was designed to do, not what it can extract from eight summary features.

The Model A (Simple LSTM) architecture consists of
```

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `allowing the model to first extract meaningful features from the raw \gls{HRV} signals and then model their temporal evolution for stress detection.`

**Substitui por:**
```latex
allowing the model to first extract local features from its input and then model their evolution along the sequence.
```

## C3. "This rigorous… ensured… reliable performance estimation" — 3.6
As limitações dizem o contrário (seleção nas mesmas partições, variação entre execuções).

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `This rigorous train-validation-test split methodology, combined with careful normalization procedures and regularization techniques, ensured robust model development and reliable performance estimation.`

**Substitui por:** *(apaga a frase)*

## C4. Contagens — Cap. 3
**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `seven deep learning models were built and compared with three reference classifiers`

**Substitui por:**
```latex
seven neural models were built and compared with three reference classifiers
```

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `The two reference classifiers used the standardisation step alone,`

**Substitui por:**
```latex
Logistic regression and the random forest used the standardisation step alone,
```

## C5. "Three independent lines of evidence" — 4.1
O efeito por dataset e a transferência vêm dos mesmos dados.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `Three independent lines of evidence are presented:`

**Substitui por:**
```latex
Three lines of evidence are presented:
```

## C6. Habituação: p = 0,78 não prova ausência — 4.1
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `($p = 0.78$), which excludes habituation within the block as an explanation and indicates that the elevation is sustained throughout.`

**Substitui por:**
```latex
($p = 0.78$), which gives no indication of habituation within the block and suggests that the elevation is sustained throughout; because windows within a block are not independent, these p-values are indicative.
```

## C7. Gama de ganhos de AUC (a da RF, 0,015, fica fora) — 4.2
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The \gls{AUC} gains, between 0.020 and 0.037, remain significant against every model except the random forest ($p_{\mathrm{adj}} \leq 0.048$),`

**Substitui por:**
```latex
The \gls{AUC} gains over the six models other than the random forest, between 0.020 and 0.037, remain significant after correction ($p_{\mathrm{adj}} \leq 0.048$), while the gain of 0.015 over the random forest does not,
```

## C8. S17: "which the model never sees" — 4.4
Com a partição aleatória o modelo viu outras janelas desse sujeito; o que não é
é uma entrada do modelo.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `What separates them is each subject's own baseline, which the model never sees.`

**Substitui por:**
```latex
What separates them is each subject's own baseline, which is not an input of the model.
```

## C9. S10: "correctly ordered, and all of them lie above 0.5" — 4.6
Com AUC 0,895 não estão todas bem ordenadas, e com especificidade de 5,1% (G)
nem todas estão acima de 0,5.

**Ficheiro:** `chapters/Applications.tex`
**Procura:** `That person's windows are correctly ordered, and all of them lie above 0.5.`

**Substitui por:**
```latex
That person's windows are largely well ordered, and almost all of them lie above 0.5.
```

## C10. Latência medida em Keras num portátil — 4.6
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `Its inference time, 49.21 ms per window, is well below the 30 seconds needed to acquire the next window.`

**Substitui por:**
```latex
Its inference time, 49.21 ms per window, is well below the 30 seconds needed to acquire the next window; it was measured with Keras on a laptop processor, where framework overhead dominates, and does not establish the latency on a wearable device.
```

## C11. Conclusões — frase-resumo sem condições
**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `that reading the interval series is better than summarising it, and that the main obstacle to a deployable system is calibration to the individual rather than the signal.`

**Substitui por:**
```latex
that, under a random split, reading the interval series is better than summarising it, and that the main obstacle to a deployable system appears to be calibration to the individual rather than the signal.
```

## C12. Resultado por sujeito ao lado do aleatório — Abstract e Resumo
O número de destaque era só o da partição que a própria tese chama otimista.

**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `reaches 89.78\% accuracy and an AUC of 0.955 under a random split, with 32,929 parameters`

**Substitui por:**
```latex
reaches 89.78\% accuracy and an AUC of 0.955 under a random split, and 80.14\% and 0.919 on unseen subjects under GroupKFold, with 32,929 parameters
```

**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `atinge 89,78\% de exatidão e uma AUC de 0,955 com partição aleatória, com 32\,929 parâmetros`

**Substitui por:**
```latex
atinge 89,78\% de exatidão e uma AUC de 0,955 com partição aleatória, e 80,14\% e 0,919 em sujeitos não vistos com GroupKFold, com 32\,929 parâmetros
```

## C13. AI Act — datas e enquadramento do sistema — Cap. 2
Junta a correção das datas com o parágrafo que enquadra o sistema no AI Act.
Artigos verificados no texto oficial do Regulamento 2024/1689 (art. 3.º, ponto 39;
considerando 18; art. 5.º, n.º 1, alínea f); art. 50.º, n.º 3; Anexo III, ponto 1,
alínea c); art. 113.º) e datas no Regulamento 2026/1744. A definição exige emoções
inferidas de dados biométricos (que incluem características fisiológicas), e o
considerando 18 lista emoções e exclui estados físicos como a dor ou a fadiga; o
stress não é mencionado, por isso o texto diz que o enquadramento é incerto em vez
de o afirmar.

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Full compliance obligations for high-risk \gls{AI} systems are due by August 2026, which means healthcare organizations and \gls{AI} developers currently face the practical necessity to implement explainability in their \gls{AI} development pipelines \citep{euaiact2024}.`

**Substitui por:**
```latex
The prohibitions of the Act apply since February 2025 and its transparency obligations since August 2026. The obligations for the high-risk systems listed in Annex III were postponed to December 2027, and those for regulated products, including medical devices, to August 2028 \citep{euaiact2024, aiomnibus2026}, so developers of such systems must plan for explainability now.

Whether a stress detector falls under these rules is not settled by the text. The Act defines an emotion recognition system as one that infers emotions or intentions from biometric data, which include physiological characteristics; its recitals list emotions such as happiness, anger or embarrassment and exclude physical states such as pain or fatigue, and do not mention stress. If a system of the kind developed here were considered an emotion recognition system, it would be high-risk under Annex III, its deployers would have to inform the people exposed to it (Article 50(3)), and its use to infer the emotions of a person in the workplace or in education would be prohibited except for medical or safety reasons (Article 5(1)(f)). It would then be lawful as a personal wellness tool chosen by its user, but not as a tool with which an employer monitors knowledge workers.
```

## C14. Apple Watch: os RR vêm do sensor ótico (PPG), não de ECG — Cap. 2
Confirmado no artigo (Hernando et al., 2018): os intervalos do Apple Watch foram
extraídos com a app Breathe, a partir do sensor ótico.

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `\subsection{Reliability of \gls{ECG} data collected from wearables}`

**Substitui por:**
```latex
\subsection{Reliability of \gls{HRV} data collected from wearables}
```

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `validate the Apple Watch in terms of \gls{HRV} derived from the RR interval series provided by the device with a total of 20 subjects`

**Substitui por:**
```latex
validate the Apple Watch in terms of \gls{HRV} derived from the RR interval series that the device estimates with its optical (photoplethysmographic) sensor, with a total of 20 subjects
```

## C15. "Has not been systematically evaluated" — Introdução
Sem uma revisão sistemática, a afirmação tem de se limitar ao que foi revisto.

**Ficheiro:** `chapters/Introduction.tex`
**Procura:** `--- has not been systematically evaluated.`

**Substitui por:**
```latex
--- has not been evaluated in the works reviewed in Chapter \ref{cha:estado_de_arte}.
```

## C16. Gralhas
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `a system to detecting and classify stress`

**Substitui por:**
```latex
a system to detect and classify stress
```

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Such information is present on Section \ref{sec:stress}.`

**Substitui por:**
```latex
Such information is presented in Section \ref{sec:stress}.
```

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `is the responsibility of loss functions The goal`

**Substitui por:**
```latex
is the responsibility of loss functions. The goal
```

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `A accuracy of 92.70\% was achieved.`

**Substitui por:**
```latex
An accuracy of 92.70\% was achieved.
```

**Ficheiro:** `chapters/Contribution.tex`
**Procura:** `using ReLU  ctivation.`

**Substitui por:**
```latex
using ReLU activation.
```

## C17. Ortografia: uniformizar para inglês britânico
O texto é maioritariamente britânico (analyse, behaviour, standardised,
summarise). Usa o *Replace all* do Overleaf **nos `.tex`** (não no `.bib`),
com *match case* e palavra inteira:

| US → UK | ocorrências |
|---|---|
| normalization → normalisation | 14 |
| normalized → normalised | 2 |
| regularization → regularisation | 6 |
| optimizer → optimiser | 1 |
| optimizers → optimisers | 1 |
| optimized → optimised | 2 |
| analyze → analyse | 2 |
| modeling → modelling | 6 |
| generalized → generalised | 1 |
| organizations → organisations | 1 |
| recognize → recognise | 1 |

Não mexer em nomes de camadas com maiúscula ("Layer Normalization") nem em
texto dentro de `\texttt{}`.

---

# D. Capítulo 2 — cortar e completar a literatura

## D1. Cortar Regression e Clustering (a Classification fica)
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `\subsubsection{Regression}`
**Seleciona até:** `calculated distance. \citep{ml3, ml5, ml6}.`

**Substitui por:** *(apaga as duas subsubsecções)*

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Deep learning is used for classification, clustering, and regression tasks in several fields, in particular healthcare \citep{ml7, ml6}`

**Substitui por:**
```latex
Deep learning is used for classification, clustering and regression tasks in several fields, in particular healthcare \citep{ml7, ml6}. This work addresses a classification task.
```

## D2. Cortar a subsecção "Deep Learning Frameworks" inteira
TensorFlow, TensorFlow Lite (hoje LiteRT), Keras e PyTorch: material de manual.
A label `cha:tensorflowlite` só é referida dentro da própria subsecção; as
ferramentas usadas continuam descritas na 3.9.

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `\subsection{Deep Learning Frameworks}`
**Seleciona até:** `was merged into PyTorch in 2018. \citep{pytorchbib, ibmfraneworks}.`

**Substitui por:** *(apaga a subsecção)*

## D3. Tabela 2.6 — tirar a review, acrescentar o Deep ECGNet
Hwang et al. (2018) é o antecedente mais próximo: CNN seguida de duas LSTM
sobre ECG em bruto de 10 s, 87,39% e 73,96% nos dois casos avaliados
(confirmado na publicação).

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `Haque et al. \citep{haque2023stateofart} & Review (multiple) & & Review & - & 2024 & \\ \hline`

**Substitui por:**
```latex
Hwang et al. \citep{article9} & Private & Raw \gls{ECG}, 10 s windows & CNN-LSTM & - & 2018 & 87.39\%, 73.96\% \\ \hline
```

## D4. Benchekroun et al. (2023) — Cap. 2
Está no `.bib` (`newArtc4`) e nunca é citado; é o trabalho mais próximo da
Contribuição 1. Resultados confirmados na publicação (Sensors 23(4):1807).

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `\subsection{Summary}`

**Substitui por:**
```latex
Evaluation across datasets remains rare in this literature. Benchekroun et al. \citep{newArtc4} trained logistic regression and random forest classifiers on \gls{HRV} features from two datasets that differ in protocol, stressors and recording devices. Both performed well under leave-one-subject-out validation within each dataset, and when trained on one dataset and tested on the other the random forest kept an F1 score of 61\%. Section \ref{sec:per_dataset} applies the same kind of test to SWELL-KW and WESAD.

\subsection{Summary}
```

## D5. Benchekroun et al. — ligação na discussão (4.6)
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `Pooling stress datasets to increase the sample size is common practice, and it rests on the assumption that the labels of different protocols describe the same state.`

**Substitui por:**
```latex
Pooling stress datasets to increase the sample size is common practice, and it rests on the assumption that the labels of different protocols describe the same state. Where transfer between datasets has been evaluated, a loss of performance has been read as a problem of generalisation \citep{newArtc4}; an \gls{AUC} below 0.5 indicates something different, that the labels themselves disagree.
```

## D6. Epidemiologia: ansiedade não é stress
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `cited among the leading concerns \citep{APA2024poll}.`

**Substitui por:**
```latex
cited among the leading concerns \citep{APA2024poll}. These figures concern anxiety, which overlaps with stress but is not the same construct; they are cited as an indication of the scale of the problem.
```

---

# E. Figuras

- **`feature_importance_global_G.png` e `gradient_by_class_mean_G.png`**: o
  título dizia "N=189"; as versões novas, em `drafts/figures_v6/`, dizem N=182.
  Os valores são os mesmos (mesmos pesos da V6). Substituir no Overleaf.
- **Fig. 3.1** (`proposed_approach_new.png`): diz 6 596 amostras (são 5 409) e
  não mostra o Model G. Editar a figura.

---

# G. Calibração do limiar por pessoa (nova)

Corrida V6 de 27/09: `Classification Models/personalised_protocol.py` com
`--models G,LogReg --scaler standard` → `results_v6_personalised.csv`, log em
`logs_v6/personalised.log`. Em LOSO no WESAD; por cada sujeito, a primeira metade
das janelas de cada condição (por ordem de gravação, 29–32 janelas ≈ 15 min) calibra
e a segunda metade testa.

## G1. 4.6.3 — o teste, em vez de "was not tested… hypothesis"
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `The thresholds that would be optimal for each subject therefore vary widely, and the per-subject calibration proposed in Section \ref{future_work} addresses this directly.`
**Seleciona até:** `should be read as the hypothesis that the results support rather than as a demonstrated cause.`

**Substitui por:**
```latex
The thresholds that would be optimal for each subject therefore vary widely. This was tested directly with a per-subject calibration under leave-one-subject-out. For each held-out subject, the first half of the windows of each condition, in recording order, served for calibration, 29 to 32 windows or about 15 minutes of labelled \gls{ECG}, and the second half for testing. Three conditions were compared on the same test windows: the model trained on the other 14 subjects with the threshold of 0.5; the same model, without retraining, with the threshold that maximises Youden's index on the subject's calibration windows; and a model retrained with the calibration windows added to its training set. Table \ref{tab:personalised} reports the result for Model G and logistic regression.

\begin{table}[h!]
\centering
\small
\begin{tabular}{|l|l|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Condition} & \textbf{Accuracy (\%)} & \textbf{Worst subject (\%)} & \textbf{Min. specificity (\%)} & \textbf{Subjects below 70\%} \\ \hline
\multirow{3}{*}{Model G} & Threshold 0.5 & 82.08 $\pm$ 14.93 & 48.4 & 21.1 & 3 \\ \cline{2-6}
 & Own threshold & \textbf{91.12 $\pm$ 7.41} & \textbf{73.3} & 68.4 & \textbf{0} \\ \cline{2-6}
 & Retrained & 90.22 $\pm$ 10.59 & 63.3 & 78.9 & 1 \\ \hline
\multirow{3}{*}{Logistic regression} & Threshold 0.5 & 81.40 $\pm$ 18.32 & 38.7 & 0.0 & 3 \\ \cline{2-6}
 & Own threshold & 85.92 $\pm$ 16.46 & 37.9 & 31.6 & 2 \\ \cline{2-6}
 & Retrained & 81.84 $\pm$ 18.19 & 38.7 & 0.0 & 3 \\ \hline
\end{tabular}
\caption{Per-subject calibration on WESAD under leave-one-subject-out. Each subject's windows are split in time: the first half calibrates, the second half is the test set of all three conditions. Mean $\pm$ standard deviation across the 15 subjects.}
\label{tab:personalised}
\end{table}

For Model G, the subject's own threshold raises the mean accuracy from 82.08\% to 91.12\% ($p = 0.014$, paired t-test; $p = 0.028$ after Holm correction for the two comparisons), halves the dispersion between subjects and raises the worst subject from 48.4\% to 73.3\%, so that no subject remains below 70\%. It improves 9 of the 15 subjects and worsens 2. Subject S10 goes from 48.4\% to 87.1\% and its specificity from 21.1\% to 100\%, without any change to the model. Retraining with the same windows gives a similar mean, 90.22\% ($p = 0.024$; 0.028 after correction), but a lower worst case, 63.3\%, and leaves one subject with no stress window detected; the threshold achieves as much without retraining. For logistic regression the gain is smaller and not significant ($+4.52$ pp, $p = 0.30$): the threshold corrects the gross failures, such as subject S10, from 38.7\% to 80.7\%, but for subject S2 the ranking of the test windows itself fails, with an \gls{AUC} of 0.32, and no threshold can recover it. The thresholds chosen for Model G range from 0.04 to 1.00, with a median of 0.65, which is a direct measurement of the problem described above.

Three limits apply to this result. The calibration and test windows come from the same two recording sessions, split in time, so the estimate is optimistic about the stability of a threshold over days. The calibration requires labelled windows of both classes, including a stress episode, which a real user would have to provide. And with 15 subjects the test has little power beyond the largest effects.
```

## G2. 4.2 — ligação à 4.6.3
**Ficheiro:** `chapters/Applications.tex`
**Procura:** `This dissociation between ranking and decision is the empirical basis for the per-subject calibration proposed in Section \ref{future_work}.`

**Substitui por:**
```latex
This dissociation between ranking and decision is the empirical basis for the per-subject calibration tested in Section \ref{sec:discussion_threshold}.
```

## G3. Conclusões — o resultado da calibração
**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `and still reach accuracies between 36.1\% and 100\% depending on the subject.`

**Substitui por:**
```latex
and still reach accuracies between 36.1\% and 100\% depending on the subject. A threshold chosen on the first half of each subject's recording raises the accuracy of Model G on the second half from 82.1\% to 91.1\%, without retraining.
```

## G4. Trabalho futuro — de "testar" para "validar entre sessões"
**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex`
**Procura:** `\textbf{Person-specific calibration.} The subject-wise results of Chapter \ref{cha:experiencias} show that the models rank each person's windows well`
**Seleciona até:** `should be measured on unseen recording sessions \citep{haque2023stateofart}.`

**Substitui por:**
```latex
\textbf{Person-specific calibration.} A threshold chosen on about 15 minutes of each user's labelled windows raised the accuracy of Model G from 82.1\% to 91.1\% within the same recording session (Section \ref{sec:discussion_threshold}). Whether that threshold remains valid across days, and how the labelled windows, including a stress episode, could be obtained from a user in practice, should be measured on unseen recording sessions \citep{haque2023stateofart}.
```

## G5. Abstract e Resumo — trabalho futuro
**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `Future work should address per-person calibration, validation on further datasets with physiologically grounded labels, and on-device deployment.`

**Substitui por:**
```latex
Future work should test per-person calibration across recording sessions, and address validation on further datasets with physiologically grounded labels and on-device deployment.
```

**Ficheiro:** `preamble/Abstract.tex`
**Procura:** `O trabalho futuro deverá abordar a calibração por pessoa, a validação noutros \textit{datasets} com rótulos fundamentados na fisiologia e a implementação no dispositivo.`

**Substitui por:**
```latex
O trabalho futuro deverá testar a calibração por pessoa entre sessões de gravação, e abordar a validação noutros \textit{datasets} com rótulos fundamentados na fisiologia e a implementação no dispositivo.
```

---

# H. Depois do zip de 27/09 (18:37)

Verificado contra o zip das 18:37: tudo aplicado, exceto o B7 e o D2, que
decidiste não aplicar, e a secção E, em que cortaste o texto da Fig. 3.1. Faltam
estas três correções.

## H1. Gralha "a a system"
**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `focuses on the development of a a system to detect`

**Substitui por:**
```latex
focuses on the development of a system to detect
```

## H2. Chave de citação estragada pela uniformização da ortografia
O *Replace all* optimizers → optimisers apanhou a chave `articleOptimizers`; no
PDF a citação aparece como "?".

**Ficheiro:** `chapters/StateOfTheArt.tex`
**Procura:** `decaying average of square gradients \citep{articleoptimisers}.`

**Substitui por:**
```latex
decaying average of square gradients \citep{articleOptimizers}.
```

## H3. Figura com "N=189" no título
A `gradient_by_class_mean_G.png` do zip ainda é a versão antiga do título (os
valores são os mesmos). Substitui-a no Overleaf pela de
`Classification Models/results/model_G_v6/gradient_by_class_mean_G.png`, que diz
N=182. A `feature_importance_global_G.png` já está certa.

---

# F. Para a defesa (não entra no texto)

Pontos da segunda avaliação que decidiste não pôr no texto ou que exigiriam
análises novas; convém ter resposta preparada:

- A baseline do WESAD também é sempre a primeira; fala e postura no TSST;
  a comparação com a assimetria que se aplica ao SWELL-KW.
- A calibração do limiar está testada (secção G), mas só dentro da mesma sessão;
  a normalização de cada sujeito pela sua própria baseline não foi testada.
- Porquê o G e não o F; a vantagem da série desaparece por sujeito — em que
  difere da crítica a Xia et al.
- Validação do pipeline do SWELL-KW (20% de janelas excluídas; os próprios
  autores assinalam "no valid HR extracted" em vários blocos).
- Porque excluir 3 participantes se os autores mantêm os 25 nos seus dados por
  minuto: os autores trabalham sobre minutos já filtrados; esta tese usa o ECG em
  bruto, e a folha de disponibilidade assinala o ECG desses três.
- Do próprio artigo do SWELL-KW, a favor da tese: "Due to our experimental design
  of always starting with the neutral baseline condition, subjects might have
  experienced order or fatigue effects", e "We did not find significant effects of
  our working conditions on perceived stress".
- Tarefa binária padrão do WESAD; BVP do pulso; janelas de 60 e 120 s.
- Modelos A–D treinados com outro protocolo; LSTM com ReLU.
- Explicabilidade: um só retreino, a máscara fora das atribuições, baseline do
  IG a 0,916, SHAP/GradientSHAP.
- Probabilidades não calibradas (Brier, curva de fiabilidade; limiar não
  otimizado na validação).
- Agente: avaliação com mais respostas, restrições do prompt, modo sem ligação.
