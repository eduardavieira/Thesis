# Literatura: Giannakakis et al. 2019 (article8) e Hong et al. 2020 (article11)

Validado contra o zip de 27/09 (18:50). Cada **Procura** aparece uma só vez; os
itens são independentes. As duas chaves já existem no `dissertation.bib`, por isso
não é preciso acrescentar entradas.

Verificado nos originais (`~/Downloads/8.A novel multi-kernel 1D ...pdf` e
`~/Downloads/11.Opportunities and Challenges ...pdf`, versão arXiv v3 de abril de 2020;
o .bib cita a versão publicada, Comput. Biol. Med. 122:103801).

## Índice por ficheiro

| Ficheiro | Itens |
|---|---|
| `chapters/StateOfTheArt.tex` | L1, L2, L3, L4, L5 |
| `chapters/Contribution.tex` | L6 |
| `chapters/Introduction.tex` | L7 |
| Cap. 4 (para a tua reescrita) | L8, texto sugerido sem procura/substitui |

## O que os artigos dizem de facto

**article8, Giannakakis et al. 2019 (ACIIW).**
- Dataset próprio: 24 participantes (7 mulheres), 47,3 ± 9,3 anos, e 4 tipos de
  stressor (exposição social, recordar um evento stressante, carga cognitiva e vídeos).
- Entrada: a série de intervalos RR, crua ou normalizada pela baseline neutra de cada
  sujeito. Batimentos ectópicos excluídos (variação de 40% face à média dos 5 anteriores).
- DWNet1D: 3 módulos, cada um com kernels de vários comprimentos; mais de 1,1 milhões
  de parâmetros.
- Janelas de 60 s **deslizantes com passo de 3 s**, com zero-padding. Janelas com mais de 50% de
  padding foram rejeitadas. As janelas foram geradas **depois** da divisão por
  sujeito, por isso não há fuga.
- 6-fold por sujeito (16 treino, 4 validação e 4 teste). **Um modelo por tipo de stressor.**
- **89,8%** é a média de 8 valores (4 stressores × série crua/normalizada); o
  máximo é **99,1%** (carga cognitiva, série crua). F1 médio de 88,1%.
- Os **84%** são de outro artigo do mesmo grupo (a ref. [16] deles: Giannakakis, Marias e
  Tsiknakis, ACII 2019, com features de HRV e ML), e os autores comparam-nos com
  os 99,1%, não com os 89,8%. É uma comparação entre estudos, com protocolos que o artigo não
  descreve, por isso não é um precedente controlado de série contra features. A
  comparação controlada é a tua, E contra F nas mesmas partições.
- Sem explicabilidade: só mostram ativações da camada densa.
- Não fala de biofeedback (ver L7).

**article11, Hong et al. 2020 (revisão sistemática).**
- 191 artigos de deep learning em ECG (jan/2010 a fev/2020), quase todos de doença
  cardíaca (arritmias, enfarte, apneia do sono, identificação biométrica).
- Conclusão (resumo): "a hybrid architecture of a convolutional neural network and
  recurrent neural network ensemble using expert features yields the best results".
- A evidência que apresentam é a **Tabela 3**: classificação de fibrilhação auricular
  no PhysioNet/CinC Challenge 2017 (ECG de uma derivação, 9 a 60 s). Os 9 métodos
  listados ficam entre F1 0,818 e 0,831. O primeiro é "RNN + expert features"; uma CNN
  simples tem 0,83, e o ENCASE (CRNN + expert features, dos próprios autores) tem 0,825.
  A conclusão é fraca e **não é sobre stress**.
- Entre as "expert features" incluem a HRV. Distinguem CNN 1D (kernels ao longo do
  tempo do ECG cru) de CNN 2D (sobre representações transformadas). Também notam que juntar
  expert features a uma rede dá "partial interpretation".
- **Não entra na tabela:** a tabela é de estudos de deteção de stress (dataset,
  dados, exatidão), e uma revisão de ECG em geral não tem nada disso.

---

## L1. Cap. 2, CNN: explicar a convolução 1D
**Ficheiro:** `chapters/StateOfTheArt.tex`

A subsecção só descreve a convolução 2D (imagens, "horizontally and vertically"), mas
os Modelos C, D, F e G usam convolução 1D.

**Procura:** `\subsubsection{Recurrent Neural Networks}`
**Substitui por:**
```latex
The description above is that of a two-dimensional convolution, in which the kernel slides over an image in two directions. For a signal that varies only in time, such as the \gls{ECG}, the convolution is one-dimensional: the kernel is a short vector that slides along the time axis, and each value of the feature map is a weighted sum of a few consecutive samples. Hong et al. \citep{article11} distinguish the two forms in their review of deep learning for \gls{ECG} data: a one-dimensional CNN applies its kernels along the time axis of the raw signal, whereas a two-dimensional CNN typically operates on a transformed representation of it. The same operation can be applied to the series of RR intervals instead of the raw \gls{ECG}, in which case each position of the kernel covers a few consecutive heartbeats, as in the stress recognition model of Giannakakis et al. \citep{article8} (Table \ref{classifiers_new}).

\subsubsection{Recurrent Neural Networks}
```

## L2. Cap. 2, tabela `classifiers_new`: linha do Giannakakis
**Ficheiro:** `chapters/StateOfTheArt.tex`

Fica a seguir ao Hwang (2018). Não há linha para o Hong (ver acima).

**Procura:** `Hwang et al. \citep{article9} & Private & Raw \gls{ECG}, 10 s windows & CNN-LSTM & - & 2018 & 87.39\%, 73.96\% \\ \hline`
**Substitui por:**
```latex
Hwang et al. \citep{article9} & Private & Raw \gls{ECG}, 10 s windows & CNN-LSTM & - & 2018 & 87.39\%, 73.96\% \\ \hline
Giannakakis et al. \citep{article8} & Private & RR interval series, 60 s windows & Multi-kernel 1D CNN & - & 2019 & 89.8\% (mean), up to 99.1\% \\ \hline
```

## L3. Cap. 2, parágrafo do Giannakakis
**Ficheiro:** `chapters/StateOfTheArt.tex`

Entra depois do parágrafo do Xiong, antes de "The accuracies in Table ... are not
directly comparable".

**Procura:** `The accuracies in Table \ref{classifiers_new} are not directly comparable.`
**Substitui por:**
```latex
Giannakakis et al. \citep{article8} applied a one-dimensional CNN directly to the series of RR intervals, instead of to features computed from it. The data were collected from 24 participants exposed to four types of stressor: social exposure, recall of a stressful event, cognitive load and stressful videos. Each module of the network, DWNet1D, applies kernels of several lengths in parallel, which the authors found more consistent across types of stressor than any single kernel length, and the network has more than 1.1 million parameters. The series were used either raw or normalised by each participant's neutral baseline, and were cut into 60-second windows sliding by 3 seconds. Evaluation used six-fold cross-validation by subject, and the windows were generated only after the participants had been assigned to folds, so no participant contributed to both training and testing. A separate model was trained for each type of stressor; the mean accuracy over the eight combinations of stressor and input was 89.8\%, and the best reached 99.1\%. The authors set the best result against the 84\% obtained on the same dataset in another study of their group, with \gls{HRV} features and classical machine learning, which is a comparison between two studies rather than between two models evaluated on the same partitions.

The accuracies in Table \ref{classifiers_new} are not directly comparable.
```

## L4. Cap. 2, parágrafo do Hong
**Ficheiro:** `chapters/StateOfTheArt.tex`

Entra logo a seguir ao parágrafo do Haque, que também fala de arquiteturas híbridas.

**Procura:** `Even though hybrid architectures can be a promising future prospect for accurate results.`
**Substitui por:**
```latex
Even though hybrid architectures can be a promising future prospect for accurate results.

Hong et al. \citep{article11} reviewed 191 studies that applied deep learning to \gls{ECG} data between 2010 and February 2020, most of them on the detection and classification of cardiac disease. They conclude that deep learning reaches an accuracy comparable to that of approaches based on expert features, and that an ensemble combining convolutional and recurrent networks with expert features, among which they include \gls{HRV} measures, gives the best results. The comparison they present in support comes from the classification of atrial fibrillation in single-lead recordings of 9 to 60 seconds, from the 2017 PhysioNet/Computing in Cardiology Challenge, where the nine methods listed are within 1.5 points of F1 score of each other. The conclusion has not been tested on stress.
```

## L5. Cap. 2, fecho da secção: o estudo mais próximo
**Ficheiro:** `chapters/StateOfTheArt.tex`

**Procura:** `None of the studies reviewed evaluates a single-modality, subject-independent model on thirty-second windows.`
**Substitui por:**
```latex
Giannakakis et al. \citep{article8} come closest, with a single modality and an evaluation by subject, but on 60-second windows and with a separate model for each type of stressor. None of the studies reviewed evaluates a single-modality, subject-independent model on thirty-second windows.
```

## L6. Cap. 3, "State of the Art Models": precedentes do F e do G
**Ficheiro:** `chapters/Contribution.tex`

Esta subsecção lista os modelos da literatura que serviram de referência, mas só
tem os que se relacionam com os Modelos A–D.

**Procura:** `\subsection{Deep Learning Models Selected}`
**Substitui por:**
```latex
Two further studies are the closest precedents for the models that operate on the interval series, Models F and G, described below. Giannakakis et al. \citep{article8} applied a one-dimensional CNN directly to the RR interval series and evaluated it by subject, although on 60-second windows and with more than 1.1 million parameters, against 27,969 for Model F. Hong et al. \citep{article11}, in their review of deep learning for \gls{ECG} data, found the best results in models that combine a network with expert features computed from the same signal, mainly in arrhythmia classification; Model G follows that design by combining the interval series with the eight summary features.

\subsection{Deep Learning Models Selected}
```

## L7. Introdução: tirar o article8 da frase do biofeedback [verificar]
**Ficheiro:** `chapters/Introduction.tex`

A frase diz "it has been proven that observing real-time biofeedback can help as a
tool for stress management" e cita o article8. O Giannakakis não fala de feedback
nem de gestão do stress. Não verifiquei os outros três (article5, article7, article37).

**Procura:** `\citep{article5, article8, article7, article37}`
**Substitui por:**
```latex
\citep{article5, article7, article37}
```

## L8. Cap. 4, frases para a tua reescrita (opcional)

Não têm procura/substitui porque vais reescrever o capítulo. Se quiseres ligar o Hong
aos resultados:

- **Escolha do modelo (4.6.1, `sec:model_choice`):** "The combination of a network with expert features,
  which Hong et al. \citep{article11} found to perform best in arrhythmia
  classification, did not improve on the interval series alone here: Models F and G
  do not differ under any protocol."
- **Explicabilidade (4.6.2, `sec:discussion_xai`):** "Hong et al. \citep{article11} note that adding
  interpretable expert features to a deep model allows a partial interpretation of
  its output. In Model G the summary features are computed from the same series the
  model receives, and their attributions cannot be read one feature at a time."
