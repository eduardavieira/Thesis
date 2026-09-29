# Literatura recente: Cap. 2, com ligações nos Cap. 3 e 4

Validado contra o zip de 27/09 (18:50) **depois de aplicado o `correcoes_literatura.md`
(L1–L7)**: aplica esse primeiro. Cada **Procura** aparece uma só vez nesse momento. Os itens
R15, R16 e R18 procuram texto que só existe depois de L5, L1 e L6; os restantes não dependem
do L e podem ir por qualquer ordem.

O objetivo é trocar ou completar referências anteriores a 2020 onde a frase fala do estado
atual do campo. Os clássicos ficam (Task Force 1996, Castaldo 2015, Kim 2018, Pecchia 2018,
LeCun 2015, Vaswani 2017, Sundararajan 2017): não há mal nenhum numa referência antiga
para um facto que não mudou.

Tudo o que as frases afirmam foi verificado nos originais (PDF ou texto integral) a 28/09.
Os metadados do .bib vêm do Crossref, exceto o Mortensen (DOI tirado do PDF: o Crossref
devolve um artigo de 2026 com o mesmo título e outros autores) e a tese de Ismail Fawaz
(arXiv). Resumo em `.claude/skills/thesis-corrections/references/facts.md`.

**Correção a uma conversa anterior:** o artigo que chamei "Byun et al. 2025b" é de
**Lee**, Cho, Kim, Jeon & Byun 2025 (chave `lee2025deep`).

## Índice por ficheiro

| Ficheiro | Itens |
|---|---|
| `chapters/StateOfTheArt.tex` | R1–R16 |
| `chapters/Contribution.tex` | R17, R18 |
| `chapters/Applications.tex` | R19–R22 |
| `dissertation.bib` | B (20 entradas novas) |

## O que os artigos dizem de facto

**Os que tocam no teu resultado central**
- **Prajod, Mahesh & André 2024** (ICMI '24). HRV com 22 features (NeuroKit2), janelas de 60 s,
  LOSO, quatro datasets: WESAD, SWELL-KW, ForDigitStress e VerBIO. SWELL dentro do próprio
  dataset: F1 0,60–0,67. Modelos do WESAD testados no SWELL: F1 0,42–0,47, exatidão 0,44–0,48.
  Modelos do SWELL testados no WESAD: exatidão 0,48–0,68. Treinar com os dois juntos baixa o F1
  no WESAD cerca de 12 pontos. Os três datasets de stress social transferem bem entre si
  (HRV do pulso ótico). Concluem que o tipo de stressor é "a crucial factor". **Não falam do
  sentido da frequência cardíaca:** essa explicação é a tua contribuição.
- **Prajod & André 2022** (ICMLA). WESAD e SWELL: o deep learning sobre ECG bruto é o melhor
  dentro de cada dataset, mas as features de HRV transferem "considerably" melhor entre eles.
- **Mishra et al. 2020** (IMWUT). 90 participantes de 4 estudos. Escolher o limiar nos dados do
  estudo novo (por agrupamento) funcionou melhor do que usar o limiar do treino.
- **Vos et al. 2023a** (IJMI). Revisão sistemática: a maioria dos estudos de stress com
  wearables não mostra generalização.
- **Vos et al. 2023b** (JBI). EDA + FC, janelas de 25 s. Um modelo treinado no SWELL falha no
  WESAD. Treinado no SWELL + NEURO + UBFC-Phys (sem WESAD), atinge 85% no WESAD. **É o
  contraponto ao teu resultado:** junta datasets e generaliza, mas usa EDA além da FC.
- **Dahal et al. 2023** (`newArtc6`, já no .bib e nunca citado). Junta SWELL e WESAD com
  features pré-calculadas e K-fold sobre amostras: 99,5%. Deixando de fora cada sujeito do
  WESAD, a exatidão fica entre 31% e 95%. O mRMR escolhe features diferentes para cada dataset.
- **Bahameish et al. 2024.** Treino num dataset próprio (PPG, janelas de 60 s). SWELL e WESAD
  juntos servem só de teste externo: stress vs relax F1 86,3%; stress vs neutro F1 65,8%.
  Features: MeanRR, RMSSD e HF (stress vs neutro); MeanRR, SDNN e SD2 (stress vs relax).
- **Kapoor & Narayanan 2023** (Patterns). Fuga de dados em 294 artigos de 17 áreas, nalguns
  casos com conclusões muito otimistas.
- **Mortensen et al. 2023** (IEEE Access). 1D CNN sobre features de HRV no tempo e na
  frequência de um SWELL pré-calculado (410 322 registos de 25 pessoas), 3 classes. 99,9% em
  5-fold e 96,1% em 70/30. O artigo não refere divisão por participante.

**HRV, ultra-curto e wearables**
- **Quigley et al. 2024** (Psychophysiology). Guia da Society for Psychophysiological Research
  para medir FC e HRV por ECG e PPG, em laboratório e ambulatório, com checklists.
- **Pham et al. 2021** (Sensors). Revisão dos índices de HRV (tempo, frequência, não lineares)
  com tutorial.
- **Immanuel et al. 2023** (Neuropsychobiology). Scoping review sobre HRV e mudanças de stress
  psicológico em adultos saudáveis.
- **Shaffer, Meehan & Zerr 2020** (Front. Neurosci.). A correlação e a comparação de médias
  não bastam para validar HRV ultra-curta; é preciso usar limites de concordância definidos à
  partida. Num dos estudos revistos: FC válida a partir de 10 s; RMSSD, SDNN e pNN50 só a
  partir de 60 s.
- **Lee et al. 2022** (`newArtc7`, já no .bib, só num comentário). 74 polícias, TSST e filme
  de terror, SVM com LOSO: 90,5% com HRV ultra-curta (< 5 min) e features de decomposição
  empírica em modos.
- **Velmovitsky et al. 2022.** ECG de 30 s do Apple Watch; 33 pessoas, 2 semanas, pedidas 6
  medições por dia; rótulos pelo DASS-21. RF com exatidão 0,55 e recall de stress 0,43.
- **Makowski et al. 2021.** NeuroKit2: deteção de picos R e correção de batimentos no ECG.

**Normalização e calibração por pessoa**
- **Byun et al. 2025** (Front. Psychiatry 15). HRV de 5 min, divisão por sujeito. RF sem
  normalização: 0,67–0,73. Com z-score por participante (sobre todos os seus dados): 0,90–0,96.
  SHAP: depois da normalização, o RR médio é a feature mais importante.
- **Nkurikiyeyezu et al. 2020** (Sensors and Materials). Modelo genérico com 42,5 ± 19,9%, que
  sobe para 95,2 ± 0,5% com 100 amostras de calibração da pessoa, acrescentadas ao treino.
  Têm 204 885 amostras de HRV do SWELL, o que sugere janelas muito sobrepostas (não
  confirmei no método); por isso o ganho deve ser otimista. Só entra no Cap. 4 (R20).

**Convolução 1D**
- **Lee et al. 2025** (Front. Psychiatry 16). Adaptação 1D da ResNet34 sobre a série RR
  reamostrada a 4 Hz; 147 pessoas (41 com depressão major, 47 com perturbação de pânico, 59
  controlos); aritmética mental vs baseline. CV 10×10-fold por participante. Segmentos de 5 min:
  86,5%–89,7%; segmentos de 1 min: cerca de 80%.
- **Kiranyaz et al. 2021** (MSSP). CNN 1D com estado da arte em classificação de dados
  biomédicos, monitorização estrutural e deteção de falhas. A estrutura compacta permite
  hardware barato e tempo real, e funcionam com poucos dados.
- **Ismail Fawaz 2020** (tese; é o teu `artigouni1.pdf`). Nove arquiteturas de deep learning
  nos 85 datasets univariados do arquivo UCR/UEA. As duas melhores, FCN e ResNet, acabam em
  global average pooling ao longo do tempo, e a ResNet é significativamente a melhor.

---

## R1. Cap. 2, definição de stress
**Ficheiro:** `chapters/StateOfTheArt.tex`

**Procura:** `\citep{article4, article25}.`
**Substitui por:**
```latex
\citep{article4, article25}. A recent scoping review of \gls{AI} applied to \gls{HRV} for stress monitoring found the same lack of agreement among the definitions of stress used in the 15 studies it included \citep{zimatore2026hrv}.
```

## R2. Cap. 2, "Detection of Stress": primeira frase
**Ficheiro:** `chapters/StateOfTheArt.tex`

O Haque 2024 diz textualmente "There is no agreed standard for stress evaluation at present".

**Procura:** `At present, there is no universally recognised standard for stress evaluation \citep{article4}.`
**Substitui por:**
```latex
At present, there is no universally recognised standard for stress evaluation \citep{article4, haque2023stateofart}.
```

## R3. Cap. 2, definição de HRV
**Ficheiro:** `chapters/StateOfTheArt.tex`

**Procura:** `(RR series) \citep{article22}.`
**Substitui por:**
```latex
(RR series) \citep{article22, quigley2024guidelines}.
```

**Procura:** `consequently stress levels \citep{article13}.`
**Substitui por:**
```latex
consequently stress levels \citep{article13, immanuel2023hrv}.
```

## R4. Cap. 2, categorias de medidas e "recent work" de 2014
**Ficheiro:** `chapters/StateOfTheArt.tex`

"Recent work" cita um artigo de 2014.

**Procura:** `geometric, and nonlinear \citep{article3} and recent work has shown that some of the \gls{HRV} features can potentially be used for distinguishing a subject’s normal mental state from a stressed one \citep{article2}.`
**Substitui por:**
```latex
geometric, and nonlinear \citep{article3, pham2021hrv}, and several studies have shown that some of the \gls{HRV} features can potentially be used for distinguishing a subject’s normal mental state from a stressed one \citep{article2, article4, immanuel2023hrv}.
```

## R5. Cap. 2, abertura do Related Work
**Ficheiro:** `chapters/StateOfTheArt.tex`

A frase "there are few studies ... employing deep learning" cita artigos de 2018 e 2019 e
já não é verdade.

**Procura:** `And there are few studies in the literature employing deep learning techniques in stress recognition through \gls{HRV} derived features \citep{article8, article37}.`
**Substitui por:**
```latex
Machine learning, and more recently deep learning, has since been applied to the problem in many studies, but reviews describe a heterogeneous field. Haque et al. \citep{haque2023stateofart} reviewed 43 studies of stress prediction from \gls{HRV} with artificial intelligence. Zimatore et al. \citep{zimatore2026hrv}, in a scoping review of 15 studies, found that most used classical supervised methods such as random forests and support vector machines, with accuracies between 56\% and 92\%, and concluded that differences in datasets, stress protocols, sensors and validation make a ranking of the models impossible. Vos et al. \citep{vos2023generalizable} found that most machine learning studies of stress from wearable devices do not show that their models generalise to unseen data.
```

## R6. Cap. 2, wearables: ECG de 30 s do Apple Watch
**Ficheiro:** `chapters/StateOfTheArt.tex`

**Procura:** `electrodermal activity, and motion data.`
**Substitui por:**
```latex
electrodermal activity, and motion data.

Velmovitsky et al. \citep{velmovitsky2022apple} used the 30-second \gls{ECG} recorded by the Apple Watch, the same window length used in this work, to derive \gls{HRV} in 33 participants over two weeks, with participants asked to record six times a day and stress labels taken from the DASS-21 questionnaire. A random forest reached an accuracy of 55\%, with a recall of 43\% for the stress class, and the authors note that it remains unclear whether a 30-second reading is sufficient for stress prediction.
```

## R7. Cap. 2, ultra-curto: critério de validade e um estudo recente
**Ficheiro:** `chapters/StateOfTheArt.tex`

**Procura:** `And thus a 30 s duration is suitable for calculating \gls{HRV} metrics using time domain features but not necessarily frequency domain features.`
**Substitui por:**
```latex
And thus a 30 s duration is suitable for calculating \gls{HRV} metrics using time domain features but not necessarily frequency domain features.

Shaffer et al. \citep{shaffer2020ultrashort} add a methodological requirement to this literature. A high correlation between ultra-short and five-minute values, or the absence of a difference between their group means, does not show that the shorter measurement can replace the longer one; that requires limits of agreement with an acceptable difference defined in advance. In one of the studies they review, heart rate was valid from 10 seconds, whereas RMSSD, SDNN and pNN50 required 60 seconds. Lee et al. \citep{newArtc7} classified acute stress in 74 police officers exposed to the \gls{TSST} and to a horror film, and reached 90.5\% accuracy under leave-one-subject-out validation with ultra-short \gls{HRV}, below five minutes, and features based on empirical mode decomposition.
```

## R8. Cap. 2, Feature Selection: o que a literatura recente usa
**Ficheiro:** `chapters/StateOfTheArt.tex`

As tabelas de features ficam (são uma revisão do que se usava); este parágrafo junta o
estado atual.

**Procura:** `Mean RR, SDNN, RMSSD, pNN50, LF, HF, and LF/HF.`
**Substitui por:**
```latex
Mean RR, SDNN, RMSSD, pNN50, LF, HF, and LF/HF.

The same measures dominate recent work. In the 15 studies reviewed by Zimatore et al. \citep{zimatore2026hrv}, RMSSD was used in 11, HF and SDNN in 9 each and pNN50 in 8, and only 5 applied a method of feature selection or feature importance. Where the features are selected from the data, the selection depends on the dataset: Dahal et al. \citep{newArtc6} selected eight features by minimum redundancy and maximum relevance and obtained a different set for SWELL-KW, for WESAD and for the two combined, and Bahameish et al. \citep{bahameish2024strategies} retained mean RR, RMSSD and HF to separate stress from a neutral state, and mean RR, SDNN and SD2 to separate stress from relaxation. Byun et al. \citep{byun2025automated} found with \gls{SHAP} that the mean RR interval became the most important feature once the features were normalised within each participant.
```

## R9. Cap. 2, ECG Preprocessing: guias e ferramentas atuais
**Ficheiro:** `chapters/StateOfTheArt.tex`

A secção só tem referências de 1996 a 2019. Entra como parágrafo novo entre a frase das
fontes de ruído e o parágrafo dos batimentos ectópicos. Nota: a tese não usa o hrv-analysis
(a correção dos intervalos é própria, Cap. 3); aqui aparece só como exemplo de ferramenta.

**Procura:** `Ectopic beats, arrhythmic events, missing data`
**Substitui por:**
```latex
Current guidelines for psychophysiological research summarise good practice for measuring heart rate and \gls{HRV} from the \gls{ECG} and from the optical pulse signal of wearable devices, in the laboratory and in ambulatory recordings \citep{quigley2024guidelines}. Open-source tools implement the processing steps: NeuroKit2 detects the R-peaks and corrects artefactual beats in the \gls{ECG} \citep{makowski2021neurokit2}, and the hrv-analysis package removes outliers and ectopic intervals from an RR series and interpolates the removed values \citep{hrvanalysispackage}.

Ectopic beats, arrhythmic events, missing data
```

## R10. Cap. 2, ECG Preprocessing: normalização por participante
**Ficheiro:** `chapters/StateOfTheArt.tex`

Liga à calibração da 4.6.3: normalizar por pessoa é prática comum.

**Procura:** `each subject’s baseline was used to normalise data for all subsequent feature analyses.`
**Substitui por:**
```latex
each subject’s baseline was used to normalise data for all subsequent feature analyses. Normalisation relative to each participant is common in recent work. Prajod et al. \citep{prajod2024stressor} scaled each \gls{HRV} feature by the minimum and maximum of five minutes of that participant's neutral data, and Byun et al. \citep{byun2025automated} standardised the features with the mean and standard deviation of each participant's own recordings, which raised the accuracy of a random forest from between 67\% and 73\% to between 90\% and 96\% in their three groups of participants.
```

## R11. Cap. 2, tabela `classifiers_new`: Mortensen e Lee
**Ficheiro:** `chapters/StateOfTheArt.tex`

Ficam a seguir à linha do Xiong (a última); a procura é o fim dessa linha.

**Procura:** `& Transformer & - & 2025 & 96.06\% \\ \hline`
**Substitui por:**
```latex
& Transformer & - & 2025 & 96.06\% \\ \hline
Mortensen et al. \citep{mortensen2023multiclass} & \gls{SWELL-KW} & Time-domain and frequency-domain features of HRV. & 1D CNN & - & 2023 & 99.9\% (3 classes) \\ \hline
Lee et al. \citep{lee2025deep} & Private & RR interval series, 1 and 5 min segments & 1D ResNet34 & - & 2025 & 86.5--89.7\% (5 min) \\ \hline
```

## R12. Cap. 2, parágrafos do Mortensen e do Lee
**Ficheiro:** `chapters/StateOfTheArt.tex`

Entram a seguir ao parágrafo do Xiong. Se L3 já estiver aplicado, ficam antes do parágrafo
do Giannakakis.

**Procura:** `The StressSentry-FHE achieved a 96.06\% accuracy.`
**Substitui por:**
```latex
The StressSentry-FHE achieved a 96.06\% accuracy.

Mortensen et al. \citep{mortensen2023multiclass} classified the three conditions of \gls{SWELL-KW} with a one-dimensional CNN applied to time-domain and frequency-domain \gls{HRV} features, and reported 99.9\% accuracy under five-fold cross-validation and 96.1\% with a 70/30 split. The features came from a pre-computed table derived from \gls{SWELL-KW}, with 410,322 records from 25 participants, and the paper does not report a partition by participant.

Lee et al. \citep{lee2025deep} applied a one-dimensional adaptation of ResNet34 directly to the RR interval series, resampled at 4 Hz, of 147 participants, including patients with major depressive disorder and panic disorder, recorded during a resting baseline and a mental arithmetic task. With the folds formed by participant, the accuracy was between 86.5\% and 89.7\% for five-minute segments and about 80\% for one-minute segments.
```

## R13. Cap. 2, fuga de dados
**Ficheiro:** `chapters/StateOfTheArt.tex`

Parágrafo "The accuracies ... are not directly comparable", que já critica o Xia. Com o
Mortensen (99,9%) na tabela, "the two highest" deixa de ser verdade: o primeiro bloco corrige
isso. O Mortensen também usa features de HRV pré-calculadas, com medidas na frequência.

**Procura:** `The two highest, Jadav and Chauhan \citep{jadav2024lstm} and Xia et al. \citep{song2024cnn}, are obtained`
**Substitui por:**
```latex
Three of the highest, Mortensen et al. \citep{mortensen2023multiclass}, Xia et al. \citep{song2024cnn} and Jadav and Chauhan \citep{jadav2024lstm}, are obtained
```

**Procura:** `a result more readily explained by subject-level memorisation than by a physiological signal.`
**Substitui por:**
```latex
a result more readily explained by subject-level memorisation than by a physiological signal. The same concern applies to the result of Mortensen et al. \citep{mortensen2023multiclass}. Kapoor and Narayanan \citep{kapoor2023leakage} identified leakage between training and test data in 294 papers across 17 scientific fields that use machine learning, in some cases with conclusions that were far too optimistic.
```

## R14. Cap. 2, avaliação entre datasets (o mais importante)
**Ficheiro:** `chapters/StateOfTheArt.tex`

"Remains rare" já não é verdade. Os estudos abaixo encontraram a mesma falha de transferência
entre SWELL e WESAD; a explicação fisiológica (o sentido da FC) continua a ser tua.

**Procura:** `Evaluation across datasets remains rare in the literature.`
**Substitui por:**
```latex
Only a few studies have evaluated stress models across datasets.
```

O bloco começa com uma linha em branco, para o Mishra abrir parágrafo novo depois do Benchekroun.

**Procura:** `Section \ref{sec:per_dataset} applies the same kind of test to SWELL-KW and WESAD.`
**Substitui por:**
```latex

Mishra et al. \citep{mishra2020reproducibility} tested stress models on 90 participants from four independent studies and found that choosing the decision threshold on the data of the new study, by clustering, worked better than keeping the threshold learned in training. Prajod and André \citep{prajod2022generalizability} trained models on WESAD and on SWELL-KW and tested each on the other: deep learning models applied to the raw \gls{ECG} were the best within each dataset, but models based on \gls{HRV} features transferred considerably better. Prajod et al. \citep{prajod2024stressor} extended the comparison to four datasets under leave-one-subject-out validation. Models trained on WESAD and tested on SWELL-KW reached F1 scores between 0.42 and 0.47 and accuracies below 50\%, models trained on SWELL-KW reached accuracies between 48\% and 68\% on WESAD, and training on both datasets together lowered the F1 score on WESAD by about 12 points. Models built on the three datasets with a social stressor, WESAD, ForDigitStress \citep{heimerl2024stressor} and VerBIO \citep{VerBIO}, transferred well between them, and the authors concluded that the type of stressor is a crucial factor. Vos et al. \citep{vos2023ensemble} likewise found that a model trained on SWELL-KW performed poorly on WESAD, whereas a model trained on a combined dataset built from SWELL-KW and two further datasets, with \gls{EDA} as well as heart rate, reached 85\% on WESAD.

Pooled datasets also appear in studies that do not test the transfer. Dahal et al. \citep{newArtc6} combined SWELL-KW and WESAD and reported 99.5\% accuracy, with pre-computed features and cross-validation over samples rather than participants; when each WESAD participant was held out, accuracy ranged from 31\% to 95\%. Bahameish et al. \citep{bahameish2024strategies} trained a model on a dataset of their own and used SWELL-KW and WESAD, combined, as an external test set, where it separated stress from relaxation with an F1 score of 86.3\% but stress from a neutral state with 65.8\%. Section \ref{sec:per_dataset} applies the same kind of test to SWELL-KW and WESAD.
```

## R15. Cap. 2, o estudo mais próximo [depende de L5]
**Ficheiro:** `chapters/StateOfTheArt.tex`

O Lee 2025 é tão próximo como o Giannakakis (série RR, uma modalidade, avaliação por sujeito).
A frase seguinte do L5 ("None of the studies reviewed ... thirty-second windows") mantém-se
verdadeira.

**Procura:** `Giannakakis et al. \citep{article8} come closest, with a single modality and an evaluation by subject, but on 60-second windows and with a separate model for each type of stressor.`
**Substitui por:**
```latex
Giannakakis et al. \citep{article8} and Lee et al. \citep{lee2025deep} come closest: both apply a one-dimensional network to the RR interval series of a single modality and evaluate it by subject, but on windows of one minute or longer, and Giannakakis et al. train a separate model for each type of stressor.
```

## R16. Cap. 2, CNN 1D: referências atuais [depende de L1]
**Ficheiro:** `chapters/StateOfTheArt.tex`

Completa o parágrafo que o L1 acrescenta à subsecção de CNN, que hoje só cita 2015–2020.

**Procura:** `as in the stress recognition model of Giannakakis et al. \citep{article8} (Table \ref{classifiers_new}).`
**Substitui por:**
```latex
as in the stress recognition model of Giannakakis et al. \citep{article8} (Table \ref{classifiers_new}). One-dimensional CNNs have reached state-of-the-art performance in applications that include the classification of biomedical signals, structural health monitoring and fault detection, and their compact structure makes a real-time implementation on low-cost hardware feasible \citep{kiranyaz2021onedcnn}. In a comparison of nine deep learning architectures on the 85 univariate datasets of the UCR/UEA time series classification archive, the two best, a fully convolutional network and a residual network, both end in a global average pooling over the time axis \citep{ismailfawaz2020thesis}, the operation that Model F (Section \ref{models}) adapts to series of variable length.
```

## R17. Cap. 3, Model E: o MLP sobre features de HRV na literatura
**Ficheiro:** `chapters/Contribution.tex`

**Procura:** `it serves as the control against which any advantage of the sequential input must be measured.`
**Substitui por:**
```latex
it serves as the control against which any advantage of the sequential input must be measured. A multilayer perceptron on \gls{HRV} features is also a common reference in the literature: Prajod et al. \citep{prajod2024stressor} used one with two hidden layers, and it was the best of their three classifiers within three of the four datasets they evaluated.
```

## R18. Cap. 3, precedentes do F: Lee 2025 [depende de L6]
**Ficheiro:** `chapters/Contribution.tex`

**Procura:** `Model G follows that design by combining the interval series with the eight summary features.`
**Substitui por:**
```latex
Model G follows that design by combining the interval series with the eight summary features. Lee et al. \citep{lee2025deep} also applied a one-dimensional network, an adaptation of ResNet34, to the RR interval series and evaluated it by participant, on segments of one and five minutes.
```

## R19. Cap. 4, 4.3: os modelos da série e a transferência entre datasets
**Ficheiro:** `chapters/Applications.tex`

Fica a seguir à frase que explica porque é que F e G perdem mais do aleatório para o
GroupKFold.

**Procura:** `which a series, with more detail per window than eight statistics, can exploit.`
**Substitui por:**
```latex
which a series, with more detail per window than eight statistics, can exploit. A related pattern has been reported across datasets: Prajod and André \citep{prajod2022generalizability} found that deep learning models applied to the raw \gls{ECG} were the best within WESAD and within SWELL-KW, but transferred to the other dataset worse than models based on \gls{HRV} features.
```

## R20. Cap. 4, 4.6.3: a calibração na literatura
**Ficheiro:** `chapters/Applications.tex`

Parágrafo novo entre os resultados da calibração e "Three limits apply to this result".
O Nkurikiyeyezu acrescenta amostras da pessoa ao treino, o que corresponde à condição
"Retrained" da Tabela `tab:personalised`. Não entra a ressalva das janelas sobrepostas
porque não a verifiquei no artigo; a frase só diz o que é certo (mesma sessão, como aqui).

**Procura:** `which is a direct measurement of the problem described above.`
**Substitui por:**
```latex
which is a direct measurement of the problem described above.

Adapting the model to the person is common in the literature. Features are normalised by each participant's neutral data \citep{article8, prajod2024stressor} or by all of their recordings \citep{byun2025automated}, and Mishra et al. \citep{mishra2020reproducibility} found that a threshold set on the data of a new study, by clustering, worked better than the threshold learned in training. Nkurikiyeyezu et al. \citep{nkurikiyeyezu2020person} evaluated on SWELL-KW and WESAD a calibration that adds 100 samples of each person to the training data, which corresponds to the retrained condition of Table \ref{tab:personalised}; in one of their results it raised the accuracy of a generic model from 42.5\% to 95.2\%, with the calibration and test samples taken from the same recording session, as here. Normalisation needs only neutral or unlabelled data from the person, whereas the calibration used here requires labelled windows of both classes.
```

## R21. Cap. 4, Relation to Previous Work: a janela de 30 s
**Ficheiro:** `chapters/Applications.tex`

Fica no fim da frase sobre a variância das estatísticas em 30 a 50 intervalos, antes de
"Whether this holds for weaker stressors". O AUC de 0,882 do hr\_mean sozinho é o da 4.4
(`sec:explainability`).

**Procura:** `statistics whose estimation variance is high on 30 to 50 intervals.`
**Substitui por:**
```latex
statistics whose estimation variance is high on 30 to 50 intervals. This agrees with the limits compiled by Shaffer et al. \citep{shaffer2020ultrashort}: in one of the studies they review, heart rate was valid from 10 seconds, whereas RMSSD, SDNN and pNN50 required 60 seconds. In this work, the mean heart rate of the window alone separates the two classes of the WESAD test partition with an \gls{AUC} of 0.882 (Section \ref{sec:explainability}).
```

## R22. Cap. 4, Relation to Previous Work: juntar datasets e transferência
**Ficheiro:** `chapters/Applications.tex`

Dois blocos no parágrafo "The negative result of this work is also a methodological
contribution". A frase sobre a frequência cardíaca é descritiva de propósito: não diz que
é a causa (decisão de 27/09 de não pôr "the cause is traced" no texto).

**Procura:** `it rests on the assumption that the labels of different protocols describe the same state.`
**Substitui por:**
```latex
it rests on the assumption that the labels of different protocols describe the same state. Studies that pool SWELL-KW and WESAD report accuracies up to 99.5\% when windows of the same participant fall on both sides of the partition \citep{newArtc6}, and a model trained on SWELL-KW together with two further datasets, using electrodermal activity as well as heart rate, reached 85\% on WESAD \citep{vos2023ensemble}.
```

**Procura:** `a loss of performance has been read as a problem of generalisation \citep{newArtc4}; an \gls{AUC} below 0.5 indicates something different, that the labels themselves disagree.`
**Substitui por:**
```latex
a loss of performance has been read as a problem of generalisation \citep{newArtc4} or attributed to the type of stressor \citep{prajod2024stressor}; an \gls{AUC} below 0.5 indicates something different, that the labels themselves disagree. Prajod et al. also found accuracies below 50\% when models trained on WESAD were tested on SWELL-KW, and the direction of the heart rate response differs between the two datasets: mean heart rate rises under stress in all 15 WESAD participants and falls in all 22 SWELL-KW participants (Section \ref{sec:per_dataset}).
```

## B. Entradas novas para o .bib

`zimatore2026hrv` segue o .bib da MDPI (ano 2026; o Crossref dá 2025 por ter sido publicado
a 19/12/2025). Todas as 20 entradas são citadas por algum item (o `nkurikiyeyezu2020person` só no R20).

acrescenta ao `dissertation.bib`:
```bibtex
@article{zimatore2026hrv,
  author  = {Zimatore, Giovanna and Russo, Samuele and Gallotta, Maria Chiara and Passalacqua, Giordano and Zaborova, Victoria and Campanella, Matteo and Fiani, Francesca and Baldari, Carlo and Napoli, Christian and Randieri, Cristian},
  title   = {{HRV} in Stress Monitoring by {AI}: A Scoping Review},
  journal = {Applied Sciences},
  volume  = {16},
  number  = {1},
  pages   = {23},
  year    = {2026},
  doi     = {10.3390/app16010023}
}

@article{quigley2024guidelines,
  author  = {Quigley, Karen S. and Gianaros, Peter J. and Norman, Greg J. and Jennings, J. Richard and Berntson, Gary G. and de Geus, Eco J. C.},
  title   = {Publication guidelines for human heart rate and heart rate variability studies in psychophysiology---Part 1: Physiological underpinnings and foundations of measurement},
  journal = {Psychophysiology},
  volume  = {61},
  number  = {9},
  pages   = {e14604},
  year    = {2024},
  doi     = {10.1111/psyp.14604}
}

@article{pham2021hrv,
  author  = {Pham, Tam and Lau, Zen Juen and Chen, S. H. Annabel and Makowski, Dominique},
  title   = {Heart Rate Variability in Psychology: A Review of {HRV} Indices and an Analysis Tutorial},
  journal = {Sensors},
  volume  = {21},
  number  = {12},
  pages   = {3998},
  year    = {2021},
  doi     = {10.3390/s21123998}
}

@article{immanuel2023hrv,
  author  = {Immanuel, Sarah and Teferra, Meseret N. and Baumert, Mathias and Bidargaddi, Niranjan},
  title   = {Heart Rate Variability for Evaluating Psychological Stress Changes in Healthy Adults: A Scoping Review},
  journal = {Neuropsychobiology},
  volume  = {82},
  number  = {4},
  pages   = {187--202},
  year    = {2023},
  doi     = {10.1159/000530376}
}

@article{shaffer2020ultrashort,
  author  = {Shaffer, Fred and Meehan, Zachary M. and Zerr, Christopher L.},
  title   = {A Critical Review of Ultra-Short-Term Heart Rate Variability Norms Research},
  journal = {Frontiers in Neuroscience},
  volume  = {14},
  pages   = {594880},
  year    = {2020},
  doi     = {10.3389/fnins.2020.594880}
}

@article{velmovitsky2022apple,
  author  = {Velmovitsky, Pedro Elkind and Alencar, Paulo and Leatherdale, Scott T. and Cowan, Donald and Morita, Plinio Pelegrini},
  title   = {Using {Apple Watch} {ECG} Data for Heart Rate Variability Monitoring and Stress Prediction: A Pilot Study},
  journal = {Frontiers in Digital Health},
  volume  = {4},
  pages   = {1058826},
  year    = {2022},
  doi     = {10.3389/fdgth.2022.1058826}
}

@article{makowski2021neurokit2,
  author  = {Makowski, Dominique and Pham, Tam and Lau, Zen J. and Brammer, Jan C. and Lespinasse, Fran{\c{c}}ois and Pham, Hung and Sch{\"o}lzel, Christopher and Chen, S. H. Annabel},
  title   = {{NeuroKit2}: A {Python} toolbox for neurophysiological signal processing},
  journal = {Behavior Research Methods},
  volume  = {53},
  number  = {4},
  pages   = {1689--1696},
  year    = {2021},
  doi     = {10.3758/s13428-020-01516-y}
}

@article{bahameish2024strategies,
  author  = {Bahameish, Mariam and Stockman, Tony and Requena Carri{\'o}n, Jes{\'u}s},
  title   = {Strategies for Reliable Stress Recognition: A Machine Learning Approach Using Heart Rate Variability Features},
  journal = {Sensors},
  volume  = {24},
  number  = {10},
  pages   = {3210},
  year    = {2024},
  doi     = {10.3390/s24103210}
}

@article{byun2025automated,
  author  = {Byun, Sangwon and Kim, Ah Young and Shin, Min-Sup and Jeon, Hong Jin and Cho, Chul-Hyun},
  title   = {Automated classification of stress and relaxation responses in major depressive disorder, panic disorder, and healthy participants via heart rate variability},
  journal = {Frontiers in Psychiatry},
  volume  = {15},
  pages   = {1500310},
  year    = {2025},
  doi     = {10.3389/fpsyt.2024.1500310}
}

@article{lee2025deep,
  author  = {Lee, Kyung Hyun and Cho, Chul-Hyun and Kim, Ah Young and Jeon, Hong Jin and Byun, Sangwon},
  title   = {Deep learning-based stress detection from {RR} intervals in major depressive disorder, panic disorder, and healthy individuals},
  journal = {Frontiers in Psychiatry},
  volume  = {16},
  pages   = {1672260},
  year    = {2025},
  doi     = {10.3389/fpsyt.2025.1672260}
}

@article{mortensen2023multiclass,
  author  = {Mortensen, J. A. and Mollov, M. E. and Chatterjee, A. and Ghose, D. and Li, F. Y.},
  title   = {Multi-Class Stress Detection Through Heart Rate Variability: A Deep Neural Network Based Study},
  journal = {IEEE Access},
  volume  = {11},
  pages   = {57470--57480},
  year    = {2023},
  doi     = {10.1109/ACCESS.2023.3274478}
}

@article{kapoor2023leakage,
  author  = {Kapoor, Sayash and Narayanan, Arvind},
  title   = {Leakage and the reproducibility crisis in machine-learning-based science},
  journal = {Patterns},
  volume  = {4},
  number  = {9},
  pages   = {100804},
  year    = {2023},
  doi     = {10.1016/j.patter.2023.100804}
}

@article{mishra2020reproducibility,
  author  = {Mishra, Varun and Sen, Sougata and Chen, Grace and Hao, Tian and Rogers, Jeffrey and Chen, Ching-Hua and Kotz, David},
  title   = {Evaluating the Reproducibility of Physiological Stress Detection Models},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume  = {4},
  number  = {4},
  pages   = {1--29},
  year    = {2020},
  doi     = {10.1145/3432220}
}

@inproceedings{prajod2022generalizability,
  author    = {Prajod, Pooja and Andr{\'e}, Elisabeth},
  title     = {On the Generalizability of {ECG}-based Stress Detection Models},
  booktitle = {2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA)},
  pages     = {549--554},
  year      = {2022},
  doi       = {10.1109/ICMLA55696.2022.00090}
}

@inproceedings{prajod2024stressor,
  author    = {Prajod, Pooja and Mahesh, Bhargavi and Andr{\'e}, Elisabeth},
  title     = {Stressor Type Matters! --- Exploring Factors Influencing Cross-Dataset Generalizability of Physiological Stress Detection},
  booktitle = {Proceedings of the 26th International Conference on Multimodal Interaction (ICMI '24)},
  pages     = {508--517},
  publisher = {ACM},
  year      = {2024},
  doi       = {10.1145/3678957.3685738}
}

@article{vos2023generalizable,
  author  = {Vos, Gideon and Trinh, Kelly and Sarnyai, Zoltan and Rahimi Azghadi, Mostafa},
  title   = {Generalizable machine learning for stress monitoring from wearable devices: A systematic literature review},
  journal = {International Journal of Medical Informatics},
  volume  = {173},
  pages   = {105026},
  year    = {2023},
  doi     = {10.1016/j.ijmedinf.2023.105026}
}

@article{vos2023ensemble,
  author  = {Vos, Gideon and Trinh, Kelly and Sarnyai, Zoltan and Rahimi Azghadi, Mostafa},
  title   = {Ensemble machine learning model trained on a new synthesized dataset generalizes well for stress prediction using wearable devices},
  journal = {Journal of Biomedical Informatics},
  volume  = {148},
  pages   = {104556},
  year    = {2023},
  doi     = {10.1016/j.jbi.2023.104556}
}

@article{kiranyaz2021onedcnn,
  author  = {Kiranyaz, Serkan and Avci, Onur and Abdeljaber, Osama and Ince, Turker and Gabbouj, Moncef and Inman, Daniel J.},
  title   = {{1D} convolutional neural networks and applications: A survey},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {151},
  pages   = {107398},
  year    = {2021},
  doi     = {10.1016/j.ymssp.2020.107398}
}

@phdthesis{ismailfawaz2020thesis,
  author  = {Ismail Fawaz, Hassan},
  title   = {Deep Learning for Time Series Classification},
  school  = {Universit{\'e} de Haute-Alsace},
  year    = {2020},
  note    = {arXiv:2010.00567}
}

@article{nkurikiyeyezu2020person,
  author  = {Nkurikiyeyezu, Kizito and Yokokubo, Anna and Lopez, Guillaume},
  title   = {Effect of Person-specific Biometrics in Improving Generic Stress Predictive Models},
  journal = {Sensors and Materials},
  volume  = {32},
  number  = {2},
  pages   = {703--722},
  year    = {2020},
  doi     = {10.18494/SAM.2020.2650}
}
```
