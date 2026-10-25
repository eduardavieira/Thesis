# Correções da orientadora à V6 (Cláudia Brito, 15/10/2026)

Origem: `~/Downloads/V6_Tese_Eduarda_Vieira_comentarios_CB.pdf` (7 anotações: o comentário geral
e 6 marcações). Cada termo de pesquisa foi validado contra o zip V6 do repositório (commit
`ac34f3d`): aparece uma única vez. Os itens estão pela ordem da tese.

Como aplicar no Overleaf: abre o ficheiro indicado, Ctrl+F com o texto de **Procura**, seleciona
desde o início desse texto até ao fim do texto de **Seleciona até** (quando existe; se não houver,
seleciona só o texto procurado) e cola o bloco **Substitui por**.

| # | Comentário dela | Item |
|---|---|---|
| 1 | Abstract passa para a 2.ª página; o Resumo acompanha | A1, A2 |
| 2 | Declaração de IA: confirmar se falta a nota no texto (falta) | B1 |
| 3 | Citação do Task Force mal formatada | C1 |
| 4 | As CNN não têm atenção | C2 |
| 5 | A frase da lacuna diz mais do que o §2.4.6 | C3 |
| 6 | Contribuição 3 assenta só na partição aleatória (+ Abstract e §5.1) | C4, A1, A2, E1 |
| 7 | §4.7 é partição aleatória: dizê-lo, juntar às limitações, enfraquece os dois erros | D1, D2, D3 |

---

## A1. Abstract a caber numa página

**Ficheiro:** `preamble/Abstract.tex`

Hoje o Abstract ocupa 36 linhas para 25 disponíveis; os três cortes que ela sugere poupam só
cerca de 3. Por isso o bloco abaixo substitui os quatro parágrafos inteiros (as Keywords ficam
como estão). Medido pela largura das linhas do PDF dela: cerca de 21 linhas de texto + Keywords,
ou seja, 24 de 25. Não consegui compilar aqui; se ainda passar para a página seguinte, corta
"(907 windows, 15 subjects)" ou a frase "Removing frequency-domain features...".

O que mudou além dos cortes dela: a frase da sobreposição das partições foi substituída pela
formulação nova da Contribuição 3; saíram os parâmetros e os 49 ms, o controlo negativo do
conjunto combinado, o "standardised laboratory stressor" e o "uses the order of the beats" (vem da
partição aleatória, ver o comentário 7; se houver espaço, pode voltar).

**Procura:** `Mental stress is a growing public health concern, associated with increased risk`
**Seleciona até:** `address validation on further datasets with physiologically grounded labels and on-device deployment.`
**Substitui por:**
```latex
Mental stress is a growing public health concern, and wearables already record cardiac signals in everyday life. This dissertation presents \sysname, a system for binary stress detection from Heart Rate Variability (HRV) using deep learning on 30-second ECG windows. Two datasets, SWELL-KW and WESAD, were processed into 5,409 windows from 37 subjects, and ten configurations, from a logistic regression to seven neural networks, were compared under random and subject-wise partitioning.

The two datasets carry physiologically opposite stress labels: every HRV feature changes in opposite directions, and models transferred between them reach an AUC of 0.17 to 0.44. On WESAD (907 windows, 15 subjects), every trained model reaches an AUC between 0.88 and 0.92 on unseen subjects (GroupKFold), and the mean heart rate alone, 0.89. The selected model, a convolutional network reading the inter-beat interval series and eight time-domain features, reaches an AUC of 0.919 and 80.14\% accuracy on unseen subjects, and 0.955 and 89.78\% under a random split, which places windows of the same subject in training and test. Under that split the models that read the interval series outperform those built on summary features, but they lose the most on unseen subjects, where no difference between models is measurable. Accuracy varies widely between subjects, but the ranking of their windows remains good: choosing the threshold of each subject on about 15 minutes of recording raises the accuracy of the selected model on the rest of the session from 82.1\% to 91.1\%.

Removing frequency-domain features produced no detectable change in performance. Gradient-based explanations show that the model responds to a faster heart rate in the expected direction, but that per-feature attributions cannot be read one by one when the inputs are redundant. A proof-of-concept agent turns the classification into suggestions with a large language model. Future work should test the calibration across sessions and on other datasets.
```

## A2. Resumo (acompanha o Abstract)

**Ficheiro:** `preamble/Abstract.tex`

Mesma medição: cerca de 21 linhas + Palavras-chave, 24 de 25. Se passar, corta "(907 janelas,
15 sujeitos)".

**Procura:** `O stress mental é uma preocupação crescente de saúde pública`
**Seleciona até:** `abordar a validação noutros \textit{datasets} com rótulos fundamentados na fisiologia e a implementação no dispositivo.`
**Substitui por:**
```latex
O stress é um problema crescente de saúde pública, e os \textit{wearables} já registam sinais cardíacos no dia a dia. Apresenta-se o \sysname, um sistema de deteção binária de stress a partir da Variabilidade da Frequência Cardíaca (VFC) com \textit{deep learning}, em janelas de ECG de 30 segundos. Dos \textit{datasets} SWELL-KW e WESAD resultaram 5409 janelas de 37 sujeitos, e compararam-se dez configurações, da regressão logística a sete redes neuronais, com partição aleatória e por sujeito.

Os dois \textit{datasets} têm rótulos de stress fisiologicamente opostos: todas as características de VFC variam em sentidos contrários, e os modelos transferidos entre eles atingem AUC de 0,17 a 0,44. No WESAD (907 janelas, 15 sujeitos), todos os modelos treinados têm AUC entre 0,88 e 0,92 em sujeitos não vistos (GroupKFold), e a frequência cardíaca média, sozinha, 0,89. O modelo selecionado, uma rede convolucional que lê a série de intervalos entre batimentos e oito características temporais, atinge AUC de 0,919 e 80,14\% de exatidão em sujeitos não vistos, e 0,955 e 89,78\% com partição aleatória, que põe janelas do mesmo sujeito no treino e no teste. Nesta partição, os modelos que leem a série superam os de características resumo, mas são os que mais perdem em sujeitos não vistos, onde não há diferença mensurável entre modelos. A exatidão varia muito entre sujeitos, mas a ordenação das suas janelas mantém-se boa: escolher o limiar de cada sujeito em cerca de 15 minutos de gravação sobe a exatidão do modelo selecionado no resto da sessão de 82,1\% para 91,1\%.

Retirar as características do domínio da frequência não alterou o desempenho de forma detetável. As explicações por gradientes mostram que o modelo reage a uma frequência cardíaca mais alta no sentido esperado, mas que, com entradas redundantes, as atribuições por característica não se podem ler uma a uma. Como prova de conceito, um modelo de linguagem converte a classificação em sugestões. O trabalho futuro deve testar a calibração entre sessões e noutros \textit{datasets}.
```

---

## B1. Nota sobre o uso de IA (falta no texto)

**Ficheiro:** `chapters/Contribution.tex` (fim da secção "Setup Environment")

A declaração (`preamble/AIDeclaration.tex`) diz que o uso de IA está "duly noted in the work",
e o template de 2026 da Escola de Engenharia diz "devidamente assinalada/descrita no trabalho".
No texto só aparece o Claude como componente do agente; não há nenhuma nota sobre o uso de IA
na elaboração da tese. O parágrafo abaixo descreve o que se vê no repositório (13 commits com
o Claude como coautor: scripts, verificação de números, propostas de redação). **Só tu sabes o
que usaste:** confirma, tira o que não se aplica e acrescenta outras ferramentas (tradução,
correção de inglês, outro chatbot) se as usaste.

**Procura:** `the complexity of the models trained were within the capacity of a single machine.`
**Substitui por:**
```latex
the complexity of the models trained were within the capacity of a single machine.

Generative AI tools were also used in the preparation of this work. Claude (Anthropic), through the Claude Code assistant, was used to help write and debug the analysis scripts, to check the numbers reported in the text against the result files, and to suggest revisions to the wording of the text. All code, results and text produced with this support were reviewed and verified by the author, who takes full responsibility for them. This use is distinct from the language model inside the feedback agent, which is a component of the system (Section \ref{sec:agent_method}).
```

---

## C1. Citação do Task Force (Cap. 1, 1.º parágrafo, e 6 vezes no Cap. 2)

**Ficheiro:** `dissertation.bib`

Basta corrigir o `.bib`; as citações no texto não mudam. As chavetas duplas impedem o BibTeX
de ler "Task Force" como nome próprio e os "and" como separadores de autores.

**Procura:** `author = {Task Force of the European Society of Cardiology the North American Society of Pacing Electrophysiology},`
**Substitui por:**
```latex
author = {{Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology}},
```

## C2. As CNN não têm mecanismos de atenção (Cap. 1, início)

**Ficheiro:** `chapters/Introduction.tex`

As citações a seguir ficam como estão.

**Procura:** `These models are considerably more complex than commonly used architectures, as they incorporate attention mechanisms that enhance their ability to capture and analyze complex patterns within the data`
**Substitui por:**
```latex
Transformer-based models, which incorporate attention mechanisms, are considerably more complex than commonly used architectures, and attention enhances their ability to capture and analyse complex patterns within the data
```

## C3. A frase da lacuna, alinhada com o §2.4.6 (Cap. 1, início)

**Ficheiro:** `chapters/Introduction.tex`

Texto proposto por ela. A frase seguinte ("Whether stress can be detected reliably under these
conditions...") continua a ler-se bem.

**Procura:** `The thirty-second windows, a single modality, and a subject-independent model configuration required by a consumer wearable application have not been evaluated in the work reviewed in Chapter \ref{cha:estado_de_arte}.`
**Substitui por:**
```latex
No study reviewed in Chapter \ref{cha:estado_de_arte} evaluates deep learning on time-domain \gls{HRV} alone, computed over thirty-second windows, on subjects the model has not seen.
```

## C4. Contribuição 3 (Secção 1.3)

**Ficheiro:** `chapters/Introduction.tex`

Texto proposto por ela, com `\gls{NN}`.

**Procura:** `\item Evidence that, under a random split, the input representation matters more than the architecture.`
**Seleciona até:** `Under subject-wise partitioning, no difference between models is measurable.`
**Substitui por:**
```latex
\item Evidence that the advantage of reading the \gls{NN} interval series depends on the evaluation protocol. Under a random split, the models that read the series exceed those built on eight summary features, including a CNN-Transformer with more than six times as many parameters; they also lose the most when the partition is made by subject, and on unseen subjects no difference between models is measurable.
```

---

## D1. §4.7: dizer que é a partição aleatória

**Ficheiro:** `chapters/Applications.tex` (início de "Explainability")

**Procura:** `The analysis was applied to Model G (Dual-Input CNN) trained on WESAD, on the partition reported in Table \ref{tab:confusion_wesad}.`
**Substitui por:**
```latex
The analysis was applied to Model G (Dual-Input CNN) trained on WESAD, on the partition reported in Table \ref{tab:confusion_wesad}. This is a random split, in which windows of the same subject appear in the training and test partitions, so what the analysis shows, including the use of the order of the beats (Section \ref{sec:branch_ablation}), may partly reflect patterns specific to each subject; it was not repeated on a model evaluated on unseen subjects (Section \ref{sec:limitations}).
```

## D2. §4.7.3: os dois erros

**Ficheiro:** `chapters/Applications.tex` (fim do parágrafo "Read together, the two errors...")

**Procura:** `since what separates them is each subject's own baseline, which is not an input of the model.`
**Substitui por:**
```latex
since what separates them is each subject's own baseline, which is not an input of the model. The partition, however, is random, and the model was trained on other windows of S17 and S2; these two cases therefore do not show how the model behaves on subjects it has not seen, and the argument rests on the subject-wise results of Section \ref{sec:discussion_threshold}.
```

## D3. §4.8.4: juntar às limitações

**Ficheiro:** `chapters/Applications.tex` (subsecção "Limitations", parágrafo da explicabilidade)

**Procura:** `The explainability analysis was carried out on a single model trained on a single partition, and its per-window rankings are only moderately stable.`
**Substitui por:**
```latex
The explainability analysis was carried out on a single model trained on a single random partition, in which windows of the same subject appear in training and test, and its per-window rankings are only moderately stable. What it shows, including the use of the order of the beats, may partly reflect patterns specific to each subject, and it was not repeated on a model evaluated on unseen subjects.
```

---

## E1. Última frase do §5.1

**Ficheiro:** `chapters/ConclusionsAndFutureWork.tex` (último parágrafo de "Conclusions")

**Procura:** `that, under a random split, reading the interval series is better than summarising it,`
**Substitui por:**
```latex
that the advantage of reading the interval series over summarising it depends on the evaluation protocol, appearing under a random split but not on unseen subjects,
```

---

## Antes de submeter (não são alterações ao texto)

- **Anexo 5 = declaração de depósito no RepositóriUM** (Despacho RT-31/2019, Anexo 5A). Preenches
  nome, email, telemóvel, n.º do Cartão de Cidadão, tipo de trabalho (Dissertação), título,
  orientadores, data de conclusão e mestrado, escolhes o regime de acesso (1: acesso mundial
  imediato; 2: só na UMinho durante 1, 2 ou 3 anos; 3: outro), e **datas e assinas**. Os regimes
  2 e 3 exigem um requerimento ao Reitor. O Anexo 5B (formulário de depósito legal) tem campos que
  só existem depois das provas (TID do RENATES, classificação).
- **Assinaturas na própria tese:** as páginas "Statement of Integrity" e "Declaration on the use
  of Artificial Intelligence (AI) tools" têm espaço para a assinatura por cima do teu nome. No
  Anexo 5 só aparece a tua assinatura; não encontrei nenhum documento que os orientadores tenham
  de assinar.
- **Confirmar na secretaria ou no portal académico** se o Despacho RT-38/2026 (o da declaração de
  IA) mudou a numeração dos anexos, e como se entregam os documentos assinados (assinatura
  digital com o Cartão de Cidadão ou digitalizada).
- **Palavras-chave:** o template de 2026 da Escola de Engenharia pede 3 a 5, por ordem
  alfabética, e o resumo e as palavras-chave numa página. A tese tem 6 e não estão por ordem
  alfabética. Confirma se isto se aplica ao teu mestrado; se cortares uma, também ganhas espaço.
