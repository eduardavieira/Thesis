# Correções V4 → V5

> **Estado a 2026-09-27:** todos os pedidos da supervisora estão cumpridos e a tese
> passou à V6; as duas avaliações externas também foram tratadas. O estado atual e
> as decisões da autora estão no `SKILL.md`; este ficheiro é o histórico dos pedidos.

Origem: PDF anotado (15 anotações, Cláudia Brito) + `notasv4.md.rtf` (12 pontos
sobre resultados, erros de conteúdo, figuras, escrita).

**Nada do que a supervisora escreveu está errado.** Todos os 12 pontos foram
verificados contra o código e os dados. Dois desvios menores, irrelevantes para
as conclusões: o "atributo exatamente zero" está na Fig. 16 e não na 15; e
`nni_mean` fica em 5.º por magnitude no IG (3.º entre positivos), não em 4.º.

Números de apoio em `facts.md`. Localizações no código em `codebase.md`.

## ESTADO FINAL (2026-09-26) — todos os pedidos cumpridos

Validado contra o zip de 26/09 14:35. O resto deste ficheiro é o histórico do
plano; ler esta tabela primeiro.

| Pedido da supervisora | Onde ficou resolvido |
|---|---|
| 1. Model D escolhido pela razão errada | Classificador passou a ser o Model G; justificação com testes e limites na 4.6 ("Choice of Model G") |
| 2. Falta baseline | Majority, LogReg, RF e regra "só sei o dataset" em todas as tabelas (4.1.1, 4.2) |
| 3. Features tratadas como sequência | 3.5 explica o reshape; Models E/F/G; o G é a sugestão dela (série NN + resumo) |
| 4. Métodos de XAI não concordam | 4.4 "Disagreement Between the Two Methods" explica porquê |
| 5. Exemplos contrariam a fisiologia | ETL elimina a janela espúria; 4.4 mostra o modelo inteiro fisiológico e as atribuições por feature condicionais |
| 6. Filtro de artefactos | 3.4: interpolação, 300–2000 ms, critério relativo, exclusão >20%, contagens por classe |
| 7. Fidelidade mal interpretada | 4.4 "Quantitative Evaluation": ablação por entrada como teste principal |
| 8. Subject-independent nunca avaliado | GroupKFold e LOSO (3.6, 4.2) |
| 9. Contribuição 1 promete demais | "não degrada", p=0,662, 5 seeds (4.3, Intro) |
| 10. AUC, PR-AUC, variância, ICs | ROC-AUC e PR-AUC nas tabelas do WESAD; 5 seeds; testes emparelhados + Holm; apêndice `ap:confidence_intervals` |
| 11. 37 sujeitos | 3.3 (com a razão dos 22/25 do SWELL), Abstract, Resumo |
| 12. Datasets juntos sem controlo | 4.1 e 4.1.1 (controlo negativo) |
| 3.6.1 definições | FP/FN, especificidade, F-score (média harmónica) corrigidos; FN duplicado removido |
| Tabelas 1 e 2, gralhas do Cap. 2, conjunto de validação | Corrigidos |
| Atenção vs gradientes | Nenhuma afirmação de explicabilidade por atenção; só figuras do G |
| Figuras 14, 15, 4.4.2 | Substituídas pelas do G; erros com figura e números |
| Citações \citet vs \citep | 0 `\cite` e 0 `\citet` com nome duplicado |
| Uniformizar nome do modelo, recall/sensitivity | `\sysname` = sistema, Model G = classificador; só "sensitivity" |
| Cortes e contrações | Feitos |
| Anotações do PDF (capa, Doutora, promising, hifenização, keywords, Resumo, Part I, 4.2, Cap. 6) | Todas feitas |

Resta apenas a gralha "valuation metrics" (3.6.1) e a reescrita do Cap. 4 na voz
da autora.


---

## P0 — Bloqueadores de validade

### P0.1 Corrigir o filtro de artefactos e re-extrair as features
Ponto 6 dela. A razão é mais grave do que ela pensa: o filtro não está mal
concebido, **está partido** (`RPeakDetection_HRV_RemoteMachine.py:174`). 24% das
janelas violam o filtro declarado. Interpolar em vez de remover, como ela pede.
Contar remoções por classe. **Fazer isto antes de tudo o resto** — invalida
qualquer resultado a jusante.

**Estado: script escrito e validado**, falta correr por inteiro. Ver
`etl-v5.md`. O ETL corrigido é `Data Processing/RPeakDetection_HRV_fixed.py`
(o original nem compila). Fica por decidir o limiar `--max-bad`, que é uma
escolha metodológica a justificar na tese.

### P0.2 Adicionar baselines
Ponto 2. Ela escreveu "é fácil de resolver". É fácil de correr; o resultado é mau:
RF 64,55% > Model D 60,68%. Ver "Discussão (a)" no fim deste ficheiro.

### P0.3 Avaliação subject-independent
Ponto 8. LOSO dá 51,13%, GroupKFold 53,37% — abaixo do majority classifier.
Enquanto não estiver na tese, cada "subject-independent" no texto é falso.
Agrupar por `dataset + sujeito` (os prefixos `S#` colidem entre datasets).

### P0.4 Múltiplas seeds, AUC e intervalos de confiança
Pontos 9 e 10. A Contribuição 1 assenta em 0,68 pp = 9 amostras em 1320. Não
sobrevive a uma defesa. Repetir com ≥5 seeds e reportar média ± SD, ROC-AUC e PR-AUC.

---

## P1 — Afirmações a corrigir (não exigem re-correr nada)

### P1.1 Atenção vs gradientes
Transversal. Nunca foi extraído um peso de atenção — o próprio código admite
(`Explainability.py:40`). Corrigir: Secções 4.2, 4.4, 5.2; título da Fig. 15;
eixos das Figs. 15/16/17 ("Attention Score" → "Gradient Magnitude"); legenda da
Fig. 14; Abstract e Resumo. Nota: gradientes funcionam igual nos Modelos A, B e C,
por isso a explicabilidade **deixa de ser vantagem do Model D**.

### P1.2 Justificação do Model D
Ponto 1. O texto diz que o D tem o melhor recall; a Tabela 7 diz B = 77,90% e
D = 76,95%. Agrava-se porque a segunda justificação (explicabilidade via atenção)
também cai com P1.1. Opções: mudar a justificação para exatidão + admitir que a
diferença está dentro do ruído (exige P0.4), ou mudar de modelo.

### P1.3 Fidelidade mal interpretada
Ponto 7. 2,20 pp de queda sobre uma margem de 4,47 pp = **49% do que o modelo
consegue**. É fidelidade alta, não moderada. Reescrever a Secção 4.4.3 e o
parágrafo da redundância na Discussão. O mecanismo (redundância) mantém-se e
agora tem evidência — ver P1.10.

### P1.4 Os dois métodos de XAI não concordam
Ponto 4. Confirmado na mesma amostra (Figs. 16 e 18). **Mas a discordância é
matematicamente esperada**: saliência é `|∂y/∂x|` sem sinal; IG é
`(x − baseline) × grad`, com sinal e proporcional à distância à baseline, que aqui
é a média do treino. A correção não é forçar concordância — é explicar o que cada
método mede. Transforma um ponto fraco numa subsecção decente.

### P1.5 Figura 14
Top-3 = 44,9%, não "the majority". Diferença 3.º–4.º = 0,009, corte arbitrário.
`nni_mean` e `hr_mean` têm r = −0,868 e recebem 1,000 vs 0,741.

### P1.6 São 37 sujeitos, não 40
Ponto 11. Corrigir no Abstract, Resumo e Introdução. Explicar por que só 22 dos
25 sujeitos do SWELL chegaram aos dados.

### P1.7 As 8 features tratadas como sequência
Ponto 3. Ela tem razão, mas "são na prática MLPs" não é exato — ver
"Discussão (b)". A sugestão dela (série NN como sequência + features resumo como
entrada auxiliar) é a correção arquitetural certa, mas é trabalho de semanas:
obriga a refazer o ETL para guardar as séries RR. Negociar âmbito.

### P1.8 Secção 4.4.2 sem figura nem números
Os ficheiros existem (`results/case_studies/false_{positive,negative}_ig.png`).
Ou incluir, ou cortar a subsecção.

### P1.9 Datasets juntos sem controlo
Ponto 12. ROC-AUC 0,819 a distinguir WESAD de SWELL só pelas features de entrada.
Proporções de stress: SWELL 60,5%, WESAD 33,0%. Reportar resultados por dataset.

### P1.10 Incluir a matriz de correlação entre features
Não foi pedido, mas **apoia** o argumento existente da tese sobre redundância
(`sdnn ~ rmssd` r = 0,978). Passa de especulação a evidência. Baixo custo, alto retorno.

---

## P2 — Erros de conteúdo (~2 h no total)

### Secção 3.6.1 (`Contribution.tex:235–275`)
- **FP** — "the number of *positive* cases that were incorrectly classified, in the
  case of the actual class being negative": contradiz-se na própria frase. É
  "negative cases incorrectly classified as positive".
- **FN** — mesmo problema, invertido.
- **Especificidade** (L255) — "ratio between negative cases correctly classified
  and the instances **classified as** negative" descreve NPV = TN/(TN+FN). A
  fórmula (correta) é TN/(TN+FP), sobre os negativos **reais**.
- **F-score** (L270) — "combining model **accuracy** and recall" → precisão e
  sensibilidade. Duas frases depois já está certo.
- A definição de precisão (L262) está correta — não mexer.

### Tabela 1 (`StateOfTheArt.tex:88–98`)
- `HR` em `ms` → `bpm`
- `NN50 count` em `ms` → contagem (adimensional)
- `7-8125 ms` → `7,8125 ms` (é 1/128 s)

### Tabela 2 (`StateOfTheArt.tex:114`)
- VLF `≤ 0,4 Hz` no bloco dos 5 min → `≤ 0,04 Hz`. A L123, do bloco 24 h, já está
  certa (0,003–0,04 Hz).

### Capítulo 2 — typos
`Seyle` → `Selye` (L14) · `homology` → `homeostasis` (L14, **e verificar a citação,
ela diz que está adulterada**) · `latter known` → `later known` (L9) ·
`occurs natural` → `occurs naturally` (L18) · `Trough` → `Through` (L135)

### Secção 3.6 — conjunto de validação
Fala em "train-validation-test split" mas só descreve treino e teste. Vem de
`Models.py:110`, `validation_split=0.2` sobre o treino. Uma frase resolve.

### Keywords
Comentadas no Abstract (`preamble/Abstract.tex:15`) e no Resumo (L37).

---

## P3 — Escrita e formatação

### P3.1 Citações — a correção de maior impacto
Diagnóstico dela: `\citet` onde devia ser `\citep`. Na realidade há **296 `\cite{}`
e zero `\citet`/`\citep`**; o efeito vem de `natbib` + `plainnat`, onde `\cite` se
comporta como `\citet`. Substituição global `\cite{` → `\citep{`, depois rever
manualmente os ~40 casos "Author et al. \cite{}" para `\citet{}` sem repetir o nome.

### P3.2 Título da capa
Cortado ("Deep Learning Learning"). Quebrar entre `\titleA`/`\titleB`
(`dissertation.tex:48`).

### P3.3 "Professora" → "Doutora"
Página 3. Anotação dela: *"sou apenas doutora neste caso :)"*.

### P3.4 Remover estrutura da proposta
`\part{...}` em `dissertation.tex:99,105,120` — ela diz "basta comentar".
Comentar `\input{chapters/PlannedSchedule}` (L110): o Capítulo 6 é da proposta.

### P3.5 Uniformizar
- Um só nome do modelo: usar `\sysname` (HRV30) em vez de alternar entre
  "Model D" / "CNN-Transformer" / "HRV30"
- *recall* **ou** *sensitivity*, não os dois. Estado: Applications 7/1,
  Contribution 4/9.

### P3.6 Cortar
- Parágrafo Google/Facebook/Apple/Amazon/IBM (`StateOfTheArt.tex:135`)
- Valorização das empresas de IA (L138)
- "Perhaps the most fascinating aspect of the brain is that it can learn" (L149)
- "presents a remarkable possibility to improve quality of life" (`Contribution.tex:11`)
- "represents a promising approach" (Abstract — anotação da p.9: *"uma vez que já
  existe, não podemos dizer que é promising"*)
- Contrações em 4.2: "it's a minor inconvenience", "why it thinks they're stressed"
  (`Applications.tex:48`)

### P3.7 Anotações soltas do PDF
- p.9 — hifenização errada: "mod- est", "gen- erating"
- p.10 — Resumo tem de refletir as alterações do Abstract

---

## Pontos a discutir com a supervisora

### (a) A baseline destrói mais do que ela previu — decisão de enquadramento
Ela escreveu que a baseline "é fácil de resolver". É fácil de correr, mas a Random
Forest faz 64,55% contra 60,68% do Model D, com melhor precisão e ROC-AUC 0,700.
Não há forma honesta de escrever "o deep learning justifica-se".

Duas saídas, e a escolha é dela:
1. **Reformular como resultado negativo** — "em HRV ultra-curto com 8 features
   resumo, o DL não supera métodos clássicos, e eis porquê". Defensável, honesto,
   alinhado com a literatura de surrogacy já citada (Castaldo, Pecchia). Custo:
   reescrever o Capítulo 5 e as Contribuições.
2. **Mudar a arquitetura** para o que ela sugere no ponto 3. Custo: refazer o ETL
   para guardar as séries RR, mais re-treino. Semanas, sem garantia de ganhar.

**Levantar isto antes de começar as fases 3+.** Determina o âmbito de tudo o resto.

### (b) Ponto 3: "são MLPs" não é exato
Um `Conv1D(kernel=3)` sobre 8 features impõe partilha de pesos entre features
adjacentes e um prior de localidade — `hr_mean`/`nni_mean`/`nni_min` partilham um
filtro só por estarem lado a lado no CSV. Não é um MLP; é *pior*, porque é um viés
indutivo arbitrário em vez de nenhum. Isto **reforça** o argumento dela.

Mas corrigir a inferência dela sobre a Secção 4.6: ela diz que o argumento "mudar
a arquitetura não alterou o resultado, logo o limite está na informação" fica
destruído. Fica — **por outra razão**. Não é que o teto não exista; é que os
modelos DL estavam *abaixo* do teto que uma random forest atinge. O teto é
≥ 64,5%, não ~61%. Muda a redação da Discussão.

### (c) Ponto 4: a discordância entre XAI é principiada
Ver P1.4. Explicar porque diferem, em vez de forçar concordância, é uma resposta
que valoriza na defesa. Idem para o "atributo exatamente zero", que é a
normalização min-max e não um bug.

### (d) Ponto 5, segunda metade: a Fig. 17 é um problema de rótulos
A janela tem HR 92,6 bpm e SDNN 34,6 ms (perfil de stress) e está rotulada
`neutral` no SWELL. O modelo classificou não-stress e **acertou face ao rótulo**.
Ela lê como falha do modelo; a leitura mais defensável é que o rótulo do SWELL
(condição experimental) não é o mesmo que o estado fisiológico do sujeito.
Dá uma limitação genuína e bem fundamentada para a Secção 4.6.1.

---

## Ordem de trabalho

| Fase | Trabalho | Esforço |
|---|---|---|
| 0 | Conversa sobre (a) — o âmbito depende disto | 1 reunião |
| 1 | P2 + P3 — independentes de tudo o resto | 1–2 dias |
| 2 | Corrigir o filtro RR + re-extrair features (P0.1) | 1 dia + máquina |
| 3 | Re-treinar os 4 modelos, multi-seed, com baselines (P0.2, P0.4) | 2–3 dias |
| 4 | LOSO/GroupKFold + resultados por dataset (P0.3, P1.9) | 1–2 dias |
| 5 | Re-gerar figuras XAI, reescrever 4.4 e 4.6 (P1.1, P1.3–P1.5, P1.8, P1.10) | 3–4 dias |
| 6 | Reescrever Abstract, Resumo, Contribuições e Cap. 5 com os números novos | 2–3 dias |

As fases 1 e 2 podem correr em paralelo.

---

# ESTADO ATUALIZADO (fim da sessão de 2026-09-16)

Esta secção **substitui** os estados indicados acima. Números em `facts.md`.

## Resolvidos — experiência feita, falta escrever

| # | Ponto | O que se mediu |
|---|---|---|
| 1 | Model D pela razão errada | Agrava-se. B ganha em recall por **7,88 pp** em V5. E o D não lidera em nada: C ganha exatidão e AUC; D é o **pior** dos quatro em subject-independent (53,74%, abaixo do majority) |
| 2 | Falta baseline | Majority 56,35 · LogReg 59,50 · RF 63,72. Mais a baseline nova **"só sei o dataset" 61,36%**, acima de três das quatro arquiteturas |
| 3 | Features como sequência | Confirmado e resolvido. Conv1D sobre a série NN: +4,48 pp e +0,071 AUC sobre o Model D. Decomposto: reshape +0,90, série real +3,09 |
| 4 | XAI não concorda | Confirmado, com a explicação matemática (saliência sem sinal vs IG com sinal e dependente da baseline) |
| 5 | Exemplos contra a fisiologia | Causa encontrada (bug do filtro) e corrigida. A amostra em causa foi descartada pelo ETL novo |
| 6 | Filtro de artefactos | Corrigido. 24% das janelas violavam o filtro declarado. Estatísticas por classe: 11,12% (stress) vs 7,27% (não-stress) |
| 8 | Subject-independent | Feito para tudo: A–D, clássicos, arquiteturas novas, em combinado, SWELL e WESAD |
| 9 | Contribuição 1 | **Não se sustenta.** F2−F1 = +0,41 ± 1,79 pp, p=0,639 no combinado; replicado no WESAD (+0,53 ± 2,51 pp, p=0,662, e AUC p=0,233) |
| 10 | AUC e variância | 5 seeds em tudo, com AUC e testes emparelhados. Recall com SD 6,18 pp e especificidade 7,62 pp |
| 11 | 37 sujeitos | Confirmado (15 + 22) |
| 12 | Datasets sem controlo | **Muito mais grave do que ela escreveu.** Rótulos fisiologicamente opostos, provado por transferência (AUC 0,18–0,44), desenho do SWELL e autorrelato dos participantes |

## Por refazer nos dados V5

**Nada.** O único item que aqui estava — ponto 7, fidelidade — foi refeito no
WESAD: fidelidade **0,1958** (era 0,022) e estabilidade **0,806** (era 0,878).
Ver `facts.md`, "Explicabilidade refeita no WESAD". Consequência: o argumento da
redundância na Secção 4.4.3 deixa de ser necessário e a subsecção reescreve-se
de raiz.

## Não começados

**P2** (Secção 3.6.1, Tabelas 1 e 2, typos do Cap. 2, conjunto de validação,
keywords) e **P3** (296 citações, título, "Professora", `\part`, Capítulo 6,
uniformização, cortes). ~1,5 dias, independentes de qualquer experiência.

**Atenção vs gradientes** — transversal a 4 secções e 5 figuras; depende das
figuras novas.

**Secção 4.4.2** — depende das figuras novas.

## Bloqueado numa decisão

Re-correr a explicabilidade exige saber **qual é o modelo da tese**. As Figuras
14–19, a Tabela 8 (matriz de confusão), os 63 ms de inferência e as duas
citações do agente são todos da V4, e a amostra dos estudos de caso já não
existe.

Recomendação: **explicar um modelo treinado no WESAD**. No conjunto combinado as
atribuições nunca vão confirmar a fisiologia, porque o SWELL tem os sinais
invertidos. No WESAD (`hr_mean` com d = 1,73) vão apontar na direção que a
Tabela 6 da tese prevê — o que resolve o ponto 5 de forma limpa, sem manipulação.

## Enquadramento sugerido para a tese

Três resultados, não um:

1. **WESAD** — a deteção funciona: AUC 0,915 subject-independent
2. **SWELL** — nenhum modelo passa do acaso, com causa identificada no desenho
   (condição neutra sempre no primeiro bloco) e confirmada pelo autorrelato
3. **Transferência** — os rótulos opõem-se: AUC 0,18–0,44

Contribuições novas, a substituir as atuais:

- Primeira avaliação subject-independent de HRV de modalidade única a 30 s
- Quantificação da inflação do split aleatório (3 a 10 pp, e **escala com o
  número de parâmetros**: 19K perde 2,67 pp, 219K perde 7,44 pp)
- Demonstração de que os rótulos do SWELL-KW não são um alvo válido para HRV,
  com três linhas de evidência independentes
- A representação sequencial (série NN) supera as features resumo, verificado
  dentro de um só dataset para excluir artefacto de origem
- Contaminação por artefactos nestes datasets: 24% das janelas, correlacionada
  com a classe

## O que NÃO se pode afirmar

- Superioridade em exatidão no conjunto combinado ou no SWELL
- Que a série NN bate as baselines em subject-independent (p=0,088 a 0,388)
- Que o SWELL "não tem gente stressada" — tem carga de trabalho elevada e
  valência negativa; o que não tem é ativação simpática
- Qualquer diferença abaixo de ~1 pp (ruído do TensorFlow)

---

# ESTADO após a explicabilidade refeita (2026-09-17)

Substitui os estados anteriores para estes pontos.

## Resolvidos nesta sessão

| Ponto | Estado |
|---|---|
| **5** — exemplos contrariam a fisiologia | ✅ **Resolvido pela causa.** Com o modelo treinado só no WESAD, a amostra de estudo de caso dá `hr_mean` +0,67 e `nni_mean` −0,83 — assinatura simpática correta, alinhada com a Tabela 6. Confiança 99,0% (era 85,0%). No conjunto combinado era impossível: o alvo misturava dois estados opostos |
| **7** — fidelidade mal interpretada | ✅ **Já não é preciso o argumento da redundância.** A fidelidade passa de 0,022 para **0,1958** (9× mais). A fidelidade baixa da V4 era sintoma de o modelo não ter sinal fisiológico coerente, não de redundância. Reescrever a Secção 4.4.3 de raiz |
| **1.1** — atenção vs gradientes (figuras) | ✅ Os 5 rótulos das figuras corrigidos para "Gradient Magnitude". Falta a **parte textual**: Secções 4.2, 4.4, 5.2, legendas, Abstract e Resumo |
| **Fig. 15** — atribuição exatamente zero | ✅ Corrigido. Era normalização min-max, não erro de cálculo. Nenhuma barra fica agora a zero (`sdnn` passa de 0,00 para 0,15) |
| **Tabela 8** — matriz de confusão | ✅ Refeita no WESAD: 107/15/17/50, N=189. **Especificidade 87,70%** contra 39,8% na V4 |

## Consequência para o texto

A limitação escrita na Secção 4.3 — *"o modelo dispara falso alarme em três de
cada cinco janelas não-stressadas"* — **desaparece**. Não é atenuação: a
especificidade passou de 39,8% para 87,70%.

O parágrafo da redundância na Discussão (Secção 4.6) também cai, pela mesma
razão do ponto 7. A matriz de correlação (`sdnn ~ rmssd` r = 0,978) continua
verdadeira, mas já não é necessária para explicar coisa nenhuma.

## Ainda por fazer em código

| Item | Nota |
|---|---|
| Demo do agente (Secção 4.5) | As duas citações referem as amostras 3 e 50 dos dados antigos. Correr o `stress_feedback_agent.py` com `saved_models/wesad_model.h5` e amostras novas |
| Nomes `attention_*.png` | Mantidos para não partir o LaTeX. Renomear é opcional e implica atualizar `chapters/Applications.tex` |
| Latência de inferência (63 ms) | Medida no Model D da V4. A arquitetura não mudou, mas convém confirmar |

## Números novos para o Capítulo 4

```
Tabela 8 (WESAD, N=189):   107 / 15 / 17 / 50
                           acc 83,07  prec 76,92  rec 74,63  spec 87,70  AUC 0,910

Figura 14 (top-3):         nni_mean 1,0000 · hr_mean 0,5222 · nni_min 0,4552
Fidelidade:                0,1958   (V4: 0,022)
Estabilidade:              0,8060   (V4: 0,878)
```

Na V4 os seis primeiros da Figura 14 ficavam todos entre 0,74 e 1,00 — ranking
achatado, top-3 a somar só 44,9%, e `hr_mean` em 6.º. Agora há hierarquia clara
e `hr_mean` sobe a 2.º, coerente com ser a feature de maior efeito no WESAD
(d = 1,73). **A crítica da supervisora à Figura 14 ("os três primeiros somam
45%, não a maioria") deixa de se aplicar** — mas confirmar na figura nova antes
de escrever.

---

# Ponto 8 — fechado com LOSO (2026-09-17)

Subject-independent corrido em **todas** as configurações: GroupKFold k=5 e LOSO,
no conjunto combinado, no SWELL e no WESAD, para clássicos, Modelos A–D e as
três arquiteturas novas.

WESAD, LOSO (15 folds): AUC 0,902–0,951 em todos os modelos, **LogReg com a mais
alta (0,951)**. Exatidão 74,8–83,5 com SD 15–21. Ver `facts.md`.

**Consequência para o ponto 1 (escolha do Model D):** com LOSO no WESAD nenhuma
arquitetura se distingue. A justificação do Model D não pode ser desempenho.
Opção honesta: apresentar a LogReg como referência e o Model D como o modelo
explicado, admitindo o empate — corrigindo a afirmação sobre atenção, já que os
gradientes funcionam em qualquer arquitetura.
