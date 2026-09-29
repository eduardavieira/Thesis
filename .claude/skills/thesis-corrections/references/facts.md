# Factos validados

> ## AVISO DE LEITURA — ler de baixo para cima
>
> Este ficheiro está em **ordem cronológica de descoberta**, e as secções finais
> **corrigem** conclusões das iniciais. Não citar uma secção sem verificar se foi
> superada.
>
> As três correções que mais importam:
>
> 1. **Todos os resultados do conjunto combinado estão confundidos.** Os dois
>    datasets têm rótulos opostos. Qualquer número sobre "WESAD + SWELL" mede a
>    média entre um problema resolúvel e um mal colocado. Ver "A DESCOBERTA
>    CENTRAL" e "Transferência entre datasets".
> 2. **"A série NN é a única representação a discriminar acima do acaso em
>    sujeitos novos" (secção do subject-independent) está ERRADO.** A regressão
>    logística também passava (AUC 0,567, p=0,032), e o valor da série (0,607)
>    estava inflacionado por separação de datasets — dentro do SWELL cai para
>    0,566 e deixa de ser significativo. Ver "Ponto 12".
> 3. **Nenhuma diferença abaixo de ~1 pp é interpretável.** O TensorFlow em CPU
>    não é determinista: a mesma configuração e as mesmas seeds deram 61,18% e
>    62,08% em duas invocações — **0,90 pp de ruído puro**, maior que a
>    Contribuição 1 da tese (+0,68 pp). Isto invalida como afirmações fortes: a
>    Contribuição 1 (F2 vs F1, +0,41 pp), a escolha do Model D por exatidão em V4
>    (+0,68 pp) e Model C vs Model D em V5 (+0,77 pp, p=0,241).
>
> **Estado: os dados V5 já existem.** `/home/eduarda/Desktop/ProcessedECGData_v5.csv`
> (5430 janelas, 56,37% stress, 37 sujeitos), gerado em 2026-09-06 com o ETL
> corrigido. Ver `etl-v5.md` e a secção "Resultados na V5" no fim deste ficheiro.
> As secções abaixo referem-se aos dados **V4** salvo indicação em contrário —
> são o que sustenta as críticas à tese como está escrita.

Todos os números abaixo foram calculados a partir de
`/home/eduarda/Desktop/ProcessedECGData.csv` em 2026-09-06, com o mesmo split da
tese (`test_size=0.2`, `random_state=42`, `stratify=y`). Re-produzir com
`scripts/validate.py`.

**Onde estes números contradizem a tese, é a tese que está errada.**

## Dataset

| | |
|---|---|
| Janelas válidas (após `dropna`) | 6596 |
| Treino / teste | 5276 / 1320 |
| Fração de stress (total) | 56,25% |
| Fração de stress (teste) | **56,21%** (742 stress, 578 não-stress) |
| Sujeitos | **37** (15 WESAD + 22 SWELL) — a tese diz 40 |
| Ficheiros únicos | 95 |

Padrões de nome que identificam o dataset:
`S#_baseline_ECG_values.csv` e `S#_stress_ECG_values.csv` → WESAD (15 sujeitos);
`S#_neutral.csv`, `S#_stress_I.csv`, `S#_stress_T.csv` → SWELL (22 sujeitos).
Os prefixos `S1..S25` **colidem** entre datasets — agrupar por sujeito exige
prefixar com o nome do dataset, senão junta pessoas diferentes.

A tese diz que o SWELL tem 25 pessoas mas só 22 chegaram à CSV. Falta explicar porquê.

## Baselines (split da tese, seed 42)

| Modelo | Acc | Prec | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Majority ("sempre stress") | 56,21 | 56,21 | 100,00 | 0,720 | 0,500 | 0,562 |
| Regressão logística | 58,41 | 60,30 | 76,15 | 0,673 | 0,599 | 0,645 |
| LogReg (balanced) | 55,76 | 63,57 | 49,87 | 0,559 | 0,598 | 0,645 |
| **Random Forest (200)** | **64,55** | **66,96** | 72,91 | **0,698** | **0,700** | **0,755** |
| Model D (tese) | 60,68 | 62,13 | 76,95 | 0,690 | n/a | n/a |

**A Random Forest bate o Model D por 3,87 pontos**, com melhor precisão e F1
comparável. A regressão logística simples atinge praticamente o mesmo recall do
Model D (76,15% vs 76,95%) com 2 pontos menos de exatidão.

Margem do Model D sobre o majority classifier: **4,47 pontos**. É este o número
que interessa para ler tudo o resto.

## Subject-independent

| Protocolo | Modelo | Acc | Recall |
|---|---|---|---|
| GroupKFold k=5 | LogReg | 53,57% (SD 6,04) | 48,04% |
| GroupKFold k=5 | RandomForest | 53,37% (SD 2,59) | 63,32% |
| LOSO (37 sujeitos) | LogReg | 53,33% (SD 21,83) | — |
| LOSO (37 sujeitos) | RandomForest | 51,13% (SD 14,37) | — |

**Ambos abaixo do majority classifier (56,21%).** O desvio-padrão entre sujeitos
em LOSO (14–22 pontos) mostra que o desempenho médio não tem significado prático.

Corolário: a exatidão de ~60% reportada na tese depende de janelas do mesmo
participante estarem em treino e em teste.

## Contaminação por artefactos

Se o filtro de 500–1200 ms tivesse funcionado, teríamos `nni_min ≥ 500`,
`nni_max ≤ 1200` e `hr_mean ∈ [50, 120]`. Contagens reais:

| Violação | Janelas | % |
|---|---|---|
| `nni_min < 500 ms` | 904 | 13,7% |
| `nni_max > 1200 ms` | 1113 | 16,9% |
| `hr_mean > 120 bpm` | 101 | 1,5% |
| `hr_mean < 50 bpm` | 57 | 0,9% |
| **Qualquer violação** | **1581** | **24,0%** |
| `sdnn > 200 ms` (implausível a 30 s) | 678 | 10,3% |

Gamas observadas: `nni_min` [14, 1782] ms · `nni_max` [431, 5465] ms ·
`hr_mean` [31,4, 258,7] bpm. Valores fisiologicamente impossíveis.

**A contaminação não é uniforme entre classes:**

| | Taxa de contaminação |
|---|---|
| Classe stress | 27,5% |
| Classe não-stress | 19,4% |
| `P(stress \| artefacto)` | 64,6% |
| `P(stress \| sem artefacto)` | 53,6% |
| Base rate | 56,2% |

Há sinal artefactual correlacionado com a label que os modelos podem estar a
explorar. Sozinho não explica o resultado (prever stress ⟺ artefacto dá 50,77%),
mas contamina qualquer afirmação sobre aprendizagem de padrões fisiológicos.

## As amostras dos estudos de caso

**Figuras 16 e 18 + amostra 3 da demo do agente** — `S15_stress_T.csv` (SWELL, label=1):

| Feature | Valor | Mediana da população |
|---|---|---|
| `hr_mean` | 83,56 | 75,38 |
| `nni_mean` | 931,90 | 809,75 |
| `nni_min` | **264,00** ← 227 bpm, impossível | 671,00 |
| `nni_max` | **1939,00** ← 31 bpm | 987,00 |
| `sdnn` | **460,76** | 71,65 |
| `rmssd` | **494,11** | 58,01 |
| `nn50` | 18,00 | 10,00 |
| `pnn50` | **94,74%** | 30,56 |

`nni_min` e `nni_max` estão ambos **fora** do filtro declarado — deviam ter sido
removidos. `pnn50` de 94,7% significa que 95% dos intervalos consecutivos diferem
mais de 50 ms: isso não é um ritmo cardíaco, é ruído.

A hipótese da supervisora ("parece um intervalo espúrio, não um estado
fisiológico") está **confirmada**, e o mecanismo é o bug do filtro.

**Figuras 17 e 19 + amostra 50 da demo** — `S24_neutral.csv` (SWELL, label=0):

`hr_mean` 92,58 · `nni_min` 558 · `nni_max` 731 · `sdnn` 34,64 · `rmssd` 20,77 · `pnn50` 4,65

Janela perfeitamente limpa. Frequência elevada com variabilidade baixa —
**perfil clássico de stress, rotulada `neutral`**. O modelo classificou não-stress
e acertou face ao rótulo. Isto é um problema de rótulos (condição experimental ≠
estado fisiológico do sujeito), não do modelo — argumento útil para a Secção 4.6.1.

## Confundimento entre datasets

| | |
|---|---|
| ROC-AUC a distinguir WESAD de SWELL só pelas 8 features | **0,819** |
| Exatidão balanceada | 64,71% |
| Fração de stress no SWELL | 60,5% (n=5580) |
| Fração de stress no WESAD | 33,0% (n=1016) |

Os datasets são fortemente distinguíveis a partir das próprias features de entrada,
e têm proporções de classe muito diferentes. O modelo pode estar parcialmente a
separar datasets em vez de estados. Efeito limitado pelo WESAD ser só 15% dos dados,
mas obriga a reportar resultados por dataset.

## Redundância entre features — isto apoia a tese

Correlações |r| > 0,8:

```
sdnn    ~ rmssd     r = +0,978
nni_max ~ sdnn      r = +0,915
nni_max ~ rmssd     r = +0,906
nn50    ~ pnn50     r = +0,887
hr_mean ~ nni_mean  r = -0,868
hr_mean ~ nni_min   r = -0,814
```

O argumento da Secção 4.4.3 (baixa fidelidade explicada por redundância) **está
correto** e passa a ter evidência. Incluir esta matriz na tese. O que muda é só a
leitura da magnitude da fidelidade, não o mecanismo.

## Fidelidade — a leitura correta

| Leitura | Valor |
|---|---|
| Queda absoluta ao mascarar as 2 features de maior atribuição | 60,68% → 58,48% = **2,20 pp** |
| Margem do modelo acima do majority classifier | **4,47 pp** |
| Fração da margem destruída pela ablação | **49,2%** |

Mascarar duas features remove **metade** do que o modelo realmente consegue.
Isto é fidelidade **alta**, não moderada. A Secção 4.4.3 e o parágrafo da
redundância na Discussão precisam de ser reescritos com esta leitura.

## Figura 14 — importância global

| Feature | Valor |
|---|---|
| nni_mean | 1,000 |
| nni_min | 0,941 |
| nn50 | 0,910 |
| sdnn | 0,901 |
| nni_max | 0,766 |
| hr_mean | 0,741 |
| rmssd | 0,554 |
| pnn50 | 0,541 |

- Soma total: 6,354. Top-3 = 2,851 → **44,9%**. A legenda diz "the majority" — falso.
- 3.º (nn50 0,910) vs 4.º (sdnn 0,901): diferença de **0,009**. O corte "top 3" é arbitrário.
- `nni_mean` (1,000) e `hr_mean` (0,741) têm r = −0,868 entre si: são quase a mesma
  grandeza e recebem importâncias muito diferentes — argumento **contra** a leitura
  fisiológica que a tese faz.

## Figuras 16 vs 18 — a mesma amostra, rankings contraditórios

| Posição | Gradientes (Fig. 16) | Integrated Gradients (Fig. 18) |
|---|---|---|
| 1.º | nni_mean (1,000) | sdnn (−0,71) |
| 2.º | nn50 (0,35) | pnn50 (+0,65) |
| 3.º | pnn50 (0,33) | nni_max (+0,53) |
| 4.º | sdnn (0,27) | nn50 (−0,26) |
| 5.º | nni_max (0,20) | nni_mean (+0,24) |
| — | nni_min (0,00) | nni_min (−0,23) |

`nni_mean` é 1.º num método e 5.º no outro (3.º entre contribuições positivas).
A supervisora disse "quarto" — a substância do ponto mantém-se.

Na Fig. 16, os valores de entrada de `sdnn` (+2,95 SD), `rmssd` (+2,65),
`nni_max` (+3,2) e `pnn50` (+2,5) estão todos muito **acima** da média, quando a
Tabela 6 da tese diz que valores **baixos** indicam stress. Contradiz a fisiologia
declarada — e explica-se pelo artefacto documentado acima.

---

# Resultados na V5 (dados limpos)

`/home/eduarda/Desktop/ProcessedECGData_v5.csv` — 5430 janelas (V4: 6596, −17,7%),
56,37% stress (V4: 56,25%), 37 sujeitos, mínimo 60 janelas/sujeito, nenhum
sujeito com uma só classe. Sinal fisiologicamente plausível:
`nni_min` [303, 1167] ms · `nni_max` [405, 1887] ms · `hr_mean` [47, 153] bpm.

## Correções por classe (o número que a supervisora pediu)

| | não-stress | stress |
|---|---|---|
| Intervalos RR analisados | 101 316 | 131 695 |
| Intervalos corrigidos | 7445 (7,35%) | 10 565 (8,02%) |
| **Cairiam na gama antiga 500–1200 ms** | **7,27%** | **11,12%** |
| Janelas descartadas (>20% corrigido) | 442 | 717 |

A última linha valida o ponto 6 dela: o filtro original *estava* a apagar
proporcionalmente mais nas janelas de stress (11,12% vs 7,27%).

## Comparação nas mesmas 5 seeds (42–46), dados V5

| Modelo | Acc | Precisão | Recall | Especificidade | ROC-AUC |
|---|---|---|---|---|---|
| Majority | 56,35 ± 0,00 | 56,35 | 100,00 | 0,00 | 0,500 |
| LogReg | 59,50 ± 1,42 | 61,11 ± 0,94 | 77,35 ± 1,81 | 36,46 ± 1,52 | 0,615 ± 0,018 |
| **RandomForest** | **63,72 ± 0,63** | **65,78 ± 0,38** | 74,25 ± 2,00 | **50,13 ± 1,64** | **0,683 ± 0,004** |
| Model D | 61,64 ± 0,89 | 62,41 ± 1,20 | 80,72 ± 6,18 | 37,00 ± 7,62 | 0,651 ± 0,012 |

Model D em `retrain_v5.py` (5 seeds, mesmo pipeline do `main.py`).
V4 reportava acc 60,68 · prec 62,13 · rec 76,95 — **uma seed, dados sujos**.

### O que isto estabelece

1. **Limpar os dados ajudou pouco.** 60,68 → 61,64 (+0,96 pp). Margem sobre o
   majority: 4,47 → 5,29 pp. Melhoria real, mesma ordem de grandeza.
2. **A Random Forest continua a ganhar**, e por mais do que exatidão: +2,08 pp
   de exatidão, +13,1 pp de especificidade e AUC 0,683 vs 0,651. Perde só em
   recall (74,25 vs 80,72), e o Model D compra esse recall com especificidade.
   A decisão (a) de `corrections-v5.md` mantém-se por resolver.
3. **A variância entre seeds destrói as afirmações pontuais da tese.**
   Recall SD = 6,18 pp (71,57–87,09). Especificidade SD = 7,62 pp (27,22–48,10).
   A tese reporta "recall 76,95%" e "especificidade 39,8%" como propriedades do
   modelo; são um sorteio de uma distribuição com ±6–8 pp de largura. É a prova
   medida do ponto 10 da supervisora.
4. **O modelo desloca o limiar, não ganha discriminação.** Exatidão SD 0,89 pp
   contra recall SD 6,18 pp: entre corridas troca recall por especificidade
   quase sem mexer na exatidão nem na AUC (SD 0,012).
5. A RF é também **muito mais estável** (acc SD 0,63; AUC SD 0,004).

## LOSO na V5

GroupKFold k=5 por sujeito: LogReg 54,11% (SD 2,58) · RandomForest 53,09% (SD 2,34).
Continuam **abaixo do majority classifier (56,35%)**. Limpar os dados não
resolveu o problema subject-independent.

## Sujeitos degradados pela limpeza

Nenhum perdido, mas quatro ficaram quase degenerados numa classe:

| Sujeito | V4 → V5 | % stress na V5 |
|---|---|---|
| SWELL_S20 | 196 → 80 | 96,2% |
| SWELL_S25 | 203 → 92 | 13,0% |
| SWELL_S19 | 272 → 161 | 89,4% |
| SWELL_S3 | 250 → 152 | 86,2% |

`S24_stress_I.csv` perdeu as 52 janelas todas. Perda por dataset: WESAD −7,4%,
SWELL −19,6%; uniforme entre condições do SWELL (neutral −20,0%, stress_I −17,9%,
stress_T −20,9%), ou seja não enviesa a classe.

---

# Tabela 7 refeita (V5, 5 seeds por arquitetura)

`Classification Models/retrain_v5.py` → `results_v5.csv` (20 treinos).
Hiperparâmetros da Secção 3.6: clipnorm 0,5 / 100 épocas (A, B, C);
clipnorm 1,0 / 150 épocas (D). Baselines nas mesmas seeds e splits.

| Modelo | Acc | Precisão | Recall | Especif. | ROC-AUC | Params | min |
|---|---|---|---|---|---|---|---|
| Majority | 56,35 | 56,35 | 100,00 | 0,00 | 0,500 | — | — |
| LogReg | 59,50 ± 1,42 | 61,11 | 77,35 | 36,46 | 0,615 | — | — |
| **RandomForest** | **63,72 ± 0,63** | **65,78** | 74,25 | **50,13** | **0,683** | — | — |
| A (LSTM) | 60,42 ± 1,05 | 61,14 | 81,96 ± 5,30 | 32,62 | 0,631 | 19 009 | 0,77 |
| B (Deep LSTM) | 60,92 ± 0,72 | 60,29 | **89,84 ± 2,29** | 23,59 | 0,625 | 132 993 | 2,39 |
| **C (CNN-LSTM)** | **61,95 ± 0,79** | 61,77 | 85,42 ± 3,04 | 31,65 | **0,660** | 77 249 | 1,06 |
| D (CNN-Transf.) | 61,18 ± 0,62 | 61,78 | 81,96 ± 4,87 | 34,35 | 0,652 | 218 625 | 2,36 |

V4 reportava (1 seed, dados sujos): A 60,00 · B 60,00 · C 59,70 · D 60,68.

## Parâmetros — a tabela da V4 está errada

| | Tese | Medido |
|---|---|---|
| A | ~50K | **19 009** |
| B | ~100K | **132 993** |
| C | ~80K | 77 249 |
| D | ~218K | 218 625 |

A e B estão errados. Não tem nada a ver com o filtro — o número de parâmetros
depende só da arquitetura e do formato de entrada, ambos inalterados desde a V4.

## Comparações emparelhadas (mesmas seeds, t-test emparelhado)

```
C - D:  +0,77 +/- 1,26 pp   p=0,241     nao significativo
C - A:  +1,53 +/- 0,69 pp   p=0,008  *
C - B:  +1,03 +/- 0,59 pp   p=0,017  *
D - B:  +0,26 +/- 1,05 pp   p=0,612     nao significativo
D - A:  +0,76 +/- 1,27 pp   p=0,255     nao significativo

RF - A: +3,30 pp  p=0,0092 *      RF - A: AUC +0,052  p=0,0049 *
RF - B: +2,80 pp  p=0,0084 *      RF - B: AUC +0,058  p=0,0011 *
RF - C: +1,77 pp  p=0,0247 *      RF - C: AUC +0,023  p=0,0242 *
RF - D: +2,54 pp  p=0,0027 *      RF - D: AUC +0,031  p=0,0048 *
```

## O que isto estabelece

1. **O Model D não é o melhor em nada.** C ganha em exatidão e AUC; B ganha em
   recall por 7,9 pp. D é o mais caro (218K params, 2,36 min). A única métrica
   em que lidera entre os DL é a especificidade (34,35), e dentro de ±7,16 de SD.
2. **O ponto 1 da supervisora agrava-se com dados limpos.** A tese escolhe o D
   por "melhor recall"; em V4 o B ganhava por 0,95 pp, em V5 ganha por **7,88 pp**.
3. **A RandomForest bate as quatro arquiteturas com significância** em exatidão
   e em AUC (p < 0,05 em todos os 8 testes). Não é ruído.
4. **As diferenças entre arquiteturas DL são quase todas ruído.** Só C > A e
   C > B sobrevivem. C vs D, D vs A e D vs B não são distinguíveis.
   Isto *apoia* o argumento da Secção 4.6 de que a arquitetura não é o fator
   limitante — mas a conclusão certa não é "o limite é a informação", e sim
   "o limite é a representação de entrada", já que a RF, com os mesmos dados,
   vai significativamente mais longe.
5. **Todas as arquiteturas colapsam para o mesmo comportamento**: recall 82–90%,
   especificidade 24–34%. Estão mais perto do majority classifier (recall 100,
   especificidade 0) do que de um classificador equilibrado. A RF é a única com
   especificidade acima de 50%. O multiplicador de 1,2 nos pesos de classe
   empurra todos para o mesmo canto — a "vantagem em recall" do D não vem de
   discriminação melhor, vem da escolha do limiar.

---

# Arquitetura sequencial (ponto 3) — RESOLVIDO, resultado positivo

`Classification Models/train_sequence.py`. Entrada: série de intervalos NN
(`/home/eduarda/Desktop/NNSeries_v5.npz`, 5430 × 73, padding a zero,
comprimento real mediana 35) + as 8 features resumo como entrada auxiliar.
Ramo sequencial: Conv1D(64,5) ×2 + pooling médio mascarado. Mesmos splits,
mesmas 5 seeds, mesmos pesos de classe (incluindo o multiplicador de 1,2).

| Modelo | Acc | Especif. | ROC-AUC | Params |
|---|---|---|---|---|
| Majority | 56,35 | 0,00 | 0,500 | — |
| LogReg | 59,50 ± 1,42 | 36,46 | 0,615 | — |
| Model A (LSTM) | 60,42 ± 1,05 | 32,62 | 0,631 | 19 009 |
| Model B (Deep LSTM) | 60,92 ± 0,72 | 23,59 | 0,625 | 132 993 |
| Model D (CNN-Transf.) | 61,18 ± 0,62 | 34,35 | 0,652 | 218 625 |
| Model C (CNN-LSTM) | 61,95 ± 0,79 | 31,65 | 0,660 | 77 249 |
| MLP resumo (sem reshape) | 62,08 ± 0,28 | 41,56 | 0,669 | 7 393 |
| RandomForest | 63,72 ± 0,63 | 50,13 | 0,683 | — |
| Conv1D só série NN | 65,17 ± 1,81 | 49,70 | 0,708 | 27 969 |
| **Conv1D série + resumo** | **65,65 ± 1,31** | **62,28 ± 6,36** | **0,723 ± 0,016** | **32 929** |

## Testes emparelhados

```
serie+resumo vs RandomForest:  acc +1,93 pp p=0,0499 *   AUC +0,040 p=0,0073 *
serie+resumo vs Model D:       acc +4,48 pp p=0,0059 *   AUC +0,071 p=0,0042 *
serie+resumo vs Model C:       acc +3,70 pp p=0,0002 *   AUC +0,063 p=0,0001 *
serie+resumo vs MLP resumo:    acc +3,57 pp p=0,0042 *   AUC +0,054 p=0,0053 *
so serie NN vs MLP resumo:     acc +3,09 pp p=0,0286 *   AUC +0,039 p=0,0288 *
MLP resumo vs Model D:         acc +0,90 pp p=0,0295 *   AUC +0,017 p=0,0049 *
```

## Decomposição do ganho (o controlo `summary` isola as causas)

| Passo | Ganho acc | Ganho AUC |
|---|---|---|
| Tirar o reshape `(8,1)` (A–D → MLP resumo) | +0,90 pp * | +0,017 * |
| Dar a série NN real (MLP → só série) | +3,09 pp * | +0,039 * |
| Juntar série + resumo | +0,48 pp | +0,015 |
| **Total (Model D → série+resumo)** | **+4,48 pp** | **+0,071** |

O reshape estava mesmo a prejudicar, mas é o efeito menor. **O grosso do ganho
vem de dar ao modelo uma sequência com ordem temporal real.** A hipótese da
supervisora no ponto 3 está confirmada experimentalmente.

## O que isto muda na tese

1. **O deep learning passa a justificar-se**, e bate os clássicos com
   significância (p=0,0499 em exatidão, p=0,0073 em AUC contra a RandomForest).
2. **O argumento do "teto de informação" cai.** Os ~61% não eram o limite da
   HRV ultra-curta — eram o limite da representação de entrada. A informação
   estava na série NN e era deitada fora pelo resumo em 8 estatísticas.
3. **É o primeiro modelo que não colapsa** para "diz stress quase sempre":
   especificidade 62,28% contra 23–34% dos Modelos A–D.
4. **Com 6,6× menos parâmetros que o Model D** (32 929 vs 218 625).
5. A Secção 4.6 tem de ser reescrita: a parte "mudar a arquitetura não muda
   nada" mantém-se verdadeira *dentro* da representação errada, mas a
   conclusão que daí se tirava está errada.

## Nota de implementação

Um BiLSTM foi testado primeiro e abandonado: >10 min/seed em CPU contra
~1,5 min do Conv1D. O Conv1D é também a comparação mais limpa — mantém a
família de arquitetura dos Modelos C/D e muda só a entrada.
Conv1D não respeita máscaras do Keras: as posições de padding são zeradas
após a convolução e a média é feita sobre o número de posições reais.

---

# Subject-independent do modelo sequencial (P0.3) — o teste decisivo

GroupKFold k=5 agrupado por `dataset + sujeito`, 37 sujeitos, sobreposição
zero entre treino e teste. `train_sequence.py --cv group`.

| Modelo | Acc | Especif. | ROC-AUC |
|---|---|---|---|
| Majority | 56,36 ± 4,14 | 0,00 | 0,500 |
| LogReg | 54,11 ± 2,88 | 57,45 | 0,567 ± 0,046 |
| RandomForest | 53,80 ± 3,25 | 39,63 | 0,533 ± 0,072 |
| MLP resumo | 53,01 ± 3,33 | 38,38 | 0,545 ± 0,063 |
| **Conv1D só série NN** | **57,13 ± 4,79** | 54,10 | **0,607 ± 0,057** |
| Conv1D série + resumo | 57,03 ± 5,40 | 58,33 | 0,596 ± 0,067 |

## AUC é significativamente > 0,5?  (t-test 1 amostra, 5 folds)

```
MLP resumo      0,545  p=0,185     nao
RandomForest    0,533  p=0,367     nao
so serie NN     0,607  p=0,014  *  SIM
serie+resumo    0,596  p=0,033  *  SIM
```

**Só os modelos com a série NN discriminam acima do acaso quando avaliados
em sujeitos nunca vistos.** Nenhum modelo sobre as 8 features resumo o faz.

## Testes emparelhados por fold

```
serie+resumo vs MLP resumo:    acc +4,02 pp p=0,039 *   AUC +0,050 p=0,092
so serie NN  vs MLP resumo:    acc +4,12 pp p=0,014 *   AUC +0,061 p=0,057
serie+resumo vs RandomForest:  acc +3,23 pp p=0,264     AUC +0,063 p=0,118
serie+resumo vs Majority:      acc +0,67 pp p=0,863     AUC +0,096 p=0,033 *
```

## Queda do split aleatório para o split por sujeito

| Modelo | acc rnd → grp | AUC rnd → grp |
|---|---|---|
| série+resumo | 65,65 → 57,03 (−8,62) | 0,723 → 0,596 (−0,127) |
| só série NN | 65,17 → 57,13 (−8,04) | 0,708 → 0,607 (−0,101) |
| MLP resumo | 62,08 → 53,01 (−9,07) | 0,669 → 0,545 (−0,124) |
| RandomForest | 63,72 → 53,80 (−9,92) | 0,683 → 0,533 (−0,150) |

**Todos os modelos perdem 8–10 pp de exatidão.** O grosso do desempenho
publicado nestes datasets com split aleatório é memorização de sujeito, e
isto quantifica-o. A série NN perde menos (−0,101 de AUC contra −0,150 da RF)
e é a única que sobrevive acima do acaso.

## Leitura honesta

1. **Em exatidão, nada bate o majority classifier.** 57,03 vs 56,36 com SD 5,4
   e p=0,863. Não há aqui nenhuma afirmação de exatidão a fazer.
2. **Em AUC, a série NN discrimina mesmo** (0,596–0,607, p<0,05 contra 0,5),
   e os modelos sobre features resumo não. AUC não depende do limiar nem do
   balanço de classes, por isso é a métrica defensável.
3. **A vantagem da representação sequencial sobrevive** à avaliação honesta,
   embora reduzida: +4,0 a +4,1 pp de exatidão sobre o MLP resumo (p<0,05).
4. **Variância entre folds enorme**: exatidão de 52,0 a 64,7; AUC de 0,552 a
   0,711. É a variabilidade inter-individual da Secção 4.6 a manifestar-se —
   alguns sujeitos são fáceis, outros impossíveis.
5. Conclusão: o sistema **não é utilizável** como detetor subject-independent
   a 30 s, mas a série NN contém sinal generalizável que o resumo em 8
   estatísticas destrói. Aponta para calibração por sujeito como via.

---

# Ponto 9 — F1 vs F2 refeito: a Contribuição 1 NÃO se sustenta

Model D, 5 seeds, mesmos splits, dados V5 limpos.

| | Acc | Recall | Especif. | ROC-AUC |
|---|---|---|---|---|
| F2 (8 temporais) | 62,08 ± 1,18 | 75,07 | 45,32 | 0,663 ± 0,014 |
| F1 (10, c/ frequência) | 61,68 ± 0,96 | 82,12 | 35,27 | 0,649 ± 0,004 |

```
F2 - F1 exatidao: +0,405 +/- 1,790 pp   p=0,639   NAO SIGNIFICATIVO
F2 - F1 AUC     : +0,013 +/- 0,012      p=0,069   NAO SIGNIFICATIVO
```

A tese afirma F2 60,68% vs F1 60,00% = +0,68 pp (uma seed, dados sujos).
Medido: +0,41 pp com SD da diferença de 1,79 — **o intervalo cobre zero com
folga**. A direção do efeito mantém-se mas a magnitude é indistinguível de ruído.

**A Contribuição 1 tem de ser reescrita.** O que os dados suportam é
"remover as features de frequência não degrada o desempenho", não "melhora-o".
Isso continua a ser útil (justifica usar 8 em vez de 10, mais barato e alinhado
com a literatura de surrogacy), mas é uma afirmação muito mais fraca do que
"empirical validation that frequency-domain features are unreliable".

# Ponto 8 — Modelos A–D subject-independent (GroupKFold k=5)

| Modelo | Acc | Especif. | ROC-AUC | AUC > 0,5? |
|---|---|---|---|---|
| Majority | 56,36 ± 4,14 | 0,00 | 0,500 | — |
| Model A (LSTM) | 57,75 ± 3,45 | 31,45 | 0,568 ± 0,064 | p=0,077 ✗ |
| Model B (Deep LSTM) | 56,00 ± 2,52 | 19,84 | 0,553 ± 0,035 | p=0,028 * |
| Model C (CNN-LSTM) | 55,29 ± 1,02 | 31,90 | 0,542 ± 0,050 | p=0,137 ✗ |
| **Model D (CNN-Transf.)** | **53,74 ± 1,69** | 41,15 | 0,540 ± 0,059 | p=0,208 ✗ |
| Conv1D só série NN | 57,13 ± 4,79 | 54,10 | 0,607 ± 0,057 | p=0,014 * |
| Conv1D série+resumo | 57,03 ± 5,40 | 58,33 | 0,596 ± 0,067 | p=0,033 * |

## Queda do split aleatório para o split por sujeito

```
Model A: 60,42 -> 57,75  (-2,67 pp)
Model B: 60,92 -> 56,00  (-4,92 pp)
Model C: 61,95 -> 55,29  (-6,66 pp)
Model D: 61,18 -> 53,74  (-7,44 pp)   <- o modelo escolhido cai mais
serie+resumo: 65,65 -> 57,03 (-8,62 pp)
RandomForest: 63,72 -> 53,80 (-9,92 pp)
```

## Resultado inesperado: a ordem inverte-se

Em split aleatório: C (61,95) > D (61,18) > B (60,92) > A (60,42).
Por sujeito: **A (57,75) > B (56,00) > C (55,29) > D (53,74)**.

A ordem inverte-se quase exactamente. **Quanto mais parâmetros, mais o modelo
memoriza sujeitos e pior generaliza**: o Model A tem 19 009 parâmetros e cai
2,67 pp; o Model D tem 218 625 e cai 7,44 pp. O Model D, escolhido na tese,
é o **pior** dos quatro em avaliação honesta — e fica 2,6 pp abaixo do
majority classifier.

## Ressalva sobre a comparação com o modelo sequencial

Emparelhado por fold, a série NN tem AUC superior a todos os A–D (+0,039 a
+0,067) mas **nenhuma diferença atinge p<0,05** (p=0,105 a 0,307) — com 5 folds
e SD de 0,05–0,06 não há potência estatística. O que é defensável é o teste de
uma amostra: só a série NN e o Model B têm AUC significativamente > 0,5, e o
Model B com especificidade de 19,84% está a prever quase tudo como stress.

---

# Ponto 12 — confundimento entre datasets: CONFIRMADO e grave

## A identificação do dataset é fácil, e a série NN piora-a

| Entrada | ROC-AUC a distinguir WESAD de SWELL |
|---|---|
| 8 features resumo | 0,730 |
| **Série NN (estatísticas + resíduo de quantização)** | **0,909** |
| Só o nº de batimentos por janela | 0,622 |

Feature dominante na identificação: **`resid_700Hz`** (importância 0,455) — a
assinatura de quantização do WESAD. WESAD a 700 Hz dá intervalos múltiplos de
1,4286 ms; SWELL a 2048 Hz, múltiplos de 0,4883 ms. O Conv1D lê a frequência
de amostragem.

## A baseline que faltava: "só sei o dataset"

SWELL 60,7% stress (n=4489) · WESAD 35,6% stress (n=941). Prever a classe
maioritária de cada dataset, ignorando toda a fisiologia: **61,36%**.

```
Model A  60,42   <-- ABAIXO
Model B  60,92   <-- ABAIXO
Model D  61,18   <-- ABAIXO
"so sei o dataset"  61,36
Model C  61,95
MLP resumo 62,08
RandomForest 63,72
Conv1D so serie 65,17
Conv1D serie+resumo 65,65
```

**Três das quatro arquiteturas da tese não superam um classificador que ignora
a fisiologia e só olha para a origem do ficheiro.** O Model D é uma delas.

## O teste decisivo: dentro do SWELL apenas

4489 janelas, 60,73% stress, 22 sujeitos. Sem mistura de datasets, sem atalho
de frequência de amostragem.

### Split aleatório (5 seeds)

| Modelo | Acc | Especif. | ROC-AUC |
|---|---|---|---|
| Majority | 60,69 | 0,00 | 0,500 |
| LogReg | 63,88 | 29,92 | 0,636 |
| MLP resumo | 63,27 ± 1,93 | 40,23 | 0,655 ± 0,028 |
| RandomForest | 65,19 | 42,04 | 0,667 |
| Conv1D só série | 65,72 ± 1,56 | 55,07 | 0,710 ± 0,011 |
| **Conv1D série+resumo** | **65,84 ± 1,70** | 54,67 | **0,716 ± 0,027** |

```
so serie     vs MLP resumo:  acc +2,45 p=0,071    AUC +0,055 p=0,016 *
serie+resumo vs MLP resumo:  acc +2,56 p=0,022 *  AUC +0,061 p=0,020 *
AUC > 0,5: ambos p<0,0001 *
```

**A vantagem da série NN SOBREVIVE dentro de um só dataset.** Não era a
frequência de amostragem: +0,055 a +0,061 de AUC sobre o mesmo modelo
alimentado pelas 8 features resumo, com p<0,05.

### Por sujeito (GroupKFold k=5, 22 sujeitos)

| Modelo | Acc | Especif. | ROC-AUC |
|---|---|---|---|
| Majority | 60,78 | 0,00 | 0,500 |
| LogReg | 58,62 | 25,24 | 0,607 |
| RandomForest | 56,45 | 31,36 | 0,523 |
| MLP resumo | 54,81 ± 3,55 | 38,34 | 0,527 ± 0,077 |
| Conv1D só série | 54,64 ± 4,53 | 48,35 | 0,566 ± 0,064 |
| Conv1D série+resumo | 53,60 ± 5,85 | 47,98 | 0,558 ± 0,087 |

```
so serie vs MLP resumo:  AUC +0,039 p=0,008 *   (acc -0,17 p=0,936)
AUC > 0,5:  so serie p=0,080 NAO   serie+resumo p=0,208 NAO
```

**Dentro do SWELL e por sujeito, NENHUM modelo discrimina acima do acaso.**
Nem os da série NN. A vantagem relativa da série sobre o resumo mantém-se
(+0,039 de AUC, p=0,008) mas ambos partem de uma base que não é melhor que
adivinhar, e todos ficam abaixo do majority classifier (60,78%).

## Conclusão final

1. A vantagem da série NN é **real e não é artefacto de dataset** — sobrevive
   dentro do SWELL em split aleatório, com significância.
2. Mas **nenhum modelo generaliza para sujeitos novos** dentro de um único
   dataset. Os 0,596–0,607 de AUC subject-independent medidos no conjunto
   completo eram em parte separação de datasets: dentro do SWELL caem para
   0,558–0,566 e deixam de ser significativos.
3. A tese não pode afirmar deteção de stress subject-independent. Pode afirmar
   que a representação sequencial é superior à de features resumo, e que o
   limite é a generalização entre indivíduos.

---

# A DESCOBERTA CENTRAL: os dois datasets são incompatíveis

## Tamanho do efeito por dataset (d de Cohen, stress vs não-stress)

| Feature | WESAD (TSST) | SWELL (email/tempo) |
|---|---|---|
| hr_mean | **+1,73** | −0,42 |
| nni_mean | −1,69 | +0,44 |
| nni_min | **−1,82** | +0,32 |
| nni_max | −1,39 | +0,40 |
| sdnn | −0,38 | +0,22 |
| rmssd | −0,74 | +0,33 |
| nn50 | −0,57 | +0,38 |
| pnn50 | −0,75 | +0,43 |
| **média \|d\|** | **1,13** | **0,37** |

```
WESAD  nao-stress 72,9 bpm -> stress 98,6 bpm   (+25,7)
SWELL  nao-stress 76,7 bpm -> stress 72,1 bpm   (-4,6)
```

**Os sinais estão invertidos em TODAS as features.** O TSST do WESAD (falar em
público + aritmética mental) produz resposta simpática enorme. O protocolo do
SWELL (interrupções de email, pressão de tempo) faz a frequência **descer** —
é carga cognitiva, não stress fisiológico agudo, e contradiz a Tabela 6 da tese.

Juntar os dois é pedir ao modelo que separe rótulos fisiologicamente
incompatíveis. Os ~61% do Capítulo 4 são a média entre um problema resolúvel
(WESAD) e um mal colocado (SWELL).

# Dentro do WESAD (941 janelas, 35,60% stress, 15 sujeitos)

## Split aleatório (5 seeds)

| Modelo | Acc | Especif. | ROC-AUC |
|---|---|---|---|
| Majority | 64,55 | 100,00 | 0,500 |  <!-- corrigido 2026-09-23: no WESAD a classe maioritaria e nao-stress -->
| Model A | 79,89 | 75,08 | 0,922 |
| Model B | 79,89 | 77,70 | 0,917 |
| Model C | 83,49 | 83,44 | 0,929 |
| MLP resumo | 84,23 | 83,77 | 0,933 |
| LogReg | 84,97 | 90,82 | 0,919 |
| Model D | 85,08 | 87,70 | 0,927 |
| RandomForest | 86,77 | 91,15 | 0,937 |
| Conv1D só série | 89,21 | 93,11 | 0,952 |
| **Conv1D série+resumo** | **89,84 ± 1,37** | 92,13 | **0,959 ± 0,011** |

## Por sujeito (GroupKFold k=5, 15 sujeitos)

| Modelo | Acc | Especif. | ROC-AUC |
|---|---|---|---|
| Majority | 64,39 | 100,00 | 0,500 |  <!-- corrigido 2026-09-23: no WESAD a classe maioritaria e nao-stress -->
| RandomForest | 80,02 | 82,85 | 0,874 |
| Model B | 77,76 | 75,36 | 0,893 |
| Model D | 80,90 | 80,85 | 0,901 |
| Model A | 77,06 | 72,57 | 0,907 |
| **LogReg** | **82,62** | 87,68 | 0,907 |
| MLP resumo | 80,46 | 79,63 | 0,909 |
| Model C | 82,59 | 81,77 | 0,913 |
| Conv1D só série | 81,92 | 82,20 | 0,913 |
| **Conv1D série+resumo** | 81,62 | 81,57 | **0,915 ± 0,087** |

## Leitura

1. **A deteção de stress a 30 s FUNCIONA** quando o rótulo corresponde a um
   estado fisiológico real: AUC 0,874–0,915 subject-independent, contra
   0,52–0,61 no SWELL.
2. **Quase não cai do split aleatório para o split por sujeito** (0,959 → 0,915).
   Generaliza mesmo entre pessoas — ao contrário de tudo o resto testado.
3. **Em split aleatório a série NN domina** (0,952–0,959 vs 0,937 da RF).
4. **Por sujeito todos empatam** (0,893–0,915, SD ~0,08 com 5 folds). Com o
   sinal tão forte, a regressão logística basta — não há margem para a
   arquitetura acrescentar. O SD de 0,08 impede qualquer afirmação de ordem.
5. **A exatidão dos modelos profundos é pior que a da LogReg** em vários casos,
   com 941 janelas e até 219K parâmetros — falta de dados, não do sinal.

## Consequência para a tese

O Capítulo 4 mede a média entre dois problemas diferentes. A análise tem de ser
**por dataset**. O resultado a reportar deixa de ser "~61% e o teto é a
informação" e passa a ser "AUC 0,91 subject-independent no protocolo com
indução validada; o protocolo do SWELL não produz resposta fisiológica
detetável e os seus rótulos contradizem a literatura de HRV".

---

# Transferência entre datasets: prova de que os rótulos se opõem

`Classification Models/transfer_test.py`. Treinar num dataset, testar no outro.
Normalização com estatísticas só do dataset de treino (cenário realista).

| Treino → Teste | LogReg | RandomForest | MLP resumo | Conv1D série | Conv1D série+res. |
|---|---|---|---|---|---|
| **WESAD → SWELL** | 0,426 | 0,441 | 0,422 ± 0,007 | 0,403 ± 0,035 | 0,388 ± 0,031 |
| **SWELL → WESAD** | 0,178 | 0,377 | 0,394 ± 0,186 | 0,306 ± 0,147 | 0,343 ± 0,197 |

**As 10 medições estão abaixo de 0,5.** Nenhuma exceção.

AUC = 0,5 significaria ausência de sinal transferível. **AUC < 0,5 significa
sinal transferível mas invertido**: o modelo aprende uma regra válida e
aplica-a a rótulos que significam o oposto. É prova positiva de
incompatibilidade, não ausência de prova.

A regressão logística treinada no SWELL atinge **AUC 0,178** no WESAD — quase
perfeitamente anti-correlacionada. Inverter as suas previsões daria 0,822.

## Causa documentada no desenho do SWELL

Verificado em `SWELL-KW - overview available data.xlsx`, folha Sheet3, nos 25
participantes: **o bloco 1 é sempre a condição neutra (N)**; os blocos 2 e 3
são T e I em ordem contrabalançada.

```
PP1  N T I      PP2  N I T      PP3  N T I   ...   (bloco 1 = N nos 25)
```

A condição neutra nunca foi contrabalançada. Consequências medidas:

- 22/22 participantes têm frequência **mais baixa** nas condições de stress
  (delta mediano −5,26 bpm); no WESAD, 15/15 têm mais alta (+20,82 bpm)
- O bloco neutro não decai ao longo do tempo (slope −0,03 bpm, p=0,98) — está
  elevado do princípio ao fim, o que exclui habituação dentro do bloco
- Os blocos de stress **sobem** ao longo do tempo (+2,16 e +6,61 bpm, p<0,05):
  o stressor funciona, mas parte de um nível basal mais baixo
- Comparar só a 2ª metade do neutro com o stress mantém a inversão (−4,38 bpm)

Interpretação: ativação sustentada por novidade/observação no primeiro bloco,
que se sobrepõe ao efeito do stressor. **Um classificador treinado nestes
rótulos aprende a ordem do bloco, não o estado de stress.**

## A exportação da tese está correta

O script MATLAB lê `ppNN_data_cX.S00`, onde `cX` é o **bloco**. Como o bloco 1
é sempre N nos 25 participantes, o rótulo binário está certo. Um mapeamento
fixo `c2→stress_T`, `c3→stress_I` troca os nomes T/I em metade dos
participantes, mas ambos são classe 1 — sem efeito nos resultados binários.
**Só afeta afirmações sobre as condições T e I em separado.**

## Sobre Xia et al. (98,84–99,31% no SWELL)

Não contradiz. Usaram o conjunto de features pré-calculado do SWELL (janelas
longas, features de frequência, até 14 features), não ECG em bruto a 30 s, e
fizeram split aleatório com 5-fold — o que a própria Secção 2 da tese já
identifica como memorização de sujeito. Medido aqui: dentro do SWELL, a
RandomForest passa de 65,19% (split aleatório) para 56,45% (por sujeito).

---

# SWELL: o desenho confirmado pelo paper original

Koldijk et al., ICMI 2014 (`~/Downloads/ICMI 2014 paper_final_cr.pdf`), Secção 3:

> *"All participants worked under all 3 conditions. **The neutral condition was
> always the first condition**, in order to collect an uninfluenced baseline of
> normal working. The order of the two stressor conditions was counterbalanced."*

> *"Each of the experimental blocks started with a **relaxation phase of about
> 8 minutes** (which is typical for stress research) in which a nature film clip
> was shown."*

Condições: Neutral (livre, máx 45 min) · Time pressure (2/3 do tempo do neutral,
máx 30 min) · Interruptions (8 emails durante a tarefa).

## As gravações incluem a fase de relaxação

Durações medidas batem com tarefa + 8 min: neutral 50,4 min (máx 45+8),
stress_T 35,1 min (máx 30+8). **As primeiras 16 janelas de cada ficheiro são
um filme da natureza, não a condição experimental.**

hr_mean, janelas 0–15 (filme) vs 16+ (tarefa):

```
neutral    77,69 -> 76,41   (-1,28)
stress_I   71,30 -> 72,10   (+0,80)
stress_T   69,37 -> 73,48   (+4,11)
```

Excluir as primeiras 16 janelas é metodologicamente correto independentemente
do resto, mas resolve pouco: o delta entre blocos passa de −4,56 para −3,76 bpm
e a inversão mantém-se em 20/22 sujeitos.

## Análise intra-bloco (tarefa menos a relaxação do próprio bloco)

Controla a ordem do bloco comparando cada condição consigo mesma.

| Condição | n | Δ hr médio | SD | direção certa | p (vs 0) |
|---|---|---|---|---|---|
| neutral | 19 | +1,14 | 4,20 | 9/19 | 0,252 |
| stress_I | 19 | +1,24 | 4,78 | 13/19 | 0,274 |
| **stress_T** | 18 | **+4,88** | 6,03 | 14/18 | **0,0032 \*** |

Emparelhado por sujeito: stress_T − neutral = +2,14 bpm (p=0,101);
stress_I − neutral = +0,62 (p=0,613).

**A pressão temporal produz uma subida significativa face à própria linha de
base (+4,88 bpm, p=0,003).** As interrupções não (+1,24, p=0,274). Mas
nenhuma condição se separa do neutral com significância nesta análise
(p=0,101 e 0,613), e a média das duas contra neutral dá +1,38 bpm, p=0,225.

Comparação das duas análises:

```
entre blocos (como a tese faz):  -6,33 bpm   0/22 sujeitos na direcao certa
intra-bloco (correcto):          +1,38 bpm   9/14 sujeitos, p=0,225
```

## Leitura

O sinal do efeito **inverte** ao usar a linha de base correta: deixa de ser
−6,33 e passa a +1,38 bpm. Ou seja, a inversão observada é artefacto da ordem
dos blocos, não evidência de ausência de stress. Mas o efeito recuperado é
pequeno e não significativo — muito longe dos +25,7 bpm do TSST.

Conclusão defensável: os stressores do SWELL produzem uma resposta cardíaca
real mas pequena (detetável só na pressão temporal, +4,88 bpm face à própria
relaxação), demasiado pequena para classificação por janela a 30 s, e a
comparação convencional neutral-vs-stress mede predominantemente a ordem do
bloco.

---

# SWELL: o autorrelato dos próprios participantes confirma a inversão

Fonte: `Questionnaire Blok Results_final - performance added.ods` (repositório
DANS do SWELL-KW), folha `Data`. 22 participantes × 3 blocos.

## Autorrelato por condição (emparelhado por participante)

| Medida | N | T | I | T vs N | I vs N |
|---|---|---|---|---|---|
| **Stress** | 2,89 | 3,84 | 3,16 | +0,94 | +0,27 |
| **Arousal** | 5,68 | 4,40 | 5,08 | **−1,28 \*** | −0,60 |
| MentalEffort | 5,54 | 6,32 | 6,53 | +0,78 | **+0,99 \*** |
| TemporalDemand | 5,74 | 7,06 | 5,90 | **+1,33 \*** | +0,16 |
| Effort | 5,24 | 6,06 | 5,89 | +0,82 | +0,65 |
| Frustration | 3,46 | 3,45 | 3,57 | −0,01 | +0,11 |
| Valence | 4,16 | 3,68 | 3,32 | −0,48 | **−0,84 \*** |

## Os dois resultados que interessam

1. **O stress autorrelatado NÃO sobe significativamente** em nenhuma condição
   manipulada (+0,94 e +0,27, ambos n.s.).
2. **O arousal autorrelatado DESCE**, significativamente na pressão temporal
   (−1,28, p<0,05) — **a mesma direção da frequência cardíaca**.

A manipulação funcionou naquilo a que se propunha: TemporalDemand sobe em T
(+1,33\*), MentalEffort sobe em I (+0,99\*), Valence desce em I (−0,84\*). Ou
seja, aumentou **carga de trabalho** e **valência negativa**, não ativação.

## Concordância entre autorrelato e fisiologia ao nível da condição

Excluindo os 8 min de filme, 22 participantes:

```
arousal  T vs N: -0,88   p=0,025 *      hr  T vs N: -4,43 bpm  p=0,013 *
arousal  I vs N: -0,25   p=0,643        hr  I vs N: -7,21 bpm  p=0,002 *
```

**Os participantes reportam menos ativação nas condições de stress, e a
frequência cardíaca acompanha.** A fisiologia medida está de acordo com a
experiência subjetiva relatada — o que valida o pipeline de extração de ECG
desta tese de forma independente.

## Correlação intra-sujeito bloco a bloco (n=59 blocos)

```
Arousal       r=+0,083  p=0,533
Stress        r=-0,151  p=0,253
Valence       r=+0,103  p=0,437
MentalEffort  r=-0,171  p=0,195
```

Nenhuma correlação ao nível do bloco individual. A concordância existe na
média das condições mas não bloco a bloco: a frequência cardíaca de um bloco
não prediz o quanto essa pessoa se sentiu ativada nesse bloco.

## Conclusão

O rótulo "stress" do SWELL-KW não corresponde a stress autorrelatado
(+0,94, n.s.) nem a ativação fisiológica (−4,4 a −7,2 bpm). Corresponde a
carga de trabalho. **A tese não pode usar estes rótulos como alvo de deteção
de stress a partir de HRV, e a razão não é uma falha de processamento — é que
os próprios participantes não reportaram estar mais stressados nem mais
ativados.**

Isto é uma validação externa do pipeline: a fisiologia medida concorda com o
autorrelato, na direção contrária ao rótulo.

---

# Explicabilidade refeita no WESAD (resolve pontos 5 e 7)

`Classification Models/explain_wesad.py` — treina o Model D nos dados V5
filtrados ao WESAD, guarda em `saved_models/wesad_model.h5`, e entrega o modelo
em memória ao `run_explainability.py`. Figuras da V4 preservadas em
`Classification Models/results_v4_backup/`.

## Matriz de confusão (nova Tabela 8) — WESAD, seed 42, N=189

| | Prev: não-stress | Prev: stress |
|---|---|---|
| **Real: não-stress** | 107 | 15 |
| **Real: stress** | 17 | 50 |

```
acc 83,07  prec 76,92  rec 74,63  spec 87,70  AUC 0,910
```

Comparar com a Tabela 8 da V4 (conjunto combinado): 230/348/171/571, acc 60,68,
**especificidade 39,8%**. A especificidade passa de 39,8% para **87,70%**.
O modelo deixa de disparar falsos alarmes em 3 de cada 5 janelas calmas.

## Importância global (nova Figura 14)

| | V4 (combinado) | V5 (WESAD) |
|---|---|---|
| 1.º | nni_mean 1,000 | **nni_mean 1,0000** |
| 2.º | nni_min 0,941 | **hr_mean 0,5222** |
| 3.º | nn50 0,910 | **nni_min 0,4552** |

Na V4 os seis primeiros ficavam entre 0,74 e 1,00 — ranking achatado, top-3 a
somar só 44,9%, e `hr_mean` em 6.º. Agora há uma hierarquia clara e `hr_mean`
sobe para 2.º, o que é coerente: no WESAD é a feature com maior tamanho de
efeito (d = 1,73). **As atribuições passam a alinhar com a Tabela 6 da tese.**

Isto resolve o ponto 5 pela causa e não por seleção de exemplos: no conjunto
combinado as atribuições nunca poderiam confirmar a fisiologia, porque o alvo
misturava dois estados opostos.

## Fidelidade e estabilidade

| | V4 (combinado) | V5 (WESAD) |
|---|---|---|
| Fidelidade | 0,022 | **0,1958** |
| Estabilidade | 0,878 | 0,8060 |

**A fidelidade sobe quase 9×.** Mascarar as features de maior atribuição custa
agora 19,6 pp de exatidão, contra 2,2 pp antes. O modelo depende mesmo das
features que diz usar.

Consequência para a escrita: o argumento da Secção 4.4.3 — fidelidade baixa
explicada por redundância — **deixa de ser necessário**. A fidelidade baixa da
V4 era sintoma de o modelo não ter sinal fisiológico a que se agarrar, não de
redundância entre features. A redundância existe (`sdnn ~ rmssd` r = 0,978) mas
não era a causa. Reescrever essa subsecção de raiz.

A estabilidade desce ligeiramente (0,878 → 0,806), o que é esperado: com
atribuições mais concentradas há menos margem para o ranking se manter idêntico
sob perturbação.

## Por corrigir ainda (só texto e rótulos)

O `AttentionExtractor` continua a chamar "atenção" a gradientes, e os eixos das
figuras novas continuam a dizer "Attention Score". O ponto 1.1 da supervisora
não é resolvido por esta corrida — é edição de texto e de rótulos.

---

# Correção da normalização das atribuições (2026-09-17)

O min-max mapeava sempre a feature de menor gradiente para 0 exacto. Trocado por
divisão pelo máximo. Mesma amostra (`true_positive_high_conf`, predição
STRESS 99,0%), mesmo modelo, só muda a apresentação:

| Feature | min-max (antes) | ÷ máximo (agora) |
|---|---|---|
| nni_min | 1,00 | 1,00 |
| nni_mean | 0,52 | 0,59 |
| hr_mean | 0,38 | 0,47 |
| nni_max | 0,30 | 0,40 |
| nn50 | 0,18 | 0,30 |
| pnn50 | 0,16 | 0,29 |
| rmssd | 0,02 | 0,17 |
| **sdnn** | **0,00** ← falso | **0,15** |

O `sdnn` tinha 0,00 apenas por ser o mínimo. Nenhuma barra fica agora a zero.
A suspeita da supervisora de "erro de implementação" era justificada quanto ao
efeito, embora a causa fosse de apresentação e não de cálculo.

# A amostra de estudo de caso agora confirma a fisiologia

`true_positive_high_conf`, WESAD, predição **STRESS 99,0%** (na V4 era 85,0%):

```
valores normalizados       atribuição (gradiente)
  hr_mean    +0,67          nni_min   1,00
  nni_mean   -0,83          nni_mean  0,59
  nni_min    -0,80          hr_mean   0,47
  nni_max    -0,35          nni_max   0,40
```

Frequência cardíaca **acima** da média com intervalos NN **abaixo** — assinatura
clássica de ativação simpática, exatamente o que a Tabela 6 da tese prevê.

Comparar com a amostra da V4 (`S15_stress_T`), onde `nni_max` +3,2, `sdnn` +2,95
e `rmssd` +2,65 estavam todos muito **acima** da média numa janela rotulada como
stress, contradizendo a Tabela 6. Essa janela era ruído (ver "As amostras dos
estudos de caso") e foi descartada pelo ETL corrigido.

**O ponto 5 da supervisora fica resolvido pela causa**, não por escolha de
exemplos: no conjunto combinado as atribuições nunca poderiam confirmar a
fisiologia, porque o alvo misturava dois estados fisiologicamente opostos.

---

# Itens finais de código (2026-09-17)

## Figura 14 — a crítica da supervisora deixa de se aplicar

Importância global no WESAD, normalizada pelo máximo:

```
nni_mean 1,0000 · hr_mean 0,5222 · nni_min 0,4552 · nn50 0,4185
nni_max  0,3799 · pnn50   0,3593 · sdnn    0,3326 · rmssd 0,2996
```

| | V4 (combinado) | V5 (WESAD) |
|---|---|---|
| Soma total | 6,354 | 3,767 |
| Top-3 | **44,9%** | **52,5%** |
| Diferença 3.º vs 4.º | 0,009 | 0,037 |
| Rácio 1.º/último | 1,85× | 3,34× |

A crítica *"as três primeiras somam 45%, não a maioria"* **já não se aplica** —
mas por pouco (52,5%). Evitar a palavra "maioria" na legenda; dizer o valor.
O corte no top-3 também deixa de ser arbitrário: a diferença 3.º–4.º passa de
0,009 para 0,037, quatro vezes maior.

## Latência de inferência

**48,18 ms/amostra** no WESAD (V4 reportava 63 ms), CPU Intel i5-8250U.
`measure_inference.py` actualizado para `saved_models/wesad_model.h5` e para a
CSV V5 filtrada ao WESAD, com seleção de features por nome. Backup em
`measure_inference.py.bak`. Numa janela de 30 s, continua desprezável.

## Demo do agente (Secção 4.5) — amostras novas

CSV do WESAD gerada no scratchpad; o `stress_feedback_agent.py` replica o split
(`random_state=42`, estratificado) e a normalização, por isso os índices batem
certo com os do `explain_wesad.py`.

**As amostras do agente são agora AS MESMAS das figuras de estudo de caso** —
na V4 não eram, e isso obrigava a texto ambíguo.

| Amostra | Índice no teste | Predição | Figura correspondente |
|---|---|---|---|
| Stress | **6** | STRESS 99,0% (rótulo: stress) | `true_positive_high_conf_*` |
| Não-stress | **3** | NO STRESS 83,0% (rótulo: não-stress) | `true_negative_high_conf_*` |

Respostas geradas guardadas abaixo para citação directa no Capítulo 4.

### Stress (99,0%)

> Your body's asking for a small reset — here are a few things that can help
> right now: **1. Try box breathing** — Inhale for 4 counts, hold for 4, exhale
> for 4, hold for 4. Repeat for 2-3 minutes. This gently signals your nervous
> system to shift out of high-alert mode. **2. Unclench and drop your
> shoulders** — Notice your jaw, hands, and shoulders. Stress often hides in
> tight muscles you're not even aware of. **3. Step away for a 5-minute
> break**, if you can. Look out a window, stretch, or simply sit somewhere
> quieter. **4. Ground yourself with your senses** — Name 3 things you can see,
> 2 you can hear, and 1 you can feel. **5. Sip some water slowly** — A small,
> calming ritual that also supports your body physically. You're doing great by
> checking in with yourself — that awareness is the first step toward feeling
> more balanced.

### Não-stress (83,0%)

> Your body's in a great rhythm right now. Here's how to keep that calm energy
> going: **Stay hydrated** — grab a glass of water if you haven't recently.
> **Keep moving** — a quick stretch or short walk can help sustain this steady
> state. **Protect this calm** — step outside for some fresh air or natural
> light. **Tonight's sleep matters** — keep a consistent wind-down routine.
> **Notice what's working** — whatever you did today is paying off. You're
> doing a wonderful job taking care of yourself.

Nota para a escrita: as respostas trazem emoji (🌿 💪✨) e markdown (`**`), que
convém remover ou mencionar ao citar em LaTeX.

A limitação da Secção 4.5 **mantém-se e deve continuar escrita**: só a classe
predita atravessa a fronteira da API, por isso 99,0% e 51% geram pedidos
idênticos. Com a especificidade agora em 87,70% (contra 39,8%), o argumento
muda de tom — o sistema já não afirma stress com falsa certeza em 3 de cada 5
janelas calmas — mas a insensibilidade à confiança continua a ser real.

## Bug encontrado (não corrigido)

`stress_feedback_agent.py` tem **duas** definições de `load_and_run`, em L255 e
L348. A segunda sobrepõe-se à primeira, que é código morto. Mesmo padrão do
`build_model` duplicado em `Models.py` (L42 e L178). Não afeta resultados, mas
confunde quem leia.

---

# LOSO no WESAD (15 folds, um por sujeito)

`--cv loso` acrescentado ao `retrain_v5.py` e ao `train_sequence.py`.
Todos os 15 sujeitos têm ambas as classes (~22 stress, ~40 não-stress), por isso
a AUC é calculável em cada fold.

| Modelo | Acc | Especif. | ROC-AUC | Params |
|---|---|---|---|---|
| Majority | 64,35 | 100,00 | 0,500 | — |  <!-- corrigido 2026-09-23: no WESAD a classe maioritaria e nao-stress -->
| Model A (LSTM) | 74,75 ± 21,31 | 69,32 ± 37,14 | 0,945 ± 0,091 | 19 009 |
| Model B (Deep LSTM) | 77,73 ± 18,24 | 74,89 ± 31,31 | 0,946 ± 0,082 | 132 993 |
| RandomForest | 79,95 ± 15,97 | 84,00 ± 25,91 | 0,902 ± 0,128 | — |
| MLP resumo | 80,21 ± 16,71 | 78,90 ± 27,35 | 0,940 ± 0,095 | 7 393 |
| Model C (CNN-LSTM) | 80,94 ± 16,18 | 79,73 ± 26,76 | 0,948 ± 0,074 | 77 249 |
| Model D (CNN-Transf.) | 81,37 ± 14,93 | 82,98 ± 25,48 | 0,937 ± 0,077 | 218 625 |
| **LogReg** | **82,55 ± 16,10** | 87,94 ± 26,10 | **0,951 ± 0,078** | — |
| Conv1D só série | 83,23 ± 17,30 | 83,49 ± 28,24 | 0,939 ± 0,103 | 27 969 |
| **Conv1D série+resumo** | **83,54 ± 16,12** | 84,73 ± 26,08 | 0,925 ± 0,131 | 32 929 |

**Tudo empata em AUC entre 0,925 e 0,951**, com SD de 0,07 a 0,13. Nenhuma
diferença é defensável. A regressão logística tem a AUC mais alta de todas.

RandomForest por sujeito: AUC mínima 0,614, máxima 1,000, **11/15 acima de 0,9**.

## LOSO dá AUC MAIS ALTA que GroupKFold k=5 (0,951 vs 0,907) — porquê

Duas causas, e a segunda é metodologicamente importante:

1. **Mais dados de treino**: LOSO treina com 14 sujeitos, GroupKFold k=5 com 12.
2. **Mede uma pergunta diferente.** Com um só sujeito no fold de teste, a AUC
   avalia se o modelo ordena bem **as janelas dessa pessoa** — comparação
   intra-sujeito. Com 3 sujeitos no fold, tem também de ordenar entre pessoas
   com basais diferentes, o que é mais difícil.

Nenhuma está errada; medem perguntas distintas. **A de LOSO corresponde ao uso
real** (um wearable acompanha uma pessoa), a de GroupKFold ao cenário de um
modelo único aplicado a uma população sem calibração.

## O contraste que justifica a calibração por sujeito

```
AUC             0,925–0,951   (SD 0,07–0,13)   ordenacao quase perfeita
Exatidao        74,8–83,5     (SD 15–21)       limiar mal calibrado
Especificidade  69,3–87,9     (SD 25–37)       varia enormemente
```

Amplitudes por fold: exatidão de 37,5% a 100%; especificidade de **0,00% a
100%**. Houve folds com AUC 0,862 e especificidade 0,00 — **ordenação perfeita,
decisão inútil**, porque o limiar fixo de 0,5 não serve a uma pessoa cujo basal
difere da média da população.

É a justificação quantitativa para a calibração por sujeito no trabalho futuro:
**a informação está no sinal, o que falta é o limiar por pessoa.**

## Consequência para a escolha do modelo

Com LOSO, as 10 configurações empatam. A regressão logística tem a AUC mais alta
(0,951) com zero hiperparâmetros. **Não há base para reivindicar superioridade
de nenhuma arquitetura profunda no WESAD** — o sinal é forte (hr_mean d=1,73) e
um modelo linear basta.

A vantagem da representação sequencial, que era clara em split aleatório
(+0,061 de AUC dentro do SWELL, p=0,020), **não se manifesta aqui**: quando o
sinal é grande e linearmente separável, não há margem para a arquitetura
acrescentar.

---

# CORREÇÕES encontradas ao escrever o README (2026-09-18)

Três afirmações anteriores deste ficheiro estavam erradas. Os valores abaixo
prevalecem.

**1. WESAD LOSO: a gama é 0,902–0,951, não 0,925–0,951.** A RandomForest tem
AUC 0,902 ± 0,128 e ficou de fora da gama citada. Os modelos profundos, o MLP e
a LogReg estão entre 0,925 e 0,951.

**2. SWELL por sujeito: a melhor AUC é 0,607 (LogReg), não 0,566.**
Recalculado, GroupKFold k=5:

```
LogReg            AUC 0,607 ± 0,087  folds [0,479 0,653 0,560 0,698 0,645]  p=0,052
LogReg balanced   AUC 0,606 ± 0,087                                          p=0,053
Conv1D serie      AUC 0,566 ± 0,064                                          p=0,080
Conv1D serie+res. AUC 0,558 ± 0,087                                          p=0,208
MLP resumo        AUC 0,527 ± 0,077                                          p=0,476
RandomForest      AUC 0,523 ± 0,083                                          p=0,569
```

A conclusão "nenhum modelo discrimina acima do acaso" mantém-se a α = 0,05,
**mas por pouco** (p = 0,052). Não a apresentar como se o SWELL fosse ruído puro.
Formulação segura: "nenhum modelo atinge significância contra o acaso; o mais
próximo é a regressão logística (AUC 0,607, p = 0,052)".

**3. Participantes do SWELL em falta: 8, 11 e 23.** Pela folha de
disponibilidade (`SWELL-KW - overview available data.xlsx`, coluna
"Physiology: heart"):

| PP | Blocos 1 / 2 / 3 | Leitura |
|---|---|---|
| 11 | yes* / **(no)** / yes* | gravação cardíaca falhou no bloco 2 |
| 23 | **no*** / yes / yes | gravação cardíaca falhou no bloco 1 (o neutral) |
| 8 | yes* / yes* / yes** | **sem falha registada** — motivo da exclusão desconhecido |

Responde parcialmente ao "falta explicar porquê" dos 22 em vez de 25. O 8 tem
de ser verificado nos ficheiros originais antes de se escrever uma justificação.
Os asteriscos remetem para notas na folha que não foram lidas.

# Reprodutibilidade: `time_duration.csv` do repositório está incompleto

`Data Processing/time_duration.csv` (no git, 95 linhas) **não tem a linha do
`S12_stress_T.csv`**. O ficheiro usado no ETL V5 é
`/home/eduarda/Desktop/Tese 2026/Data Processing/time_duration.csv` (96 linhas).
Com a cópia do repositório o ETL produz 4 janelas a menos que as 5430 citadas.
Linha em falta: `S12_stress_T.csv,2126.000,1,2048`.

---

# F1 vs F2 dentro do WESAD (2026-09-21) — o furo da Secção 4.1 fechado

O teste do ponto 9 (linha 513) foi corrido no **conjunto combinado**. Com o
WESAD a passar a resultado principal, a ablação de features tinha de ser
replicada lá, senão a Contribuição 1 ficava sustentada por dados que a tese já
não usa como resultado principal.

```
python3 retrain_v5.py /home/eduarda/Desktop/ProcessedECGData_v5.csv \
    --models D --dataset WESAD --features F2 --seeds 5 --out results_wesad_F2_D.csv
# idem com --features F1 --out results_wesad_F1_D.csv
```

Model D, 941 janelas do WESAD, split aleatório 80/20 estratificado, seeds 42–46,
mesmos splits nas duas condições.

| | Acc | Precisão | Recall | Especif. | ROC-AUC |
|---|---|---|---|---|---|
| F2 (8 temporais) | **84,97 ± 1,27** | 77,83 ± 3,62 | 80,90 ± 2,45 | **87,21 ± 2,76** | **0,9251 ± 0,0066** |
| F1 (10, c/ frequência) | 84,44 ± 1,97 | 75,88 ± 3,77 | **82,69 ± 3,44** | 85,41 ± 3,45 | 0,9193 ± 0,0106 |

## Testes emparelhados (mesmas seeds, mesmos splits)

```
exatidao   dif +0,529 +/- 2,510 pp    p=0,662   NAO SIGNIFICATIVO
AUC        dif +0,0058 +/- 0,0092     p=0,233   NAO SIGNIFICATIVO
recall     dif -1,791 +/- 3,059 pp    p=0,261   NAO SIGNIFICATIVO
especif.   dif +1,803 +/- 3,905 pp    p=0,360   NAO SIGNIFICATIVO
precisao   dif +1,955 +/- 5,064 pp    p=0,437   NAO SIGNIFICATIVO
```

**Nenhuma métrica distingue os dois conjuntos de features.** A conclusão do
conjunto combinado replica-se no WESAD, e com margem maior: o SD da diferença
de exatidão é 2,51 pp para um efeito de 0,53 pp.

A direção mantém-se a mesma que no combinado — F2 ganha em exatidão, AUC,
especificidade e precisão, F1 ganha em recall — e nenhuma sobrevive ao teste.

**Formulação para a Secção 4.1 e para a Contribuição 1:** "remover as duas
features de frequência não degrada o desempenho" (verificado no conjunto
combinado, p=0,639, e no WESAD, p=0,662). A justificação para usar 8 em vez de
10 é o custo de cálculo e o alinhamento com a literatura de *surrogacy*, não
ganho medido. **Não** escrever "melhora" nem "validação empírica de que as
features de frequência não são fiáveis".

CSVs por seed: `Classification Models/results_wesad_F{1,2}_D.csv`.

## Nota de reprodutibilidade que sai desta corrida

Esta corrida do F2 no WESAD dá **acc 84,97 e AUC 0,925**. A tabela da linha 707
(mesmo modelo, mesmo dataset, mesmas 5 seeds, mesmo split) tinha dado
**85,08 e 0,927**. Diferença de 0,11 pp e 0,002 de AUC entre duas invocações
idênticas.

Terceira medição independente do não-determinismo do TensorFlow em CPU, a somar
às duas já registadas (Model D F2 no combinado: 61,18% e 62,08%). Confirma a
regra: **nenhuma diferença abaixo de ~1 pp é interpretável**, e os valores da
Tabela 8 (uma seed) têm de ser apresentados com a média de 5 seeds ao lado.

---

# Baselines clássicos no WESAD: reproduzidos e com CSV (2026-09-22)

Os valores de LogReg e RandomForest no WESAD estavam só neste ficheiro, sem CSV.
Reproduzidos ao decimal nos três protocolos por
`Classification Models/wesad_baselines.py` → `results_wesad_baselines.csv`.

**As definições que reproduzem os valores registados** — e que a tese tem de
declarar, porque não são simétricas:

| Classificador | `class_weight` |
|---|---|
| LogReg | **nenhum** |
| RandomForest | `"balanced"` |
| Modelos profundos A–D | `"balanced"` × 1,2 na classe stress |

A LogReg com `class_weight="balanced"` dá **a mesma AUC** (0,920 vs 0,919 em
split aleatório; 0,952 vs 0,951 em LOSO) mas **3 pp menos de exatidão** e 7 pp
menos de especificidade, porque o peso desloca o limiar. Como a comparação do
Capítulo 4 se faz pela AUC, a conclusão não muda — mas a tabela tem de dizer
qual a variante usada. As duas ficam na CSV.

Ambas as variantes da RandomForest dão o mesmo, dentro de 0,7 pp.

# Qual é o melhor modelo no WESAD: depende do protocolo

Consolidado das CSVs (`results_wesad_ABCD_*`, `results_wesad_{random,group,loso}_*`,
`results_wesad_baselines.csv`). Ordenado por AUC.

## Split aleatório, 5 seeds — a sequência NN domina

| Modelo | Acc | Especif. | ROC-AUC |
|---|---|---|---|
| **Conv1D série+resumo** | **89,84 ± 1,37** | 92,13 | **0,959 ± 0,011** |
| Conv1D só série | 89,21 ± 1,78 | 93,11 | 0,952 ± 0,009 |
| RandomForest | 86,77 ± 2,28 | 91,15 | 0,937 ± 0,015 |
| MLP resumo | 84,23 ± 2,46 | 83,77 | 0,933 ± 0,012 |
| Model C | 83,49 ± 1,89 | 83,44 | 0,929 ± 0,012 |
| Model D | 85,08 ± 2,06 | 87,70 | 0,927 ± 0,013 |
| Model A | 79,89 ± 2,70 | 75,08 | 0,922 ± 0,011 |
| LogReg | 84,97 | 90,82 | 0,919 ± 0,013 |
| Model B | 79,89 ± 1,63 | 77,70 | 0,917 ± 0,012 |
| Majority | 64,55 | 100,00 | 0,500 |  <!-- corrigido 2026-09-23: no WESAD a classe maioritaria e nao-stress -->

## GroupKFold k=5 — empate (SD ~0,08 esmaga as diferenças)

| Modelo | Acc | Especif. | ROC-AUC |
|---|---|---|---|
| **Conv1D série+resumo** | 81,62 ± 8,61 | 81,57 | **0,915 ± 0,087** |
| Conv1D só série | 81,92 ± 9,85 | 82,20 | 0,913 ± 0,083 |
| Model C | **82,59 ± 9,19** | 81,77 | 0,913 ± 0,079 |
| MLP resumo | 80,46 ± 11,12 | 79,63 | 0,909 ± 0,079 |
| LogReg | 82,62 ± 10,70 | 87,68 | 0,907 ± 0,090 |
| Model A | 77,06 ± 11,42 | 72,57 | 0,907 ± 0,080 |
| Model D | 80,90 ± 7,09 | 80,85 | 0,901 ± 0,081 |
| Model B | 77,76 ± 9,16 | 75,36 | 0,893 ± 0,064 |
| RandomForest | 80,02 ± 7,78 | 82,85 | 0,874 ± 0,086 |

## LOSO, 15 folds — a LogReg tem a AUC mais alta de todas

| Modelo | Acc | Especif. | ROC-AUC |
|---|---|---|---|
| **LogReg** | 82,55 ± 16,10 | 87,94 | **0,951 ± 0,078** |
| Model C | 80,94 ± 16,18 | 79,73 | 0,948 ± 0,074 |
| Model B | 77,73 ± 18,24 | 74,89 | 0,946 ± 0,082 |
| Model A | 74,75 ± 21,31 | 69,32 | 0,945 ± 0,091 |
| MLP resumo | 80,21 ± 16,71 | 78,90 | 0,940 ± 0,095 |
| Conv1D só série | 83,23 ± 17,30 | 83,49 | 0,939 ± 0,103 |
| Model D | 81,37 ± 14,93 | 82,98 | 0,937 ± 0,077 |
| **Conv1D série+resumo** | **83,54 ± 16,12** | 84,73 | 0,925 ± 0,131 |
| RandomForest | 79,945 ± 15,97 | 84,00 | 0,902 ± 0,128 |
| Majority | 64,35 | 100,00 | 0,500 |  <!-- corrigido 2026-09-23: no WESAD a classe maioritaria e nao-stress -->

## A resposta defensável

**Em split aleatório a resposta é clara**: o Conv1D sobre a série NN mais as
features resumo ganha por 2,2 pp de AUC ao melhor clássico (0,959 vs 0,937) e
por 3,2 pp de AUC ao melhor dos quatro modelos da tese. Aqui há margem para a
arquitetura e a vantagem da representação sequencial é real.

**Nos dois protocolos por sujeito, não há vencedor.** Em GroupKFold as nove
configurações ficam entre 0,874 e 0,915 com SD de 0,08; em LOSO entre 0,902 e
0,951 com SD de 0,07 a 0,13. A ordenação inverte-se entre protocolos: o
Conv1D série+resumo é 1.º em GroupKFold e **último dos modelos** em LOSO, e a
LogReg é 5.ª em GroupKFold e 1.ª em LOSO. Nenhuma das duas ordenações é
interpretável.

**Para a tese**: reivindicar a vantagem sequencial no split aleatório, onde é
mensurável, e declarar o empate no subject-independent. Não eleger um vencedor
por AUC em LOSO — a inversão entre protocolos mostra que isso é ruído de
amostragem de 15 sujeitos.

O melhor **dos quatro modelos da tese** no WESAD: **Model D** em split aleatório
(85,08 e AUC 0,927, o melhor em exatidão e especificidade dos quatro) e **Model
C** nos dois protocolos por sujeito (AUC 0,913 e 0,948), sempre dentro do ruído.

---

# Testes emparelhados dentro do WESAD (2026-09-22)

Faltavam. A vantagem sequencial estava testada no combinado e dentro do SWELL,
nunca no WESAD, que passa a ser o resultado principal.

**Os splits são emparelháveis entre scripts.** O `retrain_v5.py`, o
`train_sequence.py` e o `wesad_baselines.py` constroem os splits exactamente da
mesma forma (`train_test_split(np.arange(len(y)), test_size=0.2,
random_state=s, stratify=y)` para s em 42–46; `GroupKFold(5)`;
`LeaveOneGroupOut()`), sobre as mesmas linhas filtradas na mesma ordem. Logo os
resultados por seed/fold são pares válidos e o t-test emparelhado aplica-se.

## Split aleatório (5 pares)

| Comparação | Δ exatidão | Δ AUC |
|---|---|---|
| Conv1D série+resumo − **MLP resumo** | +5,61 pp, **p=0,0005** | +0,026, **p=0,023** |
| Conv1D série+resumo − **LogReg** | +4,87 pp, **p=0,014** | +0,040, **p=0,014** |
| Conv1D série+resumo − **Model D** | +4,76 pp, **p=0,004** | +0,032, **p=0,004** |
| Conv1D só série − Model D | +4,13 pp, **p=0,040** | +0,024, **p=0,035** |
| Conv1D série+resumo − **RandomForest** | +3,07 pp, p=0,063 | +0,022, p=0,060 |
| Model D − Model C | +1,59 pp, p=0,222 | −0,001, p=0,692 |

**A nuance que a tese tem de escrever:** a representação sequencial bate os
modelos profundos e os lineares com significância, mas **não bate a
RandomForest** (p=0,06 nas duas métricas). A afirmação defensável é "a série NN
supera as features resumo", não "supera todos os métodos".

## GroupKFold k=5 (5 pares)

```
Conv1D serie+resumo - Model D   acc +0,72 pp p=0,718    AUC +0,014 p=0,150
Conv1D serie+resumo - LogReg    acc -1,00 pp p=0,768    AUC +0,009 p=0,368
Model C - Model D               acc +1,70 pp p=0,382    AUC +0,012 p=0,031 *
```

O único p<0,05 é o C−D em AUC, com efeito de 0,012 sobre 5 folds. Frágil e sem
correção para comparações múltiplas. **Não construir argumento sobre ele.**

## LOSO (15 pares)

```
LogReg - Model D                 acc +1,18 pp p=0,420   AUC +0,014 p=0,101
LogReg - Conv1D serie+resumo     acc -0,99 pp p=0,582   AUC +0,026 p=0,486
Conv1D serie+resumo - Model D    acc +2,17 pp p=0,198   AUC -0,012 p=0,740
Model C - Model D                acc -0,43 pp p=0,639   AUC +0,011 p=0,319
```

**Nada é significativo em LOSO.** Confirma o empate por via do teste e não só
pela sobreposição dos SD.

---

# Baselines do conjunto combinado: reproduzidos, com dois desvios (2026-09-22)

`results_pooled_baselines.csv`, mesma receita dos do WESAD (StandardScaler no
treino, LogReg sem peso de classe, RF `balanced`), mais a regra
"só sei o dataset" calculada dentro de cada split.

| | Registado antes | Reproduzido |
|---|---|---|
| LogReg (split aleatório) | 59,50 ± 1,42 · AUC 0,615 | **60,06 ± 1,22 · AUC 0,619** |
| RandomForest (split aleatório) | 63,72 ± 0,63 · AUC 0,683 | **63,85 ± 0,67 · AUC 0,683** |
| "só sei o dataset" | 61,36 | **62,58 ± 1,07** (aleatório) · **61,35 ± 4,43** (GroupKFold) |

Os desvios são de 0,56 e 0,13 pp e não mudam nenhuma conclusão. A causa provável
é a normalização: no WESAD a receita reproduz ao decimal, no combinado não, o que
aponta para a corrida antiga ter usado o `normalize_data()` do `Data.py` (com
clipping por IQR) em vez do StandardScaler. **Usar os valores reproduzidos**, que
têm CSV.

O 61,36% registado corresponde à versão por sujeito da regra, não à do split
aleatório. Citar 62,58% no split aleatório e 61,35% em GroupKFold.

## Combinado, GroupKFold k=5 (37 sujeitos) — a tabela completa

| Modelo | Acc | Especif. | ROC-AUC |
|---|---|---|---|
| **"só sei o dataset"** | **61,35 ± 4,43** | — | — |
| Conv1D só série | 57,13 ± 4,79 | 54,10 | **0,607 ± 0,057** |
| Conv1D série+resumo | 57,03 ± 5,40 | 58,33 | 0,596 ± 0,067 |
| Model A | 57,75 ± 3,45 | 31,45 | 0,568 ± 0,064 |
| LogReg | 56,32 ± 4,61 | 33,52 | 0,566 ± 0,056 |
| Majority | 56,36 ± 4,14 | 0,00 | 0,500 |
| Model B | 56,00 ± 2,52 | 19,84 | 0,553 ± 0,035 |
| MLP resumo | 53,01 ± 3,33 | 38,38 | 0,545 ± 0,063 |
| Model C | 55,29 ± 1,02 | 31,90 | 0,542 ± 0,050 |
| Model D | 53,74 ± 1,70 | 41,15 | 0,540 ± 0,059 |
| RandomForest | 53,56 ± 3,17 | 39,20 | 0,531 ± 0,072 |

**No combinado por sujeito, a regra que ignora a fisiologia bate todos os
modelos treinados.** É a frase mais forte do controlo negativo.

## Correção ao texto da Secção 4.2

A taxa de falsos alarmes do Model D no combinado é **65,65%** (especificidade
34,35 ± 7,16, 5 seeds, dados V5) e não os 60,2% da V4 (uma seed, dados sujos).
Qualquer comparação com a especificidade de 87,70% do WESAD tem de usar 34,35.

---

# Estudos de caso: os quatro, com números (2026-09-22)

Extraídos do `saved_models/wesad_model.h5` com o split replicado (seed 42,
`normalize_data`), Model D, N=189. Script em
`scratchpad/case_numbers.py` da sessão. Saliência = |∂ŷ/∂x| normalizada pelo
máximo; IG = (x − baseline) × gradiente médio, 50 passos, baseline = vector zero.

| Caso | Índice | Ficheiro | Predição |
|---|---|---|---|
| TP alta confiança | 6 | `S7_stress` | STRESS 99,0% |
| TN alta confiança | 3 | `S9_baseline` | NO STRESS (p=0,17) |
| FP | 0 | `S2_baseline` | STRESS 84,8% |
| FN | 23 | `S9_stress` | NO STRESS (p=0,423) |

Contagens: 37 TP com p>0,8 · 78 TN com p<0,2 · 15 FP · 17 FN.

## TP (índice 6, S7_stress) — confirma a fisiologia

```
feature    bruto    normaliz.  |grad|÷max     IG
hr_mean    95,09     +0,67       0,47      -0,0544
nni_min   564,00     -0,80       1,00      +0,0543
nni_mean  636,98     -0,84       0,59      +0,0364
nn50       16,00     +1,63       0,30      +0,0595
```

## FP e FN: o par que prova o problema do limiar

**FP** (S2_baseline): hr_mean **81,98 bpm** (na média do treino), sdnn 104,36
(+1,61) e rmssd 66,76 (+0,87) **altos** — perfil calmo — mas nni_min 581 (−0,67).
Predição STRESS 84,8%.

**FN** (S9_stress): hr_mean **83,00 bpm**, nn50 = 2 (−0,84), pnn50 5,41 (−0,80),
rmssd 25,93 (−0,63) — parassimpático suprimido, o que a Tabela 6 associa a
stress. Predição NO STRESS **p=0,423**, a 7,7 pontos do limiar.

**Os dois têm frequência cardíaca a 1 bpm de distância, rótulos opostos, sujeitos
diferentes, e o modelo erra nos dois em direções contrárias.** É o argumento da
calibração por sujeito num único exemplo verificável.

O S9 aparece nos dois: o modelo acerta o baseline com p=0,17 e falha a janela de
stress com p=0,423.

## A baseline do IG não é neutra: f(0) = 0,8729

O modelo prevê **STRESS com 87,3%** para uma janela com todas as features na
média do treino. Consequências:

1. Explica a completude observada: a soma das atribuições IG é sempre
   `predição − 0,873` (+0,119 no TP, −0,697 no TN, −0,023 no FP, −0,446 no FN).
2. Explica porque as atribuições IG de janelas não-stress são quase todas
   negativas — têm de empurrar para baixo a partir de 0,87.
3. Vem dos pesos de classe (balanced × 1,2 na classe stress) num dataset com
   35,6% de stress.

É uma limitação a declarar na Secção 4.4, não um bug. Torna concreto o argumento
de que o IG depende da baseline.

## Protocolos das métricas (para escrever a 4.4.3)

- **Fidelidade** = exatidão original − exatidão com as **2** features de maior
  atribuição média postas a zero (= média do treino). 0,8307 → 0,6349, queda de
  **0,1958**.
- **Estabilidade** = correlação de Spearman entre a atribuição de x e de x+ruído
  (σ = 0,05 em unidades de desvio-padrão), 100 amostras × 3 repetições, seed 42.
  Valor **0,806**.

## Figuras: duas redundâncias no código (não afetam a tese)

`attention_true_positive.png` e `gradient_stress_example.png` mostram **a mesma
grandeza na mesma amostra** (primeiro TP do teste), só com estilos diferentes.
Idem para o par negativo. A tese usa as figuras `*_high_conf_*` dos estudos de
caso, não estas, por isso não há nada a cortar no texto — os ficheiros ficam sem
uso em `images/`.

---

# Figuras de explicabilidade regeradas (2026-09-22)

**Os rótulos dos eixos já estavam corretos** — "Gradient Magnitude (Normalised)",
"Feature Importance", "Gradient Attribution: Stress vs No Stress". A nota
anterior deste ficheiro, de que os eixos ainda diziam "Attention Score", estava
desatualizada. O que faltava eram os **nomes dos ficheiros**.

Alterado no `Explainability.py` e no `run_explainability.py`: os `save_path`
passam de `attention_*` / `*_attention.png` para `gradient_*` /
`*_gradient.png`. Regerado com `explain_wesad.py --load` (2 min, sem re-treinar).
Matriz de confusão, fidelidade e estabilidade **idênticas** às registadas
(107/15/17/50 · 0,1958 · 0,8060), o que confirma que só mudaram nomes.

Ficheiros antigos `attention_*.png` e `*_attention.png` apagados do repositório.

## Figura nova: atribuição média por classe

`gradient_class_means.py` → `results/gradient_by_class_mean.png`.

A figura `gradient_by_class.png` (do `compare_attention_patterns`) compara **uma**
amostra de stress com **uma** de não-stress, não médias — a legenda diz "Stress
Sample". Não sustenta nenhuma afirmação sobre a classe e duplica os estudos de
caso. A nova calcula a média de |∂ŷ/∂x| sobre as 189 janelas de teste, por
classe real, com erro padrão:

| Feature | Stress (n=67) | Não-stress (n=122) |
|---|---|---|
| nni_mean | 1,073 ± 0,156 | 0,960 ± 0,184 |
| nni_min | **0,583 ± 0,086** | 0,385 ± 0,057 |
| nn50 | **0,559 ± 0,096** | 0,341 ± 0,051 |
| hr_mean | 0,494 ± 0,079 | 0,538 ± 0,105 |
| nni_max | 0,468 ± 0,083 | 0,331 ± 0,053 |
| pnn50 | **0,467 ± 0,075** | 0,300 ± 0,046 |
| sdnn | **0,459 ± 0,071** | 0,263 ± 0,042 |
| rmssd | 0,291 ± 0,052 | 0,304 ± 0,054 |

Barras de erro **sem sobreposição** em nni_min, nn50, pnn50 e sdnn — nessas o
modelo é mais sensível nas janelas de stress. Sobrepostas em hr_mean, nni_mean e
rmssd. São erros padrão, não um teste: escrever "as barras não se sobrepõem",
não "é significativo".

---

# Protocolo personalizado (2026-09-23): o que falta é o limiar, não os dados

`Classification Models/personalised_protocol.py` → `results_personalised.csv`.

Três condições sobre **as mesmas janelas de teste** de cada sujeito:

| | Modelo | Limiar |
|---|---|---|
| **A** | treinado nos outros 14 sujeitos | fixo 0,5 |
| **B** | o mesmo de A, **sem re-treino** | escolhido nas janelas de calibração do sujeito (índice de Youden) |
| **C** | re-treinado com os outros 14 **mais** a calibração do sujeito | fixo 0,5 |

Divisão dentro do sujeito **temporal** e por classe: as primeiras 50% das
janelas de cada classe, pela ordem do ficheiro, para calibração; as restantes
para teste. Calibração de 30 a 35 janelas, ou seja **15 a 17 minutos** de ECG
rotulado por pessoa. Teste de 30 a 36 janelas.

## Resultados (15 sujeitos, média ± SD entre sujeitos, mínimo entre parênteses)

| Modelo | Condição | Exatidão | Especificidade | Sensibilidade |
|---|---|---|---|---|
| **Model D** | A população | 80,87 ± 14,73 (46,9) | 81,79 ± 22,39 (**15,0**) | 78,18 ± 32,56 (**0,0**) |
| | **B limiar** | **85,93 ± 10,74 (63,9)** | **88,41 ± 11,67 (68,4)** | 80,76 ± 25,87 (18,2) |
| | C re-treino | 81,35 ± 16,09 (43,8) | 80,91 ± 22,98 (10,0) | 81,21 ± 31,39 (0,0) |
| **LogReg** | A população | 82,16 ± 17,19 (37,5) | 87,75 ± 26,19 (**0,0**) | 70,91 ± 36,72 (0,0) |
| | **B limiar** | **87,89 ± 11,62 (61,1)** | **91,81 ± 11,37 (64,0)** | **80,81 ± 19,53 (45,5)** |
| | C re-treino | 82,59 ± 16,70 (40,6) | 87,75 ± 24,13 (5,0) | 72,12 ± 35,62 (0,0) |
| **RF** | A população | 80,21 ± 16,10 (43,8) | 85,16 ± 23,83 (10,0) | 70,30 ± 34,95 (0,0) |
| | B limiar | 83,61 ± 14,56 (56,2) | 91,31 ± 11,76 (65,0) | 69,44 ± 31,28 (18,2) |
| | C re-treino | 84,02 ± 12,25 (62,5) | 89,45 ± 16,47 (40,0) | 73,33 ± 33,40 (9,1) |

## O que é robusto e o que não é

**Não significativo nas médias.** B − A na exatidão: +5,07 pp (Model D, p=0,173;
Wilcoxon p=0,126), +5,73 pp (LogReg, p=0,183), +3,41 pp (RF, p=0,060). Com 15
sujeitos e SD das diferenças de 13 a 16 pp, não há potência. **Não escrever que
a calibração melhora significativamente a exatidão média.**

**Robusto: o chão sobe e a dispersão colapsa.** É aqui que está o resultado.

```
Model D   exatidao minima entre sujeitos   46,9 -> 63,9
          especificidade minima            15,0 -> 68,4
          SD da especificidade             22,4 -> 11,7
LogReg    especificidade minima             0,0 -> 64,0
          SD da sensibilidade              36,7 -> 19,5
sujeitos com exatidao < 70%    Model D 4 -> 1 · LogReg 3 -> 1
sujeitos com especificidade < 50%   1 -> 0 nos tres modelos
```

**Re-treinar não serve de nada.** C − A no Model D: +0,48 pp (p=0,693); na
LogReg: +0,42 pp (p=0,438). Só a RF ganha algo (+3,82 pp, p=0,020) e mesmo assim
não mais do que o limiar. **As mesmas janelas do sujeito valem muito mais para
escolher o limiar do que para treinar.**

**Os limiares ótimos variam enormemente entre pessoas**: mediana 0,54 e gama
[0,02, 1,00] no Model D. É a medição directa do problema.

## O caso que resume tudo: S10

```
              exatidao   especificidade
A populacao      46,9%        15,0%
B limiar         90,6%       100,0%     <- sem re-treinar
C re-treino      43,8%        10,0%     <- com os dados dele no treino
```

Com o limiar do próprio sujeito, e sem tocar no modelo, o pior sujeito da
amostra passa de inutilizável a 90,6%. Com re-treino, nada.

## Limitações a declarar

1. **n=15.** As diferenças médias não atingem significância.
2. **A calibração e o teste vêm das mesmas duas sessões de gravação** (divisão
   temporal dentro de cada condição). Num uso real a calibração seria noutro
   dia, portanto estes valores são optimistas quanto à estabilidade do limiar
   ao longo do tempo. **Tem de ser dito.**
3. A calibração exige janelas rotuladas das duas classes: ~15 min de ECG com
   rótulo por utilizador, incluindo um episódio de stress.
4. O ganho em sensibilidade é menor e mais ruidoso que o de especificidade.

---

# Decisões de âmbito (2026-09-23)

Registado para não se voltar a propor o que já foi decidido:

1. **O protocolo personalizado fica fora da tese.** Os números estão acima e são
   válidos; a autora decidiu não os incluir porque o ganho médio não é
   significativo. Se a discussão com a supervisora reabrir o tema, o resultado
   está medido e o script está no repositório — não é preciso correr nada.
2. **A definição de ROC/AUC não entra no Capítulo 2.** Outras teses do mesmo
   curso não o fazem. Entra na Secção 3.6.1, com as outras métricas. As entradas
   `fawcett2006roc` e `hanley1982roc` (verificadas no Crossref) e os acrónimos
   ROC e AUC fazem falta de qualquer forma.
3. **A Secção 4.6 (discussão) fica em suspenso** até estar decidido qual é o
   modelo do sistema. Os pontos de decisão vão à supervisora.
4. **Não existem acrónimos para API, HTTP nem CPU** — ou se criam, ou se escreve
   em texto simples nas secções 3.8, 4.5 e 3.9.

---

# Correção: o classificador maioritário no WESAD (2026-09-23)

As tabelas do WESAD deste ficheiro davam **especificidade 0,00** ao classificador
maioritário. Está errado, e o erro veio de copiar a linha das tabelas do conjunto
combinado.

No conjunto combinado e no SWELL a classe maioritária é **stress** (56,4% e
60,7%), por isso prever sempre a maioria dá sensibilidade 100% e especificidade
0%. **No WESAD a classe maioritária é não-stress** (64,4%), por isso dá
**sensibilidade 0,00% e especificidade 100,00%**.

Linhas corrigidas acima. O valor da exatidão (64,55 aleatório · 64,39 GroupKFold ·
64,35 LOSO) estava certo em todas.

**A tabela da Secção 4.2 do rascunho tem o mesmo erro** e tem de ser corrigida:
`Majority classifier & 64.55 & 0.00 & 0.500` na coluna de especificidade passa a
`100.00`. Convém acrescentar a explicação, porque uma especificidade de 100% sem
contexto parece um bom resultado: é o preço de nunca prever stress.

---

# Model G escolhido; explicabilidade refeita sobre ele (2026-09-24)

**Decisão (autora + orientadora, 2026-09-24): o modelo do sistema é o Model G**
(Dual-Input CNN = "Conv1D série+resumo" nas secções acima). Desbloqueia a 4.6 e o
`\sysname`. A explicabilidade, a matriz de confusão, os estudos de caso e a
latência do Cap. 4 referiam-se ao Model D e têm de passar para o G.

Script: `Classification Models/explain_model_g.py` (treina ~2 min, `--load`
reutiliza `saved_models/wesad_model_G.weights.h5`). Saída em
`results/model_G/`. Mesma partição do Model D (seed 42, as mesmas 189 janelas).

## Matriz de confusão (seed 42, N=189)

```
                 prev nao-stress  prev stress
real nao-stress        114             8
real stress             11            56
acc 89,95  prec 87,50  rec 83,58  spec 93,44  AUC 0,961  32 929 parametros
```
Model D na mesma partição: 107/15/17/50, acc 83,07, spec 87,70. Metade dos
falsos alarmes (8 vs 15) e menos 6 falhas.

## A máscara transporta a frequência cardíaca — declarar

O comprimento da máscara é o número de batimentos em 30 s. **Correlação 0,98
com hr_mean; AUC sozinha 0,882** (hr_mean 0,884). Com série e resumo ambos na
baseline o modelo ainda tem AUC 0,882 — exactamente o número de batimentos.
Os dois métodos mantêm a máscara fixa, portanto **esta via nunca é atribuída**.
É por isso que o hr_mean aparece baixo na importância do G (0,528): a
informação dele entra também pela máscara.

## Ablação por ramo (acc / AUC)

```
intacto                               89,95  0,961
serie = media da propria janela       77,78  0,883   sem dinamica batimento a batimento
serie na baseline                     62,96  0,727
resumo na baseline                    82,54  0,961   AUC igual: o resumo so mexe no limiar
ambos na baseline                     35,45  0,882   = so a mascara (n.o de batimentos)
```
**O resultado mais forte:** achatar a série para a média da própria janela
(mantém nível e comprimento, tira a dinâmica) custa 12,2 pp e 0,078 de AUC. É
evidência directa de que a ordem dos batimentos importa — a razão de ser do G.
Tirar o resumo não muda a AUC: a ordenação vem da série.

## Integrated gradients

- Baseline (série a 0 nas posições reais, mesma máscara, resumo a 0):
  **f(baseline) = 0,959** (0,956–0,962). Pior que o 0,873 do Model D.
- Completude: erro mediano 0,007, máximo 0,147 (50 passos).
- Fração |IG série| / total: média 0,545; stress 0,430; não-stress 0,608.
- |IG| médio no resumo: nn50 0,223 · nni_max 0,207 · pnn50 0,112 · rmssd 0,107 ·
  hr_mean 0,068 · nni_mean 0,065 · sdnn 0,061 · nni_min 0,051.

## Importância global |dy/dF| (÷ máximo)

nni_max 1,000 · rmssd 0,834 · nn50 0,793 · hr_mean 0,528 · pnn50 0,466 ·
nni_mean 0,439 · nni_min 0,292 · sdnn 0,248. Top-3 = 57,1%; razão 4,04.
**Muda por completo face ao D** (nni_mean 1,000 · hr_mean 0,522 · nni_min 0,455):
o G pesa marcadores vagais/dispersão no resumo porque o nível já vem da série e
da máscara. Por classe: **todas as barras de erro se sobrepõem** (no D não se
sobrepunham em 4).

## Fidelidade e estabilidade (protocolos do D)

- Fidelidade (top-2 resumo = rmssd, nni_max a zero): 0,8995 → 0,8730,
  **queda 0,0265** (D: 0,1958). Esperado — o resumo não é onde está a ordenação;
  a medida certa para o G é a ablação por ramo.
- Estabilidade: resumo **0,676** (SD 0,34), série 0,728 (SD 0,19). D: 0,806.

## Estudos de caso (critérios do D: primeiro de cada tipo)

| Caso | Índice | Ficheiro | p | IG série | IG resumo |
|---|---|---|---|---|---|
| TP p>0,8 | 6 | S7_stress | 1,000 | −0,052 | +0,099 |
| TN p<0,2 | 0 | **S2_baseline** | 0,083 | −0,665 | −0,189 |
| FP | 1 | **S13_baseline** | 0,523 | −0,298 | −0,119 |
| FN | 23 | S9_stress | 0,345 | −0,031 | −0,570 |

Contagens: 47 TP p>0,8 · 87 TN p<0,2 · 8 FP · 11 FN.

- **O FP do D (S2_baseline, p=0,848) é agora um TN confiante (p=0,083)**, e
  78% da atribuição vem da série. A janela "calma com uma aceleração
  transitória" que o D errava, o G acerta lendo a série.
- O FP novo (S13_baseline) tem nn50=0, pnn50=0, rmssd 12,86 (−1,11), hr 92,6 —
  perfil de stress num baseline, p=0,523, **a 0,023 do limiar**.
- **O FN é o mesmo do D (S9_stress)**, p=0,345. Mesmo argumento: hr 83,00, na
  média da população; nn50 domina (IG −0,48).
- O par FP/FN do D (1 bpm de distância) desaparece; o novo par é S13 com
  92,6 bpm rotulado não-stress vs S9 com 83,0 bpm rotulado stress — o argumento
  do limiar/basal por sujeito mantém-se, mais forte (o não-stress tem a FC mais alta).

## Latência

`model.predict`, 1 janela, 1000 corridas: **53,63 ms** (D: 48,18 ms, mesmo método).

## Correções verificadas nas CSVs (2026-09-24)

- **O 0,930 do primeiro sujeito em LOSO é do Model F** (Conv1D só série), não
  do G. O rascunho da 4.2 diz "model G 0.930" — errado. Model G no 1.º sujeito:
  AUC 0,875, especificidade 5,0%, exatidão 40,6%.
- Gamas por sujeito em LOSO do Model G: exatidão 40,6–100%, especificidade
  5–100%. O 37,5% / 0% vem da LogReg, do Model F e dos Modelos A/C.
- Em LOSO a AUC mais baixa dos nove treinados é a RF (0,902); o G (0,925) é o
  último dos sete modelos neuronais.
- GroupKFold, exatidão: a mais alta é a **LogReg 82,62**, não o Model C 82,59
  (o negrito da Tabela está errado).
- Queda de exatidão aleatório → GroupKFold: 0,9 pp (C) a 8,2 pp (G), não "5 a 8".
- Model G no conjunto combinado, split aleatório: especificidade 62,28%.

## Testes emparelhados COMPLETOS no WESAD e correção de Holm (2026-09-24)

**O "nenhuma comparação em LOSO é significativa" só era verdade para os 4 pares
testados.** Com os 36 pares (9 configurações treinadas) por protocolo e métrica:

- GroupKFold: 5 com p<0,05 sem correção (RF abaixo de C, E, F, G em AUC; C−D AUC).
- LOSO: 9 com p<0,05 sem correção (RF abaixo de A e C em AUC; A abaixo de C, E,
  F, G em exatidão; B abaixo de F e G; E abaixo de G, este último −3,33 pp p=0,027).
- **Nenhum sobrevive a Bonferroni/Holm** (menor p corrigido 0,143, RF−G AUC em
  GroupKFold; em LOSO 0,659).

Formulação: "nenhuma das 36 diferenças par a par sobrevive à correção para
comparações múltiplas", não "nenhuma é significativa".

**Split aleatório, família planeada = G (ou F) contra os 7 modelos de resumo, Holm:**

```
G acc: A .002* B .002* C .003* D .012* E .003* LogReg .029* RF .063
G AUC: A .027* B .046* C .055  D .027* E .055  LogReg .055  RF .060
F acc: A .003* B .003* C .006* D .119  E .035* LogReg .119  RF .197
F AUC: nenhum
```
- A vantagem do G em **exatidão** sobrevive contra os seis modelos de resumo
  exceto a RF. Em **AUC** só contra A, B e D.
- **F − D (4,13 pp, p=0,040) não sobrevive** (0,119). O argumento "a vantagem
  vem da representação, não da capacidade" deve assentar no G vs D (Holm 0,012;
  32 929 vs 218 625 parâmetros), não no F vs D.
- Não corrigidos: G−A 9,95 pp p=.0002; G−B 9,95 p=.0003; G−C 6,35 p=.0008.

## Outros factos verificados

- Primeiro fold de LOSO = **S10** (ordem lexicográfica: S10, S11, S13…, S9).
  64 janelas, 37,5% stress. LogReg e Model F com especificidade 0% e exatidão
  37,5% ⇒ **todas** as janelas classificadas como stress (não "quase todas").
- SWELL só foi avaliado por sujeito em GroupKFold (não há LOSO): AUC 0,523–0,607.
- A matriz de confusão do D (83,07%) e a do G (89,95%) vêm de re-treinos com
  seed 42; as linhas seed 42 das CSVs são 82,54% (D) e 89,42% (G).
- O label `sec:agent_results` é usado nas l. 13, 206 e 276 do Cap. 4 e não
  existe; o label da secção é `sec:feedback_agent`.

# Sentido dos efeitos aprendidos pelo Model G (2026-09-26) — IMPORTANTE

Duas vias independentes sobre as 189 janelas de teste (seed 42, pesos
`saved_models/wesad_model_G.weights.h5`, script
`Classification Models/model_g_directions.py` — reproduz estes valores ao decimal):

| feature | d WESAD | IG vs valor (Spearman) | perturbação +0,5 SD só no resumo | leitura |
|---|---|---|---|---|
| hr_mean | +1,54 | −0,54 | −0,022 | **contrária** nas duas |
| nni_mean | −1,64 | +0,60 | −0,016 | ambígua |
| nni_min | −1,80 | −0,53 | −0,001 | fisiológica / nula |
| nni_max | −1,38 | −0,91 | **−0,105** | fisiológica, a mais forte |
| sdnn | −0,45 | +0,54 | −0,002 | ambígua |
| rmssd | −0,85 | +0,49 | **+0,073** | **contrária** nas duas |
| nn50 | −0,68 | +0,82 | +0,023 | **contrária** nas duas |
| pnn50 | −0,88 | −0,54 | −0,005 | fisiológica |

(d aqui calculado no conjunto de teste; o da Tabela 4.1 é sobre as 941 janelas.)

**O modelo inteiro responde no sentido fisiológico:** intervalos ×0,9 com série
e resumo recalculados de forma coerente → P(stress) +0,138 em média, sobe em
**87%** das janelas. Série sozinha −0,5 SD (resumo fixo) → +0,115. Por
batimento, Spearman(intervalo, IG) = **−0,12** (p=1,7e−23): intervalos longos
empurram para não-stress.

**Leitura:** as 8 features resumo são funções determinísticas da série que o G
também recebe (e o nº de batimentos = nn50/pnn50 exactamente). O ramo resumo
aprende correcções condicionais, não efeitos marginais. **As atribuições por
feature do G não podem ser lidas uma a uma como "a feature X indica stress".**
Leitura válida: ao nível do modelo inteiro, dos ramos (ablação) e da série.
Correlações: nn50–pnn50 0,974; nn50–rmssd 0,913; sdnn–rmssd 0,799.

É um custo da escolha do G face ao D em explicabilidade. Tem de ir para a 4.4,
4.6, contribuições da Introdução e Abstract.

Probabilidades exactas dos casos: TP S7 1,0000 · TN S2 0,0831 · FP S13 0,5233 ·
FN S9 0,3453.

# Verificações da 4.1 / 4.1.1 (2026-09-26)

- Tabela de d de Cohen: reproduz com desvio-padrão agregado ponderado
  ((n1−1)s1² + (n0−1)s0²)/(n1+n0−2). ✓
- Tabela de transferência: ✓ as 10 células.
- **"Dataset-only 62,58% exceeds Models A, B and D" está errado: excede os
  quatro** (C = 61,95%). A nota do SKILL.md ("três das quatro") vinha do 61,36%.
- "within 9 pp of the majority" errado: G 65,65 − 56,35 = **9,30 pp**.
- RF − A..D no combinado: **1,90 a 3,43 pp** (texto diz 1,77–3,30, calculado
  com a RF antiga de 63,72). Os 8 testes sobrevivem a Holm (p_adj ≤ 0,038).
- C − D combinado: SD da diferença 1,26 pp, p=0,241 ✓; WESAD 2,45 pp ✓.
- G no combinado, split aleatório: 65,65%, especificidade 62,28%.

# Validação do zip de 26/09 contra as notas da supervisora (2026-09-26)

Aplicado: quase tudo. 4.4 e 4.6 idênticas aos rascunhos.
**Por fazer no texto:**
- 3.6.1: frase duplicada e contraditória do FN ainda lá; "both is \"negative\"".
- 3.4: falta a regra de exclusão (>20% corrigido) e a contagem por classe
  (ponto 6 da supervisora). Taxas de janelas excluídas: não-stress 442/2811 =
  15,7%; stress 717/3778 = 19,0% (totais finais 2369 / 3061).
- 4.1: parágrafo do autorrelato não entrou (o texto anuncia 4 linhas).
- 4.5: parágrafo da latência não entrou (4.6 e Cap. 5 citam 53,63 ms).
- 25 citações vivas: 20 \cite (→ \citep) e 5 \citet precedidos do nome
  (Xia, Jadav e Chauhan, Moser, Xiong, Schmidt) — o PDF repete os autores.
- 3.5: "for interpretable temporal modeling" (resto da confusão atenção).
- 4.2: "Model G has 32,929 parameters a sixth of the size..." (vírgula, "less than").
- Resumo PT: ainda "representa uma abordagem promissora" (anotação p.9/p.10).
**Pedido da supervisora nunca feito:** PR-AUC e intervalos de confiança (ponto 10).
Custo: WESAD A–D 5 seeds = 6,8 min; probabilidades não foram guardadas nas CSVs.
Verificado e OK: capa, Doutora, \part, Cap. 6, keywords, hifenização
(\babelhyphenation), Tabelas 1 e 2, typos Cap. 2, 37 sujeitos (+ razão dos 22/25),
cortes, contrações, só figuras do G em uso, refs e cites resolvem.

# Decisão (2026-09-26): os questionários do SWELL NÃO entram na tese

A autora não quer falar do autorrelato. A 4.1 passa a "três linhas de
evidência" (efeito por dataset, transferência, desenho do SWELL); tirar também
das Contribuições (Intro) e do 2.º parágrafo das Conclusões. Não repropor.

# Corrida PR-AUC (2026-09-26)

Scripts alterados (backups `*.bak_prauc`): `retrain_v5.py`, `train_sequence.py`
(coluna `prauc`), `wesad_baselines.py` (`prauc`, `--out`, especificidade do
majority = 100). Runner `run_pr.sh {abcd|ef|g}`, logs em `logs_pr/`.
Saídas `results_wesad_pr_{random,group,loso}_{ABCD,summary,seq,both}.csv` e
`results_wesad_pr_baselines.csv`. As baselines são deterministas: reproduzem
exatamente acc/spec/AUC anteriores.

# Corrida PR-AUC concluída (2026-09-26, tarde) — SUBSTITUI os números WESAD anteriores

CSVs `results_wesad_pr_*` (o D do LOSO foi refeito inteiro numa invocação,
`results_wesad_pr_loso_D.csv`, e fundido). Tabelas e apêndice de ICs:
`Classification Models/results/ci_tables.tex` (cópia em `drafts/`), gerados por
`confidence_intervals.py`; números do texto por `pr_report.py`. Substituições no
texto: `drafts/correcoes_prauc_ic.md` (27 itens).

Split aleatório (5 seeds): G 89,95 ± 1,50 · AUC 0,959 · PR-AUC 0,933. F 89,21.
E 85,19 · C 83,92 · D 83,49 (spec 84,26) · B 82,75 · A 79,37. LogReg/RF/majority
iguais aos anteriores (deterministas). Majority PR-AUC = 0,354 (prevalência).
Holm (família: G vs 7 modelos de resumo): exatidão sobrevive vs A–E
(p_adj ≤ 0,028), NÃO vs LogReg (+4,97, 0,076) nem RF (+3,18, 0,076).
AUC sobrevive vs os 6 exceto RF, C/E/LogReg a 0,0499. PR-AUC: nada sobrevive.
F − D agora sobrevive (+5,71, adj 0,021). C à frente de D (0,42 pp) — ordem
inverteu face à corrida anterior.
Por sujeito: 36 pares × {acc, AUC, PR-AUC} × {GKF, LOSO}: nenhum sobrevive.
GKF AUC 0,874–0,917 (G 1.º, LogReg 3.º); LOSO 0,902–0,951 (LogReg 1.º, G 0,919,
último dos neuronais, SD 0,144). LOSO G: acc 39,1–98,4, spec 2,5–100.
S10: LogReg 0,901 / F 0,923, ambos spec 0.

**Ruído entre invocações (mesmo código, mesmas seeds):** B +2,86 pp, D −1,59,
E +0,95, A −0,53, C +0,42, F 0,00, G +0,11. Substitui o "0,90 pp".

# Avaliação externa (2026-09-26, noite): o que se confirmou nos dados

Um agente independente avaliou a tese. Verificações feitas contra o zip V5 de
26/09 e as CSVs. Scripts no repositório (não no scratchpad).

## 1. Teste t corrigido de Nadeau & Bengio — `corrected_ttest.py`

As 5 seeds do split aleatório são particoes sobrepostas dos mesmos dados; o
t-test emparelhado habitual subestima a variância. Correcção:
var × (1/J + n_teste/n_treino), J=5, rácio 0,25 → t desce 1,5×. Não se aplica
a GroupKFold/LOSO (testes disjuntos). O script reproduz ao decimal os p de Holm
da tese (coluna "Holm") e acrescenta os corrigidos:

```
G - x, exatidao   Holm habitual     Holm Nadeau-Bengio
  A  +10,58         0,018 *            0,077
  B   +7,20         0,024 *            0,097
  C   +6,03         0,024 *            0,097
  D   +6,46         0,028 *            0,105
  E   +4,76         0,010 *            0,044 *    <- o unico que sobrevive
  LogReg +4,97      0,076              0,223
  RF     +3,18      0,076              0,223
G AUC: nenhum sobrevive com a correcao (Holm NB 0,086-0,180).
F: so F - A em exatidao sobrevive (0,028).
F2 - F1 (Model D): +0,53 pp, IC95% [-2,59; +3,65] habitual, [-4,15; +5,20] NB.
```

**G vs E é a comparação limpa** (mesma receita de treino: lr 5e-4, batch 64,
clipnorm 1,0; mesmas 8 features; a única diferença é a série). A–D têm outra
receita (lr 1e-4, batch 32, clipnorm e épocas diferentes). O argumento
"representação > arquitetura" deve assentar no G − E.

## 2. Teste de permutação da ordem — `model_g_permutation.py`

A ablação da tese (série = média da janela) tira a variabilidade toda, não só a
ordem, e cria entrada fora da distribuição. Baralhar os intervalos dentro de
cada janela mantém média/min/max/sdnn/nº de batimentos. 50 permutações, pesos
V5 seed 42, N=189:

```
intacto      89,95   AUC 0,961
flat         77,78   0,883    (-12,17 pp / -0,079)  = ablacao da tese
perm         81,69   0,823    (-8,25 / -0,138)       serie baralhada, resumo intacto
perm+rec     79,09   0,858    (-10,86 / -0,104)      serie baralhada E resumo recalculado
```
**O G usa mesmo a ordem** — o teste limpo (perm+rec) custa 10,86 pp e 0,104 de
AUC. O que está errado na tese é só a frase "those statistics do not depend on
the order" (Applications l.272 e 4.6): rmssd, nn50 e pnn50 dependem da ordem.

## 3. BUG: durações erradas no `time_duration.csv` → janelas que não têm 30 s

O `split_ecg()` do ETL deriva o tamanho da janela da duração listada
(`30 / (duracao / n_amostras)`). Comparando com amostras/fs de cada ficheiro
(`Data Processing/fix_durations.py`):

| Ficheiro | Listada | Real | Janela efectiva |
|---|---|---|---|
| S2_baseline_ECG_values.csv | 1518,3 s | 1144,0 s | **22,9 s** (50 janelas; devia dar 38) |
| S5_stress_I.csv | 1595,0 | 2940,8 | 55,3 s (trocada com a do stress_T) |
| S5_stress_T.csv | 2940,0 | 1595,4 | 16,3 s |
| S12_stress_T.csv | 2126,0 | 918,8 | **13,0 s** (a linha que foi acrescentada à mão) |
| S3_stress_T.csv | 2142,0 | 2322,7 | 32,5 s |

Restantes ficheiros do WESAD: −2% a +7% (janelas de 28,0 a 30,7 s). SWELL: exacto.

Sintoma que o avaliador apanhou: a janela TN do estudo de caso (S2_baseline,
29 intervalos, 81,98 bpm) cobre 21,6 s. **As 50 janelas do S2_baseline cobrem
20,7–22,9 s**; nenhum outro ficheiro do WESAD tem janelas < 26 s. É 8% das
janelas não-stress do WESAD, e o G lê o nº de batimentos (AUC 0,882 sozinho):
as janelas encurtadas têm menos batimentos → parecem mais "calmas" → correlacionado
com o rótulo. O TN do estudo de caso da 4.4 e o texto "one beat among 29" vêm daqui.
A Tabela A.1 do apêndice mostra os 1518,312 s errados.

**Exclusões por dataset (V5):** WESAD 0 (941 = Σ floor(duração/30)); as 1159
exclusões são todas do SWELL.

## 4. V6 (em curso, 2026-09-26)

`Data Processing/time_duration_v6.csv` = amostras/fs para os 96 ficheiros.
`Classification Models/run_v6.sh {etl_wesad|etl_all|base|abcd|ef|g}`.
WESAD V6: **907 janelas** (581 não-stress, 326 stress = 35,94%), 0 exclusões,
taxa de correção 0,36% / 0,99%. Saídas `results_wesad_v6_*.csv`, logs em
`logs_v6/`. Ficheiros: `/home/eduarda/Desktop/ProcessedECGData_v6_wesad.csv`,
`NNSeries_v6_wesad.npz`; completo (SWELL incluído, ~2 h) em
`ProcessedECGData_v6.csv` / `NNSeries_v6.npz`.

## 5. Outras verificações do avaliador

- Seed 42 = média de 5 corridas (89,95 / 93,44): **coincidência aritmética**
  (170/189 = 850/945; 114/122 = 570/610). Não foi copiado.
- Transferência: `results_transfer.csv` tem **3 seeds** nos neuronais e 1 corrida
  na LogReg/RF. Com n=3, três dos dez ICs passam 0,5 (limites 0,86/0,67/0,83).
- Perdas aleatório→GroupKFold (V5): G −8,64 · F −7,38 · **B −7,05 · RF −6,76** ·
  E −5,57 · A −3,93 · LogReg −2,35 · C −1,44 · D −0,69. O avaliador omitiu B e RF;
  o padrão é "modelos flexíveis", não "a série".
- O 0,909 de identificação do dataset vem da quantização 700 vs 2048 Hz, que não
  identifica sujeitos dentro do WESAD.
- A tese não define a tarefa do WESAD (baseline vs TSST, amusement excluído);
  a tarefa binária habitual (Schmidt 2018) põe o amusement no não-stress.
- Detetor de picos: `biosppy.signals.ecg.ecg()` usa o **Hamilton** por omissão;
  a tese diz só "BioSPPy, which contains several methods".
- Erros de compilação confirmados: `images/cnn.png` (é CNN.png),
  `\gls{TSST}` sem acrónimo, `ap:datasets` duplicado, índice e glossário vazios.

# V6 ADOTADA (2026-09-26, noite) — SUBSTITUI todos os números anteriores

Decisão da autora: passar a tese para a V6 (durações reais). Dados:
`/home/eduarda/Desktop/ProcessedECGData_v6.csv` + `NNSeries_v6.npz` (5409 janelas),
e `..._v6_wesad.*` (as 907 linhas do WESAD, idênticas às do ficheiro completo).
Runner: `Classification Models/run_v6.sh`. Relatórios: `logs_v6/pr_report_v6.txt`,
`logs_v6/dataset_analysis.log`, `logs_v6/explain_g.log`, `directions_g.log`,
`permutation_g.log`. Tabelas: `confidence_intervals.py --v6` → `results/ci_tables_v6.tex`.
Rascunhos entregues: `drafts/sec_4_{1,2,3,4,6}_v6.tex`,
`drafts/correcoes_avaliacao_externa_parte{1,2_v6}.md`, `drafts/appendix_tables_v6.tex`,
`drafts/figures_v6/`. Verificação: `drafts/simular_aplicacao.py <zip extraído>`.

## Corpus
5409 janelas (WESAD 907: 581/326, 35,94%; SWELL 4502: 1763/2739, 60,84%; total 56,66%).
Janelas de 30,00–31,18 s (mediana 30,21). 6523 antes da exclusão. Intervalos corrigidos
7,35% / 8,07%; excluídas 442 / 672 (15,9% / 18,0%), **todas do SWELL** (20,0% / 19,7%);
WESAD 0 exclusões, 0,36% / 0,99% de intervalos corrigidos. Gama 500–1200: 7,27% / 11,15%.
Durações: neutral 84 120,308 s (23,4 h), stress 112 942,165 s (31,4 h, +34,3%);
WESAD baseline 1140–1198 s, stress 615–725 s.

## 4.1
d de Cohen WESAD/SWELL: hr +1,73/−0,42 · nni_mean −1,69/+0,44 · nni_min −1,84/+0,32 ·
nni_max −1,40/+0,40 · sdnn −0,42/+0,22 · rmssd −0,74/+0,33 · nn50 −0,58/+0,37 ·
pnn50 −0,75/+0,42 · média |d| 1,14/0,36. FC WESAD 72,9→98,7 (+25,8); SWELL 76,7→72,1.
Sujeitos: WESAD 15/15 +20,78 (mediana); SWELL 22/22 −5,26. Filme excluído: −5,76, 18/20.
Blocos do SWELL (receita reconstruída, `dataset_analysis.py`): neutral +0,13 (p=0,78),
interrupções +2,24, pressão temporal +6,45 (p<0,001); 2.ª metade do neutral −4,46.
Domínio (RF, CV 5): 8 features 0,706; nº batimentos 0,623; série+resíduo 0,993
(a 700 Hz os intervalos em ms nunca acabam em 3, 6, 9).
Transferência (5 seeds nos neuronais): W→S LogReg 0,421 · RF 0,440 · E 0,421±0,003 ·
F 0,381±0,009 · G 0,433±0,050; S→W LogReg 0,185 · RF 0,385 · E 0,275±0,160 ·
F 0,170±0,046 · G 0,245±0,056. Todos < 0,5; limites superiores dos ICs ≤ 0,494.
SWELL por sujeito: LogReg 0,612 (p=0,015; Holm nos 5 → 0,075); RF 0,523; E 0,523;
F 0,549; G 0,541. RF só SWELL: aleatório 64,62 → GroupKFold 54,74.

## 4.2 (WESAD)
Aleatório: G 89,78±2,08 · spec 94,36 · AUC 0,955 · PR 0,940; **F 90,44 · AUC 0,965 · PR 0,952**;
E 87,25; RF 86,70; D 85,71; C 84,51; LogReg 82,64; B 81,21; A 79,78; majority 64,29.
G−F −0,66 pp (p=0,651). Holm habitual (G vs 7 de resumo): acc sobrevive vs A, B, C,
LogReg; NÃO vs D (0,052), E (0,056), RF (0,059). AUC vs todos menos RF; PR-AUC vs os 7.
F: acc, AUC e PR-AUC vs os 7 (p_adj ≤ 0,041). **Nadeau–Bengio + Holm**: G só vs A (acc);
F: PR-AUC vs os 7, acc vs LogReg e RF, AUC vs A.
GroupKFold AUC 0,878–0,919 (G 1.º, LogReg 2.º); LOSO 0,913–0,970 (G 1.º 0,970 — o
0,919 da V5 vinha de um fold instável, S5 0,450 → 0,984). 36 pares: nada sobrevive.
Quedas aleatório→GKF: F −11,41, G −9,64, E −6,67, B −6,04, RF −5,56, C −4,70,
D −3,91, A −1,74, LogReg +1,14. S10: os 9 com especificidade ≤ 7,7%.
Matriz de confusão G (retreino seed 42, N=182): 112/5/11/54, acc 91,21, prec 91,53,
rec 83,08, spec 95,73, AUC 0,965. Linha seed 42 da CSV: 90,66. D seed 42: 12 FP, 14 FN.

## 4.3 (F1/F2, Model D)
F2 84,95±2,29 / F1 85,82±2,25; F2−F1 −0,88 pp (p=0,398), IC [−3,46; +1,70]
(NB [−4,75; +2,99]); AUC −0,004 (p=0,178); sensibilidade −5,85 (p=0,030, Holm 0,15).

## 4.4 (G, pesos `saved_models/wesad_model_G_v6.weights.h5`)
Máscara: r=0,998 com hr_mean; AUC 0,889 (hr_mean 0,882). Ablação: intacto 91,21/0,965 ·
baralhado+recalc 78,02±1,09/0,923±0,009 · série=média 74,18/0,842 · série baseline
42,31/0,520 · resumo baseline 78,57/0,949 · ambos 35,71/0,889. IG: baseline 0,916
(0,912–0,919); completude 0,006; série 70,7% (não-stress 79,4, stress 55,2).
Importância: sdnn 1,000 · nn50 0,876 · nni_max 0,775 · nni_mean 0,627 · nni_min 0,618 ·
pnn50 0,577 · hr_mean 0,532 · rmssd 0,518 (top-3 48,0%). Não se sobrepõem por classe:
nni_max e pnn50. Sentidos: fisiológicos nni_max, nni_min; **contrários hr_mean,
nni_mean, nn50, pnn50** (4); ×0,9 → +0,197, sobe em 98%; ρ por batimento −0,18.
Fidelidade (sdnn, nn50) 10,44 pp = o maior dos 28 pares (média 4,24, SD 3,14).
Estabilidade resumo 0,729 (SD 0,302), série 0,681. Casos: TP S8_stress p=0,868 ·
TN S7_baseline 0,015 · FP S17_baseline 0,628 · FN S2_stress 0,382.
Latência: 145 ms medidos com a máquina carregada — **refazer com a máquina livre**.

## Combinado V6 (4.1.1) e latência — fecho da V6 (2026-09-26)
Aleatório: majority 56,65 · F 65,75 · G 65,53 · C 62,27 · RF 62,00 · E 61,66 · D 61,22 ·
LogReg 60,67 · B 60,59 · A 60,50 · regra "só sei o dataset" 62,44 (> A–D, RF, LogReg, E).
RF > A, B (acc e AUC) e > D (só AUC) com Holm; **não > C** (na V5 batia os quatro — o
atalho eram as janelas curtas do SWELL). A–D: spec 22,86–33,65, sens 82,32–89,46.
F > os 7 de resumo em AUC com Holm (adj ≤ 0,003). C−D: SD da diferença 1,17 (combinado)
vs 2,89 (WESAD), p=0,114. GroupKFold: AUC 0,511–0,599; 8/9 abaixo do majority (56,74;
exceção B 57,45); regra 61,44 > todos. Tabela `tab:pooled_subject` refeita em sec_4_1_v6.
Série NN: 21–73 intervalos (mediana 35; o 14 da V5 era o S12). Pesos WESAD 1,39/0,78 → 1,67.
**Latência G (máquina livre): 49,21 ms.** Tudo aplicado nos rascunhos; simulação sem problemas.

# Segunda avaliação externa e zip de 27/09 (13:47)

- Zip de 27/09: Parte 2 e secções 4.1–4.4/4.6 V6 aplicadas; Parte 1 a ~2/3. Tinha
  ainda as frases SWELL/WESAD anteriores à decisão da autora (repostas em
  `drafts/correcoes_finais.md`, secção A). O acrónimo TSST já está definido.
- **Documento a aplicar: `drafts/correcoes_finais.md`** (validado contra este zip;
  `drafts/simular_aplicacao.py <zip> correcoes_finais.md` → sem problemas).
  Estado por item: `drafts/estado_aplicacao.py <zip> <md>...`.
- **Exclusões do SWELL (folha `SWELL-KW - overview available data.xlsx`):** PP23 bloco
  neutro "recording too short"; PP11 sem gravação cardíaca no bloco 2 e "no valid HR
  extracted" nos outros; PP8 "no valid HR extracted" nos três blocos, o 3.º "skipped
  (very strange data)". Critério coerente: sem bloco com FC válida ou sem neutro.
  Os ficheiros do PP8 e PP11 existem em `Tese 2026/Datasets` mas nunca estiveram no
  `time_duration.csv`.
- Hernando et al. 2018 (Apple Watch): RR do sensor ótico (PPG) via app Breathe, não ECG.
- Hwang et al. 2018 (Deep ECGNet, `article9`): CNN + 2 LSTM, ECG em bruto de 10 s,
  87,39% / 73,96%. Benchekroun et al. 2023 (`newArtc4`, Sensors 23(4):1807): LogReg e
  RF, LOSO dentro de cada dataset e entre datasets; RF F1 61% entre datasets.
- Figuras G: título tinha "N=189" fixo no `explain_model_g.py`; corrigido (N={len(yte)}),
  regeneradas com os mesmos pesos V6 → `drafts/figures_v6/`.
- Latência do G varia entre medições: 49,21 ms (máquina livre, a da tese) e 54,38 ms.
- G − RF em AUC (V6, aleatório): 0,0145 → "0.015".

# Calibração por pessoa na V6 (2026-09-27) — ENTRA na tese (4.6.3)

Decisão revertida pela autora: a calibração passa a estar no texto. Script
`personalised_protocol.py --models G,LogReg --scaler standard` (dados V6_wesad) →
`results_v6_personalised.csv`. LOSO; 1.ª metade de cada condição (29–32 janelas,
~15 min) calibra, 2.ª testa. Condições: A limiar 0,5 · B limiar de Youden na
calibração, sem retreino · C retreinado com a calibração.

| | A | B | C |
|---|---|---|---|
| **G** exatidão | 82,08 ± 14,93 | **91,12 ± 7,41** | 90,22 ± 10,59 |
| G pior sujeito | 48,4 | **73,3** | 63,3 |
| G espec. mínima | 21,1 | 68,4 | 78,9 |
| G sujeitos < 70% | 3 | **0** | 1 |
| LogReg exatidão | 81,40 ± 18,32 | 85,92 ± 16,46 | 81,84 ± 18,19 |
| LogReg pior / espec. mín. | 38,7 / 0,0 | 37,9 / 31,6 | 38,7 / 0,0 |

G: B−A +9,04 pp, p=0,014 (Wilcoxon 0,018), Holm (2 comparações) 0,028; C−A +8,14,
p=0,024 (Holm 0,028); B≈C (p=0,71) mas B tem melhor pior caso e C deixa o S5 com
sensibilidade 0. B melhora 9 sujeitos, piora 2 (S14 100→90, S9 76,7→73,3). S10:
48,4→87,1 (espec. 21,1→100). Limiares do G: mediana 0,65, gama 0,04–1,00.
LogReg: +4,52, p=0,30 (n.s.); S10 38,7→80,7; S2 falha na ordenação (AUC teste 0,32).
**Diferença para a V5 (Model D):** lá o retreino não ajudava; com o G ajuda em média.
Limites escritos no texto: mesma sessão (otimista entre dias), exige janelas
rotuladas das duas classes, n=15.

# Artigo do SWELL-KW (Koldijk et al., ICMI 2014) — verificado 2026-09-27
PDF em `~/Downloads/The SWELL Knowledge Work Dataset for Stress and User.pdf`.
- **Não regista falhas por participante.** Diz que o conjunto tem 25 participantes,
  que "none of the subjects needed to be excluded" (qualidade das tarefas) e que, nos
  dados por minuto, excluíram minutos com intervalos RR anómalos (>1,2 s ou <0,5 s).
- As falhas dos PP 8, 11 e 23 estão só na folha `SWELL-KW - overview available
  data.xlsx`. Citação no C1: dataset no DANS, `swell_dans`, DOI 10.17026/dans-x55-69zp
  (a autora confirma se a folha veio do DANS).
- Frases úteis para a defesa: "Due to our experimental design of always starting with
  the neutral baseline condition, subjects might have experienced order or fatigue
  effects"; "We did not find significant effects of our working conditions on
  perceived stress" (o autorrelato não entra no texto — decisão da autora).

# Estado final — zip de 27/09 (18:37)

`drafts/correcoes_finais.md` aplicado, exceto por decisão da autora: B7 (LF/HF +
Billman), D2 (cortar "Deep Learning Frameworks"; a orientadora não pediu) e Fig. 3.1
(a autora tirou o texto da imagem). Regression/Clustering ficaram comentados em vez
de apagados; o parágrafo do Benchekroun entrou com redação ligeiramente diferente.
Três correções pendentes (secção H): "a a system" (StateOfTheArt l.4), chave
`\citep{articleoptimisers}` → `articleOptimizers` (estragada pelo *Replace all* da
ortografia) e a figura `gradient_by_class_mean_G.png` ainda com "N=189" no título.
README reescrito para a V6 no mesmo dia.

# Literatura recente verificada nos originais (2026-09-28)

A autora está a reescrever a tese na sua voz e a atualizar a literatura. Base:
Zimatore et al. 2026, "HRV in Stress Monitoring by AI: A Scoping Review", Appl. Sci.
16(1):23, doi 10.3390/app16010023 (`~/Downloads/applsci-16-00023-v2.pdf`; .bib em
`~/Downloads/applsci-v16-i01_20260928.bib`). 15 estudos, pesquisa 2005–jan/2025, sem
IEEE Xplore nem ACM. **A revisão tem erros: citar sempre os originais.**

- **Bahameish et al. 2024** (Sensors 24(10):3210, PMC11126126) — a revisão descreve-o mal.
  Treino e os 61,2% / 85,5% são num dataset **próprio** (38 pessoas, PPG, N-back e
  respiração ritmada, janelas de 60 s sem sobreposição, CV aninhada LOGO por sujeito).
  SWELL+WESAD foram **juntos só como teste externo** (120 amostras de 300 s: 38 neutro,
  53 stress, 29 relax; só resultados combinados): stress vs neutro F1 65,8% / acc 70,3%;
  stress vs relax F1 86,3% / acc 84,1%. Features stress vs neutro: MeanRR, RMSSD, HF
  (MeanRR desce com stress no primário, t(105)=−6,84). Atribuem os >80% de outros estudos
  a segmentação sobreposta e a demasiadas features. Excluíram 8 sujeitos do WESAD.
- **Dahal et al. 2023** (`newArtc6`, no .bib mas **sem \cite no texto**): SWELL+WESAD
  juntos, features pré-calculadas do Kaggle (527 288 instâncias), K-fold estratificado
  (K=15) ao nível da amostra; WESAD sem-stress = baseline+amusement. ≥99% por dataset,
  99,5% combinado. No WESAD, leave-one-out por sujeito: 0,31–0,95. mRMR escolhe features
  diferentes em cada dataset ("depend on the dataset").
- **Mortensen et al. 2023** (IEEE Access 11:57470): 1D CNN no SWELL (Kaggle, 410 322
  registos de 25 pessoas), 5-fold e 70/30 aleatórios, 99,9% em 3 classes.
- **Byun et al. 2025a** (Front. Psychiatry 15:1500310, PMC11754969): RF/MLP, HRV de 5 min,
  split por sujeito; sem escala 0,67–0,73 → com z-score por participante 0,90–0,96;
  SHAP: RRI domina depois da escala. A normalização usa todos os dados da pessoa (12 semanas).
- **Lee, Cho, Kim, Jeon & Byun 2025** (Front. Psychiatry 16:1672260, PMC12508820; chave
  `lee2025deep`; não está na revisão; chamei-lhe "Byun 2025b" por engano): 1D ResNet34 sobre RR brutos (4 Hz), CV por sujeito; 5 min acc
  0,87–0,90 (AUROC 0,91–0,94); 1 min ~0,79–0,82. Sem XAI.
- **Velmovitsky et al. 2022** (Front. Digit. Health 4:1058826): ECG de 30 s do Apple Watch,
  33 pessoas, 2 semanas, rótulos DASS-21; RF acc 0,55 (recall stress 0,43). "it is
  currently unclear whether the brief 30-second ECG reading will be sufficient".
- **Tutunji et al. 2023** (JMIR 25:e39995): erro LOBO 29,87% vs LOSO ~42% (a revisão dá
  42,22% no texto e 42,44% na tabela; confirmar no original).
- Erros da revisão notados: Bahameish (acima); Mendes com RMSSD "maior" 11,6 vs 14,5;
  Tabela 7 troca [41]/[42]; afirma que DL tem mais exatidão que ML (a tese não mostra isso:
  G≈F, e por sujeito nada sobrevive a Holm).

## Giannakakis 2019 (`article8`) e Hong 2020 (`article11`) — verificados 2026-09-28
PDFs em `~/Downloads/8.A novel multi-kernel 1D ...pdf` e `~/Downloads/11.Opportunities ...pdf`
(arXiv v3). Correções propostas em `drafts/correcoes_literatura.md` (L1–L8).
- **Giannakakis et al. 2019** (ACIIW, DWNet1D): dataset próprio, 24 pessoas, 4 tipos de
  stressor; CNN 1D multi-kernel sobre a série RR (crua ou normalizada pela baseline);
  >1,1 M parâmetros; janelas de 60 s com passo de 3 s, geradas depois do 6-fold por sujeito;
  um modelo por stressor. 89,8% = média de 8 (4 stressores × 2 entradas); máx. 99,1%.
  Os 84% com features de HRV + ML são de outro artigo do grupo (ACII 2019) e comparados
  com os 99,1%: não é comparação controlada. Não fala de biofeedback (a Introdução cita-o
  para isso: L7).
- **Hong et al. 2020** (Comput. Biol. Med. 122:103801): 191 artigos de DL em ECG
  (2010–fev/2020), quase só doença cardíaca. "CNN+RNN ensemble using expert features
  yields the best results" assenta na Tabela 3 (FA no CinC 2017; 9 métodos entre F1
  0,818 e 0,831). Não é sobre stress; não vai para a tabela `classifiers_new`. Na tese
  o G (rede + features) não supera o F (só série).

# Conv 1D e literatura recente para o Cap. 2 (2026-09-28, verificado nos originais)

Não existe Model H (modelos A–G; E/F/G = ablação entrada × arquitetura). A subsecção
"Deep Learning Models Selected" (Contribution.tex) não tem nenhum \cite. Cap. 2: 96 refs
únicas, 47 anteriores a 2020; piores: "Detection of Stress" (11/12), ultra-curto (5/5),
"ECG Preprocessing" (11/11), fundamentos de ML. PDFs em `~/Downloads/artigos/`
(`artigouni1.pdf` = tese Ismail Fawaz 2020, "Deep Learning for Time Series Classification",
arXiv:2010.00567, não está no .bib).
- `article8` Giannakakis 2019: CNN 1D multi-kernel sobre série RR (crua ou normalizada pela
  baseline do sujeito), janelas 60 s, 24 sujeitos, 6-fold por sujeito, 89,8% (ML com features
  no mesmo dataset: 84%, = `article17`). `article6` Cho 2019: Conv 1-D sobre ECG bruto
  10/30/60 s, 10-fold por sujeito, 90,19%. `article33` He 2019: CNN, 10 s, 20 pessoas, erro
  17,3% (só resumo). `article11` Hong 2020: melhor = CNN+RNN com features de especialista.
- **Prajod, Mahesh & André 2024** (ICMI '24, doi 10.1145/3678957.3685738): HRV (22 features,
  NeuroKit2, 60 s), LOSO, WESAD/SWELL/ForDigitStress/VerBIO. SWELL dentro: F1 0,60–0,67.
  WESAD→SWELL: F1 0,42–0,47, acc 0,44–0,48. SWELL→WESAD: acc 0,48–0,68. SWELL+WESAD juntos:
  WESAD −12 pp F1. Datasets de stress social generalizam entre si; conclusão: "stressor type".
  Normalização min-max por participante com 5 min de neutro.
- **Prajod & André 2022** (ICMLA, arXiv 2210.06225): WESAD e SWELL; DL em ECG bruto melhor
  dentro do dataset, features de HRV transferem melhor entre datasets.
- **Mishra et al. 2020** (IMWUT, doi 10.1145/3432220): 90 pessoas, 4 estudos; limiar por
  agrupamento no novo estudo > limiar do treino.
- **Vos et al. 2023a** (IJMI 173:105026): revisão; a maioria dos estudos não generaliza.
  **Vos et al. 2023b** (JBI 148:104556): EDA+HR, janelas 25 s; SWELL→WESAD ~50–68%;
  treino em SWELL+NEURO+UBFC (sem WESAD) → 85% no WESAD.
- **Kapoor & Narayanan 2023** (Patterns 4:100804): leakage em 294 artigos de 17 áreas.
- **Nkurikiyeyezu et al. 2020** (Sensors and Materials): calibração pessoal no SWELL.
- **Quigley et al. 2024** (Psychophysiology 61:e14604): guia SPR de HR/HRV (ECG e PPG).
- **Pham et al. 2021** (Sensors 21:3998): revisão de índices de HRV + tutorial.
- **Shaffer, Meehan & Zerr 2020** (Front. Neurosci., doi 10.3389/fnins.2020.594880): ultra-curto.
- **Makowski et al. 2021** (Behav. Res. Methods 53:1689): NeuroKit2.
- **Kiranyaz et al. 2021** (MSSP 151:107398): survey de CNN 1D.
- **Bent et al. 2020** (npj Digit. Med. 3:18): fontes de erro do PPG ótico.
- Haque 2024 (PDF) cobre 2016–2021 e diz "There is no agreed standard for stress evaluation
  at present". Resp-BoostNet: k-fold (k=8), sem LOSO. Xia: divisão aleatória, 100% com 14 features.

- Documento de correções: `drafts/correcoes_literatura_recente.md` (R1–R22 + 20 entradas .bib; R19–R22 no Cap. 4),
  a aplicar **depois** do `drafts/correcoes_literatura.md` (L1–L7). Validado a 28/09 contra o zip
  de 27/09 com L aplicado: 22 itens pendentes → aplicados, sem problemas (só `articleoptimisers`,
  que é da secção H).

# Terceira avaliação externa (2026-09-29): verificações nos dados

Avaliador independente só com o zip de 29/09 (15:00). Triagem completa em
`drafts/triagem_avaliacao_3.md`. Scripts novos em `Classification Models/` (dados V6_wesad).

## Baseline só com FC — `hr_only_baseline.py` → `results_wesad_v6_hr_only.csv`
LogReg sem pesos, mesmos protocolos do `wesad_baselines.py` (a linha "8 features" reproduz
a LogReg da tese ao decimal).
```
                 aleatório        GroupKFold        LOSO
hr_mean          AUC 0,882        0,889 ± 0,099     0,925 ± 0,116   sens 58/58/57
n_beats          0,889            0,899             0,942
8 features       0,922            0,913             0,939
```
Emparelhado por fold contra hr_mean: G GKF +0,030 AUC (p=0,36), LOSO +0,045 (p=0,079);
LogReg-8 GKF +0,024 (p=0,24), LOSO +0,014 (p=0,70). **Em sujeitos novos nenhum modelo
tem AUC significativamente acima da FC sozinha.** A sensibilidade do G a 0,5 é maior
(+25,7 pp LOSO, p=0,03), mas é efeito de limiar/pesos (a LogReg não tem pesos).

## De onde vem o f(baseline)=0,916 do G — `model_g_baseline_check.py` (pesos V6, seed 42)
Entradas reais: P média 0,402 (prevalência no treino 0,360; P média no treino 0,399) →
os pesos de classe só deslocam a probabilidade média ~4 pp. Baseline da tese 0,916;
série com a variabilidade real mas recentrada na média do treino (resumo na média) 0,765;
série plana + resumo real 0,724; série real + resumo na média 0,475 (0,256 / 0,869 por classe).
Leitura: a série plana soma ~0,15; o resto vem de a janela "média" (FC ≈ 82 bpm) ficar do
lado do stress da fronteira do G. **A frase "consistent with the class weighting" (4.4.4)
não se sustenta**; o avaliador tem razão.

## Calibração só com repouso — `rest_normalisation.py` → `results_v6_rest_normalisation.csv`
Mesmo LOSO e mesma divisão temporal do `personalised_protocol.py`; cada sujeito centrado
(ou z-score) pela 1.ª metade do seu baseline, **sem janelas de stress**; LogReg sem pesos, 0,5.
```
                         exat.   DP     pior   espec.mín  <70%
LogReg 8 feat., centre   88,22  14,52   62,07   73,68      3
LogReg 8 feat., zscore   87,74  14,57   62,07   52,63      3
LogReg hr_mean, centre   87,32  14,16   62,07   73,68      3
(tese) G limiar próprio  91,12   7,41   73,33   68,42      0
(tese) LogReg A / B      81,40 / 85,92
```
centre − LogReg A +6,82 pp (t p=0,056; Wilcoxon 0,047); centre − G B −2,91 (p=0,42).
Falta correr o G com a mesma normalização (~30–40 min).

## Outros factos verificados
- Limiar "1,00" da 4.6.3 = **0,997** (S10 e S11). S10 com 0,997: espec. 100%, sens 66,7%;
  S11: 96,7%. Não é "nunca prever stress"; é saída saturada perto de 1 nesses sujeitos.
- Sensibilidade por sujeito (média dos folds): GKF LogReg 73,78 · RF 74,55 · E 82,79 ·
  F 82,23 · G 81,05 · A 81,55 · B 76,95 · C 82,48 · D 77,25; LOSO LogReg 71,59 · RF 72,68 ·
  E 81,76 · F 84,08 · G 82,85 · A 83,01 · B 77,01 · C 83,01 · D 78,86. Em sujeitos novos as
  redes têm sens ≥ espec. (G LOSO 82,85 / 83,91), ao contrário do aleatório (81,54 / 94,36).
- Filtro do WESAD (`filtering_wesad.ipynb`): Butterworth de ordem 2, `filtfilt` (fase zero),
  0,5–40 Hz, na gravação inteira. O `biosppy.ecg.ecg()` aplica ainda um FIR passa-banda
  3–45 Hz de ordem int(0,3·fs) a cada segmento. O Hamilton (biosppy 0.8.0) reinicializa em
  cada segmento de 30 s (`init_ecg = 8` s); o intervalo que atravessa a fronteira perde-se.
- Model A (`model2_simple`) **não tem batch normalisation** (Contribution l.193 diz que os 4 têm).
- `article10` (Maritsch 2019) é sobre smartwatches, mas HRV por PPG; `article6`, `article9`,
  `article20` não são sobre smartwatches. `velmovitsky2022apple` (ECG de 30 s do Apple Watch)
  já está no .bib e é citado no SotA l.416.

## Artigo do WESAD (Schmidt et al., ICMI 2018) — verificado 2026-09-29
PDF em `~/Desktop/Tese 2026/artigos_teses/wesad/ubi_icmi2018.pdf`.
- Baseline de 20 min, sempre **primeiro** (versões A e B só trocam diversão e stress, Fig. 2).
- Postura: "the baseline, amusement and stress conditions were conducted either standing or
  sitting. For each condition, approximately half of the subjects were standing and the other
  half were sitting" → a postura **não** é confundidor ao nível do dataset.
- Fala: "the study participants spoke during the TSST. Hence, the classifiers might have
  partially learned to distinguish between speaking (stress condition) and non-speaking".
- Tarefa binária = baseline + diversão vs TSST; janelas de 60 s (passo 0,25 s); LOSO.
  **Só ECG do peito (Tabela 4), exatidão / F1:** DT 80,17/77,01 · RF 82,78/79,64 ·
  AB 83,37/80,20 · **LDA 85,44/81,31** · kNN 79,19/75,39. A nossa LOSO (30 s, sem diversão):
  74,75–83,74%, LogReg 82,60%.
- G − LogReg em LOSO (AUC): +0,031, IC95 [−0,024; +0,086], p=0,245 (`hr_only_baseline.py`,
  que também imprime a sensibilidade/especificidade por sujeito de todos os modelos).

# Quarta avaliação externa (2026-09-29, fim do dia) — verificações

Zip de 29/09 16:57. Confirmado no texto: notas a vermelho (Contribution l.223, l.314;
Applications l.79), instruções do template no Copyright.tex, F1 ×100 duas vezes
(eq. `f1score`, com precisão e sensibilidade já ×100), gralhas ("so is not used",
"a annotation file", "among others factors", "sampled of 10, 30 or 60 seconds",
"thirty seconds,a", "configuration … has not been evaluated", "will be used", "We use"),
"spatial feature extraction" (l.174/179), "more resistant to overfitting" (l.168),
Intro l.90 ("DL is scarce … promising") e l.94 ("interpretability … feasible"),
SotA l.389 ("necessary instruments for demonstrating legal compliance") vs l.393
("does not prescribe specific XAI techniques"), SotA l.540 ("subject-level memorisation"),
URL do SWELL em easy.dans.knaw.nl, declaração de integridade sem referência a IA.
.bib: `Vaswani2017` e `article35` são @article com booktitle; `article10/15/17/19` sem
journal; `malviya2023lstm` @inbook sem booktitle; misc `pickle`, `r_peaks_biosppy`,
`rr_interval_pyHRV`, `googlenn`, `neuronSite` sem year (têm urldate); `neuron` é @phdthesis.

**Validação nos protocolos por sujeito:** o código baralha os índices de treino antes do
`validation_split` em todos os protocolos (`train_sequence.py` l.170–175,
`retrain_v5.py` l.133–136) → validação = 20% aleatório das janelas dos sujeitos de treino.
O texto (Contribution l.219) só diz "in random order under the random split" → corrigir.

**Nadeau–Bengio com a validação descontada** (`corrected_ttest.py --prefix results_wesad_v6
--ratio 0.3125`, 20/64 em vez de 20/80; só lê CSVs):
- G: igual — só a exatidão vs A sobrevive (Holm NB 0,038).
- F: exatidão vs LogReg (0,042) e RF (0,031), AUC vs A (0,022), **PR-AUC só vs A (0,025)**;
  vs B–E, LogReg, RF passa a 0,054–0,057. Com 0,25 a PR-AUC do F sobrevivia vs os 7.
- F2−F1 (D): IC NB [−5,01; +3,25].
→ "a vantagem é robusta para o F em PR-AUC" depende da razão escolhida; dizer as duas.

**Calibração só com repouso:** t p = 0,056 (n.s.), Wilcoxon 0,047. Reportar os dois só
porque um passa é escolha seletiva: usar o t (como no resto) e dizer que não é significativo.
