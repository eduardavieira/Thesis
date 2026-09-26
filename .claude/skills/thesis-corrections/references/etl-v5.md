# ETL corrigido (P0.1)

`Data Processing/RPeakDetection_HRV_fixed.py` substitui
`RPeakDetection_HRV_RemoteMachine.py`. **O original não é recuperável por remendo:**
não compila (TabError na L213, tabs misturados com espaços) e a L214 chamaria
`time.time()` sobre um float, porque a L148 reatribui o nome `time` que a L28
tinha importado como módulo. Esse bloco final nunca correu.

## Onde estão os dados

| | |
|---|---|
| ECG em bruto | `/home/eduarda/Desktop/Tese 2026/Datasets/` (101 ficheiros) |
| Durações/labels | `/home/eduarda/Desktop/Tese 2026/Data Processing/time_duration.csv` (96 linhas) |
| Formato | `ficheiro,duração_s,label,fs` |

Pacotes já instalados no ambiente (Python 3.8.10): biosppy, pyhrv, scipy,
numpy, opensignalsreader.

## O que mudou

1. **Filtro corrigido.** O original mutava a lista durante a iteração, por isso
   o elemento a seguir a cada remoção nunca era testado.
2. **Interpola em vez de apagar**, como a supervisora pediu — apagar cria uma
   diferença falsa entre batimentos vizinhos, e RMSSD/NN50/pNN50 dependem
   dessas diferenças.
3. **Gama 300–2000 ms** (30–200 bpm) por omissão em vez de 500–1200 (50–120 bpm),
   porque o TSST passa dos 120 bpm. Configurável em `--bounds`.
4. **Critério relativo** a 20% da mediana local, para ectópicos dentro da gama.
5. **Descarta janelas com >20% de intervalos corrigidos** (`--max-bad`).
6. Caminhos por argumento, escreve cabeçalho, recusa sobrescrever sem `--force`.

## Caminho rápido — validado como idêntico

`pyhrv.hrv()` gasta **1,7 s por janela** para calcular uma média e um
desvio-padrão sobre ~25 números (gera figuras matplotlib internamente). O script
calcula as 8 features de tempo diretamente e é **~13000x mais rápido**.

Verificado em execução paralela sobre os mesmos ficheiros: erro absoluto máximo
0 em `hr_mean`, `nni_mean`, `nni_min`, `nni_max`, `nn50`, `pnn50`, `LF/HF ratio`
e `Total power`; 2,8e-14 em `sdnn` e `rmssd` (ruído de vírgula flutuante).
Execução completa: **~6 min em vez de ~3 h**.

O detalhe que torna isto exato: `pyhrv.utils.nn_format` faz
`[int(x * 1000) for x in nni]` quando `max < 10` — **trunca** para ms inteiros,
não arredonda. `to_nni_ms()` replica a truncatura. Sem isso os valores divergem
~1e-3 relativo, e o `nn50` bastante mais por efeitos de limiar nos 50 ms.

`--use-pyhrv` força o caminho lento, para reverificar.

`welch_psd(mode='dev')` dá `fft_ratio`/`fft_total` idênticos aos de `hrv()` por
~1 ms. Mantidos porque a comparação F1 vs F2 (Contribuição 1) precisa deles,
embora os modelos leiam só as colunas 2:10. Fogem figuras matplotlib a cada
chamada — daí o `plt.close("all")` por ficheiro.

## Efeito medido em `S15_stress_T.csv`

O ficheiro que produziu a amostra das Figuras 16/18 e da demo do agente:

| | V4 | V5 |
|---|---|---|
| Janelas | 55 | 25 (30 descartadas) |
| `nni_min` | [15, 1003] ms | [599, 1004] ms |
| `nni_max` | [1060, 2632] ms | [1073, 1575] ms |
| `sdnn` | [55, 556] ms | [56, 189] ms |
| `hr_mean` | [53, 245] bpm | [52, 75] bpm |
| Janelas com `sdnn > 200 ms` | 24 | 0 |
| Janelas com `pnn50 > 80%` | 10 | 0 |

A janela das Figuras 16/18 (`nni_min` 264, `sdnn` 460, `pnn50` 94,7%) passa a ser
**descartada**. Taxa de correção nesse ficheiro: 25,3% dos intervalos.

## Ponto de decisão: `--max-bad`

Em `S15_stress_T` foram descartadas 30 de 55 janelas (55%). O limiar de 20% é
uma escolha metodológica, não um facto — **tem de ser justificado na tese** e a
sensibilidade a esse valor vale a pena ser reportada.

Se a perda de amostras for grande demais, as alternativas são subir `--max-bad`
(mantém mais janelas, mais interpoladas) ou tratar as janelas descartadas como
uma medida de qualidade de sinal por classe, que é em si um resultado
reportável — e responde diretamente ao pedido da supervisora de contar as
remoções por classe.

## Depois de correr

1. Comparar contagens por classe com a V4 (6596 janelas, 56,2% stress). Se a
   proporção mudar muito, a comparação com os resultados da V4 deixa de ser
   direta e isso tem de ser dito.
2. Correr `scripts/validate.py <novo.csv>` — a secção de contaminação deve dar
   0% em todas as linhas.
3. Só então apontar `Classification Models/main.py:13` para o novo ficheiro.
4. **Guardar o CSV da V4.** É a única forma de reproduzir os números que estão
   na tese enquanto o texto não for atualizado.
