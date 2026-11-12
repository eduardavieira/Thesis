# HRV30 — Stress Detection from Ultra-Short HRV

> Master's Thesis — Eduarda Vieira | Universidade do Minho

Binary stress detection from 30-second ECG windows, the acquisition length of
current consumer smartwatch ECG. The repository covers the full pipeline: ECG
processing and HRV feature extraction, classical and deep learning classifiers,
subject-independent evaluation, per-subject threshold calibration,
gradient-based explainability, and an AI feedback agent.

The LaTeX source of the thesis is `V8_Tese_Eduarda_Vieira_Detecao_Stress_HRV_ECG.zip`.

---

## Key findings

**Detection works when the label reflects a physiological state.** On WESAD,
evaluated by subject (no subject in both train and test), every trained model
reaches a ROC-AUC between 0.878 and 0.919 under GroupKFold and between 0.913 and
0.970 under leave-one-subject-out (LOSO).

**The two public datasets used carry physiologically opposite labels.** In
WESAD, heart rate rises by 25.8 bpm under stress. In SWELL-KW it *falls* by
4.6 bpm, in 22 of 22 participants. A model trained on one dataset and tested on
the other scores ROC-AUC 0.17–0.44 in all 10 combinations tested. That is not
chance level (0.5): the signal transfers, but inverted.

**Results must therefore be reported per dataset.** Pooling the two datasets
produces a figure (~61%) that averages a solvable problem with an ill-posed one.
A rule that only knows which dataset a window came from reaches 62.44% under a
random split and 61.44% by subject, above every trained model in the latter.

**Reading the NN-interval series beats summarising it, under a random split.**
The two models that read the series (Models F and G) reach 90.44% and 89.78%
accuracy and ROC-AUC 0.965 and 0.955 on WESAD, above every model built on the 8
summary features; the advantage is most robust for Model F. By subject, no
difference between any two models survives correction for multiple comparisons.

**Ranking is good, the fixed threshold is not.** Under LOSO the AUC stays high,
but per-subject accuracy ranges from 36.1% to 100% and specificity from 0% to
100%. Choosing the threshold of Model G on about 15 minutes of each subject's
own recording raises its accuracy on the rest of the session from 82.1% to
91.1%, and that of the worst subject from 48.4% to 73.3%, without retraining.

---

## Results

All WESAD numbers below come from one run of every configuration on identical
partitions (`results_wesad_v6_*.csv`). 95% confidence intervals for every
result are in `Classification Models/results/ci_tables_v6.tex`.

### WESAD — random 80/20 split (5 seeds)

| Model | Accuracy (%) | ROC-AUC | PR-AUC | Parameters |
|---|---|---|---|---|
| Majority class | 64.29 | 0.500 | 0.357 | — |
| Logistic regression | 82.64 ± 2.54 | 0.922 ± 0.023 | 0.887 ± 0.027 | — |
| Random forest | 86.70 ± 1.19 | 0.941 ± 0.020 | 0.910 ± 0.020 | — |
| Model E — MLP on summary features | 87.25 ± 0.98 | 0.935 ± 0.017 | 0.911 ± 0.019 | 7,393 |
| Model A — LSTM | 79.78 ± 4.06 | 0.922 ± 0.014 | 0.891 ± 0.020 | 19,009 |
| Model B — Deep LSTM | 81.21 ± 4.91 | 0.918 ± 0.031 | 0.881 ± 0.043 | 132,993 |
| Model C — CNN-LSTM | 84.51 ± 3.71 | 0.934 ± 0.019 | 0.909 ± 0.023 | 77,249 |
| Model D — CNN-Transformer | 85.71 ± 2.30 | 0.929 ± 0.019 | 0.899 ± 0.024 | 218,625 |
| **Model F — Conv1D on NN series** | **90.44 ± 1.81** | **0.965 ± 0.006** | **0.952 ± 0.008** | 27,969 |
| Model G — NN series + summary | 89.78 ± 2.08 | 0.955 ± 0.024 | 0.940 ± 0.028 | 32,929 |

Paired t-tests over the five seeds, Holm-corrected for the seven comparisons
with the models built on the summary features. Model G's accuracy advantage
survives against Models A, B, C and logistic regression (*p*<sub>adj</sub> ≤
0.038), not against Models D and E or the random forest; its ROC-AUC advantage
survives against every model except the random forest, and its PR-AUC advantage
against all seven. Model F's advantage survives in all three metrics against
all seven (*p*<sub>adj</sub> ≤ 0.041). Because the five random partitions
overlap, the corrected resampled t-test of Nadeau and Bengio is also reported:
with it, only Model F keeps a robust advantage (PR-AUC against all seven).
Models F and G are indistinguishable (−0.66 pp, *p* = 0.651).

### WESAD — by subject

| Model | GroupKFold ROC-AUC | LOSO accuracy (%) | LOSO ROC-AUC | LOSO PR-AUC |
|---|---|---|---|---|
| Majority class | 0.500 | 64.08 | 0.500 | 0.359 |
| Logistic regression | 0.913 ± 0.087 | 82.60 ± 16.84 | 0.939 ± 0.136 | 0.921 ± 0.166 |
| Random forest | 0.882 ± 0.112 | 81.06 ± 16.45 | 0.913 ± 0.127 | 0.875 ± 0.184 |
| Model E | 0.908 ± 0.071 | 79.81 ± 17.70 | 0.946 ± 0.080 | 0.925 ± 0.138 |
| Model A | 0.891 ± 0.098 | 74.75 ± 22.91 | 0.947 ± 0.094 | 0.923 ± 0.147 |
| Model B | 0.878 ± 0.084 | 75.37 ± 18.93 | 0.943 ± 0.096 | 0.904 ± 0.168 |
| Model C | 0.897 ± 0.091 | 80.88 ± 16.66 | 0.950 ± 0.080 | 0.917 ± 0.137 |
| Model D | 0.902 ± 0.067 | 79.98 ± 16.54 | 0.945 ± 0.075 | 0.915 ± 0.131 |
| Model F | 0.903 ± 0.075 | 82.66 ± 18.58 | 0.945 ± 0.093 | 0.923 ± 0.140 |
| Model G | **0.919 ± 0.063** | **83.74 ± 16.52** | **0.970 ± 0.047** | **0.955 ± 0.087** |

No model is distinguishable from the others: none of the 36 pairwise
comparisons survives Holm correction, in either protocol or any metric. Model G
has the highest mean in both protocols, but each LOSO fold rests on a single
training run and the mean of 15 folds depends on individual subjects. The two
models that read the series lose the most accuracy when leakage is removed
(−11.41 pp for Model F and −9.64 pp for Model G from the random split to
GroupKFold). The signal is large (Cohen's *d* = 1.73 for mean heart rate), so
logistic regression is enough to find it.

LOSO gives a higher AUC than GroupKFold with k = 5. This is expected rather than
contradictory. With a single subject per test fold, AUC measures how well each
person's own windows are ranked. With several subjects per fold, it also
requires ranking across people with different baselines.

### Per-subject threshold calibration (LOSO, WESAD)

For each held-out subject, the first half of the windows of each condition, in
recording order (29–32 windows, about 15 minutes), is used to choose a
threshold (Youden's index); the second half is the test set.

| Model | Condition | Accuracy (%) | Worst subject (%) | Subjects below 70% |
|---|---|---|---|---|
| Model G | Threshold 0.5 | 82.08 ± 14.93 | 48.4 | 3 |
| Model G | **Own threshold** | **91.12 ± 7.41** | **73.3** | **0** |
| Model G | Retrained with the calibration windows | 90.22 ± 10.59 | 63.3 | 1 |
| Logistic regression | Threshold 0.5 | 81.40 ± 18.32 | 38.7 | 3 |
| Logistic regression | Own threshold | 85.92 ± 16.46 | 37.9 | 2 |

For Model G the gain is significant (+9.04 pp, *p* = 0.028 after Holm
correction). For logistic regression it is not (+4.52 pp, *p* = 0.30): for one
subject the ranking of the test windows itself fails. Calibration and test come
from the same recording session, so the estimate is optimistic about the
stability of a threshold across days.

### SWELL-KW — subject-independent (GroupKFold, 22 subjects)

No model discriminates above chance after correction for multiple comparisons.
The highest ROC-AUC is 0.612 (logistic regression, *p* = 0.015 against 0.5,
0.075 after Holm correction across the five trained configurations); the others
lie between 0.523 and 0.549. Every model's accuracy falls below the majority
class (61.02%).

The cause lies in the dataset design. The SWELL-KW paper states that *"the
neutral condition was always the first condition"*, so the neutral/stress
contrast is confounded with block order.

### Selected model: Model G

Model G combines a convolutional branch over the NN-interval series (zero-padded
to 73 beats, with a mask and masked average pooling) and a dense branch over the
8 summary features. Its gain comes from the representation, not from capacity:
it has less than a sixth of Model D's parameters, and shuffling the intervals
within each window (recomputing the summary features, so that neither input
keeps the original order) lowers accuracy by 13.19 pp. Model F, which reads the
series alone, is an equally defensible choice.

Confusion matrix on the reference partition (seed 42, N = 182):

|  | Predicted no stress | Predicted stress |
|---|---|---|
| **Actual no stress** | 112 | 5 |
| **Actual stress** | 11 | 54 |

Accuracy 91.21% · precision 91.53% · sensitivity 83.08% · specificity 95.73% ·
ROC-AUC 0.965. Inference takes 49.21 ms per window with Keras on a laptop CPU,
which is negligible against a 30-second window but is not a measurement on a
wearable.

---

## Repository structure

```
V8_Tese_Eduarda_Vieira_Detecao_Stress_HRV_ECG.zip   # LaTeX source of the thesis

Data Processing/
├── RPeakDetection_HRV_fixed.py     # ETL: ECG -> HRV features + NN series
├── fix_durations.py                # duration of each recording = samples / sampling rate
├── time_duration_v6.csv            # file, duration, label, sampling rate per recording
├── filtering_wesad.ipynb           # 0.5–40 Hz band-pass for WESAD
└── Obtaining data/                 # dataset extraction (MATLAB for SWELL-KW, pickle for WESAD)

Classification Models/
├── run_v6.sh                       # every experiment (ETL, WESAD, pooled, SWELL, transfer, XAI)
├── Data.py, Models.py              # loading, normalisation and Models A–D
├── retrain_v5.py                   # Models A–D: multi-seed, F1/F2, random/GroupKFold/LOSO, per dataset
├── train_sequence.py               # Models E, F, G (summary MLP, NN series, both)
├── wesad_baselines.py              # majority, logistic regression, random forest on WESAD
├── transfer_test.py                # train on one dataset, test on the other
├── dataset_analysis.py             # effect sizes, per-subject consistency, SWELL-KW blocks,
│                                   #   pooled baselines, dataset-only rule, domain classifier
├── pr_report.py                    # paired tests with Holm correction (--v6)
├── corrected_ttest.py              # Nadeau–Bengio corrected resampled t-test
├── confidence_intervals.py         # 95% CIs + LaTeX tables (--v6 -> results/ci_tables_v6.tex)
├── personalised_protocol.py        # per-subject threshold calibration
├── explain_model_g.py              # train/load Model G and run the explainability analysis
├── model_g_directions.py           # direction of the effects learned by Model G
├── model_g_permutation.py          # within-window permutation test of beat order
├── model_g_faithfulness_ref.py     # faithfulness against all 28 pairs of summary features
├── appendix_tables.py              # appendix tables of recording durations
├── hr_only_baseline.py             # mean heart rate alone as a classifier
├── rest_normalisation.py           # features centred on each subject's rest windows
├── model_g_baseline_check.py       # where Model G's 0.916 at the Integrated Gradients baseline comes from
├── stress_feedback_agent.py        # AI feedback agent (Anthropic API + local fallback)
├── results_*v6*.csv                # per-seed / per-fold results behind the thesis
├── logs_v6/                        # summaries of the analyses (dataset, XAI, reports)
├── results/
│   ├── model_G_v6/                 # Model G explainability figures and summaries (thesis)
│   └── ci_tables_v6.tex            # WESAD tables and the confidence-interval appendix
└── saved_models/
    └── wesad_model_G_v6.weights.h5 # Model G trained on WESAD (seed 42), behind the figures
```

---

## Datasets

- **WESAD** — 15 subjects, Trier Social Stress Test (public speaking and mental
  arithmetic). ECG at 700 Hz. Baseline against stress; the amusement and
  meditation segments are excluded.
  [Download](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)
- **SWELL-KW** — 25 subjects recorded, 22 included (the dataset's overview of
  available data flags the cardiac recordings of participants 8, 11 and 23).
  Knowledge work under email-interruption and time-pressure conditions. ECG at
  2048 Hz.
  [Download](https://ssh.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/DANS-X55-69ZP)

After processing: 5,409 thirty-second windows from 37 subjects.

| Dataset | Neutral | Stress | Total | Stress (%) |
|---|---|---|---|---|
| WESAD | 581 | 326 | 907 | 35.94 |
| SWELL-KW | 1,763 | 2,739 | 4,502 | 60.84 |

The first ~8 minutes of every SWELL-KW block are a relaxation phase (a nature
film), not the experimental condition. See [Limitations](#limitations).

---

## Pipeline

```
ECG signal
  ↓ band-pass 0.5–40 Hz (WESAD; SWELL-KW provided pre-filtered)
  ↓ non-overlapping 30-second windows (length from samples / sampling rate)
  ↓ R-peak detection (BioSPPy, Hamilton segmenter)
  ↓ RR intervals
  ↓ artefact correction: intervals outside 300–2000 ms, or deviating >20% from
    the median of the five intervals centred on them, are linearly interpolated
  ↓ windows with >20% corrected intervals are discarded (all in SWELL-KW)
  ↓ 8 time-domain HRV features  +  NN-interval series
  ↓ split (random, GroupKFold or LOSO by subject)
  ↓ normalisation fitted on the training set only
  ↓ classifier
  ↓ AI feedback agent
```

Normalisation: Models A–D use IQR clipping at 3×IQR, StandardScaler and a clip
at ±5 SD; Models E–G standardise the series and the summary features and clip
at ±5 SD; the reference classifiers use StandardScaler alone.

Artefacts are interpolated rather than deleted because RMSSD, NN50 and pNN50
are computed from successive differences, and deleting an interval creates a
false difference between its neighbours. The old 500–1200 ms range would have
marked 11.15% of the intervals in stress windows against 7.27% in no-stress
windows. The feature computation replicates `pyhrv.hrv()` exactly, including
its truncation to integer milliseconds, and runs about 13,000× faster.

### HRV features

| Feature | Description |
|---|---|
| `hr_mean` | Mean heart rate |
| `nni_mean` | Mean NN interval |
| `nni_min` | Minimum NN interval |
| `nni_max` | Maximum NN interval |
| `sdnn` | Standard deviation of NN intervals |
| `rmssd` | Root mean square of successive differences |
| `nn50` | Successive differences > 50 ms |
| `pnn50` | Percentage of `nn50` |

LF/HF ratio and total power (feature set F1) were compared against the 8
time-domain features (F2) with Model D on WESAD. Removing them produces no
detectable change (−0.88 pp accuracy, *p* = 0.398, 95% CI −3.5 to +1.7 pp,
5 seeds).

### Class weighting

Class weights are computed with `compute_class_weight('balanced')`, and a 1.2×
multiplier is applied to the stress class (all neural models). In WESAD, where
stress is the minority class, this gives about 1.67 for stress against 0.78 for
no stress. Logistic regression is unweighted and the random forest uses
balanced weights.

---

## Installation

```bash
git clone https://github.com/dmlrhaslab/HRV-Stress.git
cd HRV-Stress
pip install tensorflow==2.13.1 scikit-learn pandas numpy scipy matplotlib \
            biosppy pyhrv requests
```

| | |
|---|---|
| OS | Ubuntu 20.04 LTS |
| Python | 3.8.10 |
| TensorFlow | 2.13.1 |
| Hardware | Intel Core i5-8250U, CPU only |

> **Reproducibility.** Seeds are set, but TensorFlow is not deterministic on a
> multi-threaded CPU (deterministic execution was not enabled). The same code
> and seeds, run twice, changed the mean accuracy of Model B by 2.86 pp and of
> Model D by 1.59 pp; Models F and G changed by less than 0.2 pp. The
> `results_*v6*.csv` files hold the exact per-seed and per-fold numbers behind
> every table, and `wesad_model_G_v6.weights.h5` is the model behind the
> figures.

---

## Usage

The raw data are not in the repository; download both datasets (see
[Datasets](#datasets)) and extract the ECG with the scripts in
`Data Processing/Obtaining data/`. `run_v6.sh` reads the paths from two
environment variables: `RAW_DIR`, the folder with the raw ECG CSV files
(default `data/raw` at the root of the repository), and `DATA_DIR`, the folder
for the extracted features and NN series (default `data`). The `data/` folder is
ignored by git. The default paths of several scripts point to files outside the
repository; pass the paths explicitly, as `run_v6.sh` and the commands below do.

### 1. Extract features

```bash
cd "Data Processing"
python3 fix_durations.py --datasets /path/to/raw/ecg \
  --durations time_duration.csv --out time_duration_v6.csv   # only if rebuilding it
python3 RPeakDetection_HRV_fixed.py \
  --datasets  /path/to/raw/ecg \
  --durations time_duration_v6.csv \
  --out        ProcessedECGData_v6.csv \
  --series-out NNSeries_v6.npz
```

It prints artefact-correction statistics per class. The CSV and the `.npz` are
aligned row by row, so always generate them in the same run.

### 2. Run the experiments

```bash
cd "Classification Models"
export RAW_DIR=/path/to/raw/ecg DATA_DIR=/path/to/data   # optional, see above

./run_v6.sh etl_wesad          # WESAD-only extraction (fast)
./run_v6.sh base               # reference classifiers on WESAD
./run_v6.sh abcd; ./run_v6.sh ef; ./run_v6.sh g     # Models A–G, three protocols
./run_v6.sh etl_all            # full extraction (~2 h, SWELL-KW at 2048 Hz)
./run_v6.sh pooled_abcd; ./run_v6.sh pooled_efg      # pooled negative control
./run_v6.sh swell_transfer     # SWELL-KW by subject + cross-dataset transfer
./run_v6.sh f1f2_xai           # F1/F2 ablation + Model G explainability

python3 dataset_analysis.py --csv "$DATA_DIR/ProcessedECGData_v6.csv" \
  --series "$DATA_DIR/NNSeries_v6.npz" --pooled-out results_v6_pooled_baselines.csv
python3 pr_report.py --v6                 # numbers and paired tests quoted in the thesis
python3 corrected_ttest.py --prefix results_wesad_v6 \
  --f1 results_wesad_v6_F1_D.csv --f2 results_wesad_v6_F2_D.csv
python3 confidence_intervals.py --v6      # -> results/ci_tables_v6.tex

python3 personalised_protocol.py --csv "$DATA_DIR/ProcessedECGData_v6_wesad.csv" \
  --series "$DATA_DIR/NNSeries_v6_wesad.npz" --models G,LogReg --scaler standard \
  --out results_v6_personalised.csv
```

Common options: `--cv random|group|loso`, `--dataset all|WESAD|SWELL`,
`--seeds N`, `--out file.csv`. `retrain_v5.py` also takes `--models A,B,C,D`
and `--features F1|F2`; `train_sequence.py` takes `--arch summary|seq|both`
(Models E, F, G). Features are selected by column name.

### 3. Explainability

```bash
ARGS="--csv $DATA_DIR/ProcessedECGData_v6_wesad.csv --series $DATA_DIR/NNSeries_v6_wesad.npz \
      --weights saved_models/wesad_model_G_v6.weights.h5"
python3 explain_model_g.py $ARGS --load --out results/model_G_v6
python3 model_g_directions.py $ARGS
python3 model_g_permutation.py $ARGS --out results_v6_model_g_permutation.csv
python3 model_g_faithfulness_ref.py $ARGS
```

### 4. Feedback agent

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python3 stress_feedback_agent.py --prediction stress        # agent alone, no model

# Model G classifies a WESAD test window (seed-42 random split) and passes its
# label to the agent; windows 3 (S8, stress) and 0 (S7, baseline) are the two
# shown in the thesis
python3 stress_feedback_agent.py --csv "$DATA_DIR/ProcessedECGData_v6_wesad.csv" \
  --series "$DATA_DIR/NNSeries_v6_wesad.npz" --sample 3
```

The agent receives only the predicted label. `--weights` defaults to
`saved_models/wesad_model_G_v6.weights.h5`, and the windows are numbered as in the
explainability analysis.

---

## Explainability

Two gradient-based methods are applied to Model G: the gradient magnitude
|∂y/∂x| per input (unsigned) and Integrated Gradients (signed, from a baseline
at the training mean, 50 steps). The mask that marks real beats is held fixed.
Attributions are normalised by their maximum, so no feature is mapped to zero by
construction.

What the analysis shows:

- **The model uses the order of the beats.** Shuffling the intervals within each
  window, with the summary features recomputed, costs 13.19 pp of accuracy (50
  permutations). Setting the summary features to the training mean costs
  12.64 pp of accuracy but only 0.016 of ROC-AUC, so that branch mainly moves
  the threshold.
- **The mask carries heart rate.** Its length is the number of beats
  (*r* = 0.998 with `hr_mean`, ROC-AUC 0.889 on its own), and neither method
  attributes it.
- **The model as a whole follows the physiology**: shortening every interval by
  10% raises the predicted probability of stress in 98% of the windows.
- **Per-feature attributions cannot be read one by one.** The summary features
  are computed from the series the model also receives, and the effects learned
  for `hr_mean`, `nni_mean`, `nn50` and `pnn50` run against the physiology when
  the series is held fixed. They are conditional corrections, not effects of
  each feature.
- The two methods disagree on rankings, as expected from what each measures, and
  the Integrated Gradients baseline is not neutral: the model assigns 0.916 to a
  window at the training mean.

| Metric | Value | Meaning |
|---|---|---|
| Faithfulness | 10.44 pp | Accuracy drop when the top-2 summary features are set to the training mean; the largest of the 28 possible pairs (mean 4.24 pp) |
| Stability | 0.729 (summary) · 0.681 (series) | Mean Spearman correlation between attributions of an input and of the same input with σ = 0.05 noise |

---

## AI feedback agent

The agent turns a classification into a short lifestyle suggestion through the
Anthropic API. The classifier and the agent are fully decoupled.

**Only the predicted class crosses the API boundary.** No HRV values, no
confidence score and no user identifier are sent. The label itself is inferred
health information. As a consequence, a prediction at 99% confidence and one at
51% produce identical requests.

API calls have a 30-second timeout and handle timeouts, connection errors, HTTP
errors and malformed responses. On any failure the agent returns locally held
content for the predicted class, so the user still gets a usable message
without connectivity.

Requests are issued to `claude-sonnet-4-20250514`, the model used in the thesis,
with a maximum response length of 512 tokens. The model can be changed through
the `CLAUDE_MODEL` environment variable.

Example output for the stress label (WESAD test window 3, subject S8,
classified as stress by Model G with probability 0.868):

```
Your body's asking for a small reset — here are a few things that can help
right now:
1. Try box breathing — Inhale for 4 counts, hold for 4, exhale for 4, hold for
   4. Repeat for 2-3 minutes.
2. Unclench and drop your shoulders — Notice your jaw, hands, and shoulders.
3. Step away for a 5-minute break, if you can.
4. Ground yourself with your senses — Name 3 things you can see, 2 things you
   can hear, and 1 thing you can feel.
5. Sip some water slowly.
You're doing great by checking in with yourself.
```

---

## Limitations

- **WESAD is small**: 15 subjects and 907 windows. Between-subject standard
  deviations are large, so the models cannot be ranked against each other by
  subject.
- **Per-subject calibration was tested within one recording session.** Whether
  a threshold remains valid across days is not established, and it requires
  labelled windows of both classes from each user.
- **SWELL-KW results include the relaxation phase.** The first ~8 minutes of
  each block, a nature film, were not excluded. Excluding the first 16 windows
  of each recording does not remove the inverted heart-rate effect (median
  −5.76 bpm against −5.26 bpm, negative in 18 of 20 participants).
- **Laboratory stressors only.** Neither dataset shows that the same
  discrimination is achievable for everyday stressors.
- **Model G was selected on the same test partitions** used to report it, so its
  reported performance is an optimistic estimate.
- **Feature-level explanations of Model G are not interpretable one by one**,
  and the heart rate carried by the beat count is not attributed.
- **The agent ignores confidence.** Only the predicted class reaches the API.

---

## License

Thesis document: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).
Code: please cite this work if you use the code or methodology.

## Acknowledgements

- WESAD — Schmidt et al., University of Siegen
- SWELL-KW — Koldijk et al., Radboud University
- [BioSPPy](https://biosppy.readthedocs.io/) and [pyHRV](https://pyhrv.readthedocs.io/)
- Anthropic Claude API for the feedback agent
