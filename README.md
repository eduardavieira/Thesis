# HRV30 — Stress Detection from Ultra-Short HRV

> Master's Thesis — Eduarda Vieira | Universidade do Minho

Binary stress detection from 30-second ECG windows, the acquisition length of
current consumer smartwatch ECG. The repository covers the full pipeline: ECG
processing and HRV feature extraction, classical and deep learning classifiers,
subject-independent evaluation, gradient-based explainability, and an AI
feedback agent.

---

## Key findings

**Detection works when the label reflects a physiological state.** On WESAD,
evaluated by subject (GroupKFold and leave-one-subject-out, no subject in both
train and test), every trained model reaches a ROC-AUC between 0.87 and 0.95.

**The two public datasets used carry physiologically opposite labels.** In
WESAD, heart rate rises by 25.7 bpm under stress. In SWELL-KW it *falls* by
4.6 bpm, in 22 of 22 participants. A model trained on one dataset and tested on
the other scores ROC-AUC 0.18–0.44 in all 10 combinations tested. That is not
chance level (0.5): the signal transfers, but inverted.

**Results must therefore be reported per dataset.** Pooling the two datasets
produces a figure (~61%) that averages a solvable problem with an ill-posed one.
A rule that only knows which dataset a window came from reaches 62.58% under a
random split and 61.35% by subject, above every trained model in the latter.

**The input representation matters more than the architecture.** A two-input
network that reads the NN-interval series alongside the 8 summary features
(Model G) reaches 89.95% accuracy and ROC-AUC 0.959 on WESAD under a random
split, and beats the neural models built on the summary features alone. By
subject, no difference between any two models survives correction for multiple
comparisons.

**Ranking is good, the fixed threshold is not.** Under LOSO the AUC stays above
0.90, but per-subject accuracy ranges from 37.5% to 100% and specificity from
0% to 100%. A single 0.5 threshold cannot serve people whose resting heart
rates span 55–99 bpm. That is the quantitative case for per-subject calibration.

### Note on earlier results

Earlier revisions of this repository reported 60.68% accuracy for a
CNN-Transformer on the pooled dataset. Those figures are superseded by the
analysis above, for three reasons:

- The original RR-interval filter modified a list while iterating over it, so
  about 24% of windows kept physiologically implausible intervals.
- Windows were split at random, so the same participant appeared in both train
  and test.
- The two datasets have opposite labels, as described above.

---

## Results

All WESAD numbers below come from one run of every configuration on identical
partitions (`results_wesad_pr_*.csv`). 95% confidence intervals for every
result are in `Classification Models/results/ci_tables.tex`.

### WESAD — random 80/20 split (5 seeds)

| Model | Accuracy (%) | ROC-AUC | PR-AUC | Parameters |
|---|---|---|---|---|
| Majority class | 64.55 | 0.500 | 0.354 | — |
| Logistic regression | 84.97 ± 2.44 | 0.919 ± 0.014 | 0.880 ± 0.025 | — |
| Random forest | 86.77 ± 2.28 | 0.937 ± 0.015 | 0.908 ± 0.023 | — |
| Model E — MLP on summary features | 85.19 ± 0.53 | 0.929 ± 0.009 | 0.899 ± 0.021 | 7,393 |
| Model A — LSTM | 79.37 ± 2.77 | 0.922 ± 0.010 | 0.892 ± 0.021 | 19,009 |
| Model B — Deep LSTM | 82.75 ± 2.71 | 0.918 ± 0.008 | 0.886 ± 0.016 | 132,993 |
| Model C — CNN-LSTM | 83.92 ± 0.96 | 0.926 ± 0.011 | 0.894 ± 0.023 | 77,249 |
| Model D — CNN-Transformer | 83.49 ± 1.77 | 0.924 ± 0.008 | 0.891 ± 0.018 | 218,625 |
| Model F — Conv1D on NN series | 89.21 ± 1.86 | 0.955 ± 0.011 | 0.931 ± 0.025 | 27,969 |
| **Model G — NN series + summary** | **89.95 ± 1.50** | **0.959 ± 0.012** | **0.933 ± 0.028** | 32,929 |

Paired t-tests over the five seeds, Holm-corrected for the seven comparisons:
Model G's accuracy advantage over Models A–E survives the correction
(*p*<sub>adj</sub> ≤ 0.028; +6.46 pp over Model D). Over logistic regression
(+4.97 pp) and the random forest (+3.18 pp) it does not (*p*<sub>adj</sub> =
0.076). In ROC-AUC the gain survives against every model except the random
forest, three of them at the limit (*p*<sub>adj</sub> = 0.0499). No PR-AUC
difference survives.

### WESAD — by subject

| Model | GroupKFold ROC-AUC | LOSO accuracy (%) | LOSO ROC-AUC | LOSO PR-AUC |
|---|---|---|---|---|
| Majority class | 0.500 | 64.35 | 0.500 | 0.356 |
| Logistic regression | 0.907 ± 0.090 | 82.55 ± 16.10 | **0.951 ± 0.078** | 0.924 ± 0.134 |
| Random forest | 0.874 ± 0.086 | 79.95 ± 15.97 | 0.902 ± 0.128 | 0.853 ± 0.192 |
| Model E | 0.903 ± 0.086 | 79.76 ± 18.17 | 0.940 ± 0.084 | 0.907 ± 0.145 |
| Model A | 0.902 ± 0.078 | 76.30 ± 21.18 | 0.945 ± 0.089 | 0.918 ± 0.147 |
| Model B | 0.893 ± 0.064 | 76.19 ± 18.72 | 0.941 ± 0.078 | 0.907 ± 0.137 |
| Model C | 0.911 ± 0.071 | 81.24 ± 17.04 | 0.949 ± 0.073 | 0.912 ± 0.134 |
| Model D | 0.902 ± 0.074 | 81.28 ± 14.69 | 0.944 ± 0.071 | 0.906 ± 0.137 |
| Model F | 0.899 ± 0.108 | 82.06 ± 19.79 | 0.945 ± 0.086 | **0.926 ± 0.127** |
| Model G | **0.917 ± 0.086** | **82.67 ± 17.18** | 0.919 ± 0.144 | 0.907 ± 0.150 |

No model is distinguishable from the others: none of the 36 pairwise
comparisons survives Holm correction, in either protocol or any metric, and the
ranking reverses between protocols. The signal is large (Cohen's *d* = 1.73 for
mean heart rate), so logistic regression is enough to find it.

LOSO gives a higher AUC than GroupKFold with k = 5 (0.951 vs 0.907 for logistic
regression). This is expected rather than contradictory. With a single subject
per test fold, AUC measures how well each person's own windows are ranked. With
several subjects per fold, it also requires ranking across people with
different baselines.

### SWELL-KW — subject-independent (GroupKFold, 22 subjects)

No model discriminates above chance at α = 0.05. The highest ROC-AUC is 0.612
(logistic regression, *p* = 0.058 against 0.5), followed by 0.566 for the
Conv1D on the NN series. Every model's accuracy falls below the majority class
(60.78%).

The cause lies in the dataset design. The SWELL-KW paper states that *"the
neutral condition was always the first condition"*, so the neutral/stress
contrast is confounded with block order.

### Selected model: Model G

Model G combines a convolutional branch over the NN-interval series (zero-padded
to 73 beats, with a mask and masked average pooling) and a dense branch over the
8 summary features. Its gain comes from the representation, not from capacity:
it has less than a sixth of Model D's parameters, and replacing each window's
series with its own mean interval, which keeps heart rate and beat count but
removes the beat-to-beat dynamics, lowers accuracy from 89.95% to 77.78%.

Confusion matrix on the reference partition (seed 42, N = 189):

|  | Predicted no stress | Predicted stress |
|---|---|---|
| **Actual no stress** | 114 | 8 |
| **Actual stress** | 11 | 56 |

Accuracy 89.95% · precision 87.50% · sensitivity 83.58% · specificity 93.44% ·
ROC-AUC 0.961. Inference takes 53.63 ms per window on CPU, which is negligible
against a 30-second window.

---

## Repository structure

```
Data Processing/
├── RPeakDetection_HRV_fixed.py     # ETL: ECG -> HRV features + NN series (current)
├── RPeakDetection_HRV_RemoteMachine.py   # original ETL (superseded, see note above)
├── time_duration.csv               # file, duration, label, sampling rate per recording
├── filtering_wesad.ipynb           # 0.5–40 Hz band-pass for WESAD
└── Obtaining data/                 # dataset extraction (MATLAB for SWELL-KW, pickle for WESAD)

Classification Models/
├── Data.py                         # loading and normalisation (Models A–D)
├── Models.py                       # Models A–D + training loop
├── retrain_v5.py                   # Models A–D: multi-seed, F1/F2, random/GroupKFold/LOSO, per dataset
├── train_sequence.py               # Models E, F, G (summary MLP, NN series, both)
├── wesad_baselines.py              # majority, logistic regression, random forest on WESAD
├── transfer_test.py                # train on one dataset, test on the other
├── run_pr.sh                       # final WESAD run of every model, 3 protocols, with PR-AUC
├── pr_report.py                    # paired tests with Holm correction, numbers quoted in the thesis
├── confidence_intervals.py         # 95% CIs of every result + LaTeX tables (results/ci_tables.tex)
├── explain_model_g.py              # train/load Model G on WESAD and run the explainability analysis
├── model_g_directions.py           # direction of the effects learned by Model G
├── explain_wesad.py                # same analysis for Model D (earlier selected model)
├── Explainability.py               # gradient attribution, Integrated Gradients (Model D)
├── gradient_class_means.py         # per-class attribution (Model D)
├── personalised_protocol.py        # per-subject threshold calibration (not in the thesis)
├── measure_inference.py            # inference latency (Model D)
├── stress_feedback_agent.py        # AI feedback agent (Anthropic API + local fallback)
├── main.py                         # original V4 training entry point
├── results_*.csv                   # per-seed / per-fold results of every experiment
├── results/
│   ├── model_G/                    # Model G explainability figures and summaries (thesis)
│   └── ci_tables.tex               # WESAD tables with PR-AUC and the CI appendix
├── results_v4_backup/              # figures from the earlier revision
└── saved_models/
    ├── wesad_model_G.weights.h5    # Model G trained on WESAD (seed 42)
    ├── wesad_model.h5              # Model D trained on WESAD
    └── used_model.h5               # Model D from the earlier revision
```

`main.py`, `run_explainability.py` (when run on its own) and `used_model.h5`
belong to the earlier revision and read absolute paths. Use the scripts below
for the current results.

---

## Datasets

- **WESAD** — 15 subjects, Trier Social Stress Test (public speaking and mental
  arithmetic). ECG at 700 Hz.
  [Download](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)
- **SWELL-KW** — 25 subjects recorded, 22 included. Knowledge work under
  email-interruption and time-pressure conditions. ECG at 2048 Hz.
  [Download](https://ssh.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/DANS-X55-69ZP)

After processing: 5,430 thirty-second windows from 37 subjects.

| Dataset | Neutral | Stress | Total | Stress (%) |
|---|---|---|---|---|
| WESAD | 606 | 335 | 941 | 35.60 |
| SWELL-KW | 1,763 | 2,726 | 4,489 | 60.73 |

The first ~8 minutes of every SWELL-KW block are a relaxation phase (a nature
film), not the experimental condition. See [Limitations](#limitations).

---

## Pipeline

```
ECG signal
  ↓ band-pass 0.5–40 Hz (WESAD; SWELL-KW provided pre-filtered)
  ↓ 30-second windows
  ↓ R-peak detection (BioSPPy)
  ↓ RR intervals
  ↓ artefact correction: intervals outside 300–2000 ms, or deviating >20% from
    the local median, are linearly interpolated (not deleted)
  ↓ windows with >20% corrected intervals are discarded
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
marked 11.12% of the intervals in stress windows against 7.27% in no-stress
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
time-domain features (F2) with Model D on WESAD. Removing them does not change
performance (+0.53 pp accuracy, *p* = 0.662, 5 seeds), so no claim is made that
they hurt.

### Class weighting

Class weights are computed with `compute_class_weight('balanced')`, and a 1.2×
multiplier is applied to the stress class (all neural models). In WESAD, where
stress is the minority class, this gives about 1.68 for stress against 0.78 for
no stress, shifting the decision boundary towards sensitivity. Logistic
regression is unweighted and the random forest uses balanced weights.

---

## Installation

```bash
git clone https://github.com/eduardavieira/Thesis.git
cd Thesis
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
> multi-threaded CPU. The same code and seeds, run twice, changed the mean
> accuracy of Model B by 2.86 pp and of Model D by 1.59 pp; Models F and G
> changed by less than 0.2 pp. Differences of a few percentage points between
> Models A–D should not be interpreted. The `results_*.csv` files hold the exact
> per-seed and per-fold numbers behind every table, and
> `wesad_model_G.weights.h5` is the model behind the figures.

---

## Usage

### 1. Extract features

```bash
cd "Data Processing"
python3 RPeakDetection_HRV_fixed.py \
  --datasets  /path/to/raw/ecg \
  --durations time_duration.csv \
  --out        ProcessedECGData_v5.csv \
  --series-out NNSeries_v5.npz
```

This takes about 13 minutes for the full dataset. It prints artefact-correction
statistics per class. The CSV and the `.npz` are aligned row by row, so always
generate them in the same run.

### 2. Run the experiments

```bash
cd "Classification Models"

# Every model on WESAD, three protocols, with PR-AUC (paths set inside the script)
./run_pr.sh abcd && ./run_pr.sh ef && ./run_pr.sh g
python3 wesad_baselines.py --out results_wesad_pr_baselines.csv

# Numbers and paired tests quoted in the thesis; tables with 95% CIs
python3 pr_report.py
python3 confidence_intervals.py

# Single configurations
python3 retrain_v5.py ProcessedECGData_v5.csv --models D --dataset WESAD --cv loso
python3 train_sequence.py --csv ProcessedECGData_v5.csv --series NNSeries_v5.npz \
                          --arch both --dataset WESAD --cv loso

# Cross-dataset transfer
python3 transfer_test.py --csv ProcessedECGData_v5.csv --series NNSeries_v5.npz
```

Common options: `--cv random|group|loso`, `--dataset all|WESAD|SWELL`,
`--seeds N`, `--out file.csv`. `retrain_v5.py` also takes `--models A,B,C,D`
and `--features F1|F2`; `train_sequence.py` takes `--arch summary|seq|both`
(Models E, F, G). Features are selected by column name.

### 3. Explainability

```bash
python3 explain_model_g.py          # train Model G on WESAD (~2 min) and explain
python3 explain_model_g.py --load   # reuse saved_models/wesad_model_G.weights.h5
python3 model_g_directions.py       # direction of the learned effects
```

Figures and summaries are written to `results/model_G/`.

### 4. Feedback agent

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python3 stress_feedback_agent.py --prediction stress        # standalone, no model
```

The agent receives only the predicted label. Its built-in integration with a
saved model (`--model ... --data ... --sample N`) supports the single-input
Models A–D only; for Model G, obtain the label from `explain_model_g.py`.

---

## Explainability

Two gradient-based methods are applied to Model G: the gradient magnitude
|∂y/∂x| per input (unsigned) and Integrated Gradients (signed, from a baseline
at the training mean, 50 steps). The mask that marks real beats is held fixed.
Attributions are normalised by their maximum, so no feature is mapped to zero by
construction.

What the analysis shows:

- **The model uses the order of the beats.** Replacing the series with its own
  mean costs 12.17 pp of accuracy; removing the summary branch lowers accuracy
  by 7.41 pp but leaves the ROC-AUC unchanged, so that branch only moves the
  threshold.
- **The mask carries heart rate.** Its length is the number of beats
  (*r* = 0.98 with `hr_mean`, ROC-AUC 0.882 on its own), and neither method
  attributes it.
- **The model as a whole follows the physiology**: shortening every interval by
  10% raises the predicted probability of stress in 87% of the windows.
- **Per-feature attributions cannot be read one by one.** The summary features
  are computed from the series the model also receives, and the effects learned
  for `hr_mean`, `rmssd` and `nn50` run against the physiology when the series is
  held fixed. They are conditional corrections, not effects of each feature.
- The two methods disagree on rankings, as expected from what each measures, and
  the Integrated Gradients baseline is not neutral: the model assigns 0.959 to a
  window at the training mean.

| Metric | Value | Meaning |
|---|---|---|
| Faithfulness | 0.027 | Accuracy drop when the top-2 summary features are set to the training mean |
| Stability | 0.676 (summary) · 0.728 (series) | Mean Spearman correlation between attributions of an input and of the same input with σ = 0.05 noise |

---

## AI feedback agent

The agent turns a classification into a short lifestyle suggestion through the
Anthropic API. The classifier and the agent are fully decoupled.

**Only the predicted class crosses the API boundary.** No HRV values, no
confidence score and no user identifier are sent. As a consequence, a
prediction at 99% confidence and one at 51% produce identical requests.

API calls have a 30-second timeout and handle timeouts, connection errors, HTTP
errors and malformed responses. On any failure the agent returns locally held
content for the predicted class, so the user still gets a usable message
without connectivity.

The model is configurable through `CLAUDE_MODEL` (default `claude-sonnet-5`).
The earlier thesis revision used `claude-sonnet-4-20250514`, which the provider
retired on 15 June 2026.

Example output for the stress label (WESAD test window 6, subject S7,
classified as stress by Model G with probability above 0.999):

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

- **WESAD is small**: 15 subjects and 941 windows. Between-subject standard
  deviations are large, so the models cannot be ranked against each other by
  subject.
- **SWELL-KW results include the relaxation phase.** The first ~8 minutes of
  each block, a nature film, were not excluded. Excluding the first 16 windows
  of each recording does not remove the inverted heart-rate effect (median
  −5.44 bpm against −5.26 bpm, negative in 18 of 20 participants).
- **Laboratory stressors only.** Neither dataset shows that the same
  discrimination is achievable for everyday stressors.
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
