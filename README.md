# HRV30 — Stress Detection from Ultra-Short HRV using Deep Learning

> Master's Thesis — Eduarda Vieira | Universidade do Minho

A deep learning system for real-time binary stress classification using ultra-short (30-second) Heart Rate Variability (HRV) features extracted from ECG signals. The system combines a CNN-Transformer hybrid architecture with gradient-based explainability analysis and an AI-powered feedback agent.

---

## Overview

Mental stress is associated with increased risk of cardiovascular disease, cognitive dysfunction and reduced quality of life. HRV30 develops and evaluates four deep learning architectures for stress detection using only 8 time-domain HRV features extracted from 30-second ECG windows — the acquisition length of current consumer smartwatch ECG.

The best-performing model (Model D, CNN-Transformer) achieves **60.68% accuracy** and **76.95% recall** on a combined SWELL-KW and WESAD dataset. This figure should be read against the constraint rather than against the literature at large: see [Why ~60%?](#why-60-accuracy) below.

---

## Repository Structure

```
├── main.py                     # Training and evaluation pipeline
├── Models.py                   # All four model architectures + training loop
├── Data.py                     # Data loading, preprocessing, normalisation
├── Explainability.py           # Attention proxy, Integrated Gradients, quality metrics
├── run_explainability.py       # Full explainability analysis (entry point)
├── stress_feedback_agent.py    # AI feedback agent (Anthropic API + local fallback)
├── saved_models/
│   └── used_model.h5           # Saved Model D weights
└── results/                    # Generated figures
    └── case_studies/           # Per-case explainability visualisations
```

---

## Models

| Model | Architecture | Accuracy | Recall | F1 | Parameters |
|-------|-------------|----------|--------|----|------------|
| Model A | Simple LSTM | 60.00% | 72.24% | 0.67 | ~50K |
| Model B | Deep LSTM | 60.00% | 77.90% | 0.69 | ~100K |
| Model C | CNN-LSTM | 59.70% | 62.94% | 0.64 | ~80K |
| **Model D** | **CNN-Transformer** | **60.68%** | **76.95%** | **0.69** | **218,625** |

Model D is the recommended model: highest accuracy, competitive recall, and interpretability through multi-head self-attention with gradient-based attribution.

All four architectures span three families — recurrent, convolutional and attention-based — and converge within one percentage point of each other, which is the central experimental finding of this work.

---

## Datasets

- **SWELL-KW** — 25 subjects performing knowledge work under email-interruption and time-pressure stressors. ECG at 2048 Hz.
  [Download](https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:58624)

- **WESAD** — 15 subjects exposed to the Trier Social Stress Test (TSST). ECG at 700 Hz.
  [Download](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)

Combined: 6,596 labelled 30-second windows from 40 subjects. Stress is the majority class (~56%).

---

## Pipeline

```
ECG Signal
    ↓ Bandpass filter 0.5–40 Hz (WESAD only; SWELL-KW pre-filtered)
    ↓ QRS detection (BioSPPy)
    ↓ 30-second windowing
    ↓ RR interval extraction (pyHRV)
    ↓ Interval filtering (500–1200 ms)
    ↓ Feature extraction (8 time-domain features)
    ↓ ── train/test split (80/20, stratified, random_state=42) ──
    ↓ IQR clipping at 3×IQR (bounds from training set only)
    ↓ StandardScaler (fitted on training set only)
    ↓ Final clip at ±5 SD
    ↓ Reshape to (samples, 8, 1)
    ↓ CNN-Transformer classification
    ↓ AI Feedback Agent
```

### HRV Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `hr_mean` | Mean heart rate |
| 2 | `nni_mean` | Mean NN interval |
| 3 | `nni_min` | Minimum NN interval |
| 4 | `nni_max` | Maximum NN interval |
| 5 | `sdnn` | Standard deviation of NN intervals |
| 6 | `rmssd` | Root mean square of successive differences |
| 7 | `nn50` | Count of successive differences > 50 ms |
| 8 | `pnn50` | Percentage of nn50 |

Frequency-domain features (LF/HF ratio, Total Power) were evaluated and excluded: they require a minimum duration of stationary data for meaningful spectral estimation and are not reliable at 30 seconds.

### Class weighting

Class imbalance is handled with `compute_class_weight('balanced')`, followed by a **1.2× multiplier applied to the stress class**. Since stress is the majority class, the balanced weighting assigns it a weight below unity; the multiplier partially offsets this, deliberately shifting the decision boundary toward higher recall at the cost of specificity. This is a design choice, not a balancing correction — see the confusion matrix below.

---

## Installation

```bash
git clone https://github.com/eduardavieira/Thesis.git
cd Thesis

pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pyhrv biosppy scipy requests
```

### Environment used for the reported results

| | |
|---|---|
| OS | Ubuntu 20.04 LTS |
| Python | 3.8.10 |
| TensorFlow | 2.13.1 |
| Hardware | Intel Core i5-8250U, CPU only (no CUDA) |

> **Reproducibility note.** The train/test split is fixed by `random_state=42`, so the data partition is identical across runs. TensorFlow and NumPy are **not** seeded, so weight initialisation, dropout masks and batch shuffling vary between runs: retraining will produce a model whose metrics differ slightly from those reported. The saved `used_model.h5` is the model behind every number in this README.

---

## Usage

### Train from scratch

```bash
python3 main.py
```

Loads and preprocesses `ProcessedECGData.csv`, trains Model D, evaluates on the held-out test set, and saves to `saved_models/used_model.h5`.

### Run the explainability analysis

```bash
python3 run_explainability.py
```

Loads the saved model and produces all figures in `results/`, plus the faithfulness and stability metrics. Deterministic given the saved model.

### Run the feedback agent on a test sample

```bash
python3 stress_feedback_agent.py \
  --model saved_models/used_model.h5 \
  --data /path/to/ProcessedECGData.csv \
  --sample 3
```

### Run the feedback agent standalone

```bash
python3 stress_feedback_agent.py --prediction stress
python3 stress_feedback_agent.py --prediction no_stress
```

No model or TensorFlow required — the agent depends only on an HTTP client.

---

## AI Feedback Agent

The agent translates a binary classification into a personalised lifestyle intervention message via the Anthropic API. The two stages are fully decoupled: the classifier has no dependency on the agent, and the agent holds no reference to the model.

**Only the predicted class crosses the API boundary.** No HRV values, no confidence score, no feature attributions and no user identifier are transmitted, so no physiological data leaves the device.

### Setup

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

The model identifier is configurable:

```bash
export CLAUDE_MODEL="claude-sonnet-5"   # default
```

> The results reported in the thesis were generated with `claude-sonnet-4-20250514`, which the provider retired on 15 June 2026. Requests to retired identifiers return an error with no automatic substitution. Check the [current model list](https://docs.claude.com/en/docs/about-claude/models/overview) if calls fail.

### Failure handling

API calls are wrapped with a 30-second timeout and handlers for timeouts, connection failures, HTTP errors, malformed bodies, unexpected response structures and any remaining transport error. A missing API key is detected before the request is issued. On any failure the agent returns locally held content appropriate to the predicted class and logs the reason to stderr — the user still receives a usable intervention message without connectivity.

### Example output (stress detected, 85.0% confidence)

```
WELLNESS FEEDBACK:
I can see you're experiencing some stress right now, and that's completely
normal — let's help you find your calm again. Take 5 deep breaths — breathe
in for 4 counts, hold for 4, then exhale slowly for 6 counts. Ground yourself
with the 5-4-3-2-1 technique: notice 5 things you can see, 4 you can touch,
3 you can hear, 2 you can smell, and 1 you can taste. Relax your shoulders
and jaw — we often hold tension here without realising it. Step outside or
look out a window — even 2-3 minutes of natural light can help reset your
nervous system. Remember, stress is temporary and you have the strength to
work through this moment.
```

---

## Explainability

Two complementary gradient-based methods are applied:

- **Gradient attribution** — absolute first-order gradients of the prediction with respect to each input feature
- **Integrated Gradients** — directional attributions accumulated along the path from a zero baseline to the input

SHAP is implemented in `Explainability.py` but **was not used** in the reported analysis: the two gradient-based methods converge on the same feature ranking, and SHAP's DeepExplainer carries disproportionate cost on an 8-feature input for no additional information.

### Global feature importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `nni_mean` | 1.000 |
| 2 | `nni_min` | 0.941 |
| 3 | `nn50` | 0.910 |
| 4 | `sdnn` | 0.901 |
| 5 | `nni_max` | 0.766 |
| 6 | `hr_mean` | 0.741 |
| 7 | `rmssd` | 0.554 |
| 8 | `pnn50` | 0.541 |

Shorter NN intervals and reduced variability are hallmarks of sympathetic activation, so this ranking aligns with the known physiology. The low ranking of `rmssd` and `pnn50` is corroborated by Castaldo et al. (2019), whose trend analysis found that RMSSD, NN50 and pNN50 cease to differ significantly between rest and stress once the excerpt is shortened to 30 seconds.

### Explanation quality

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Faithfulness | 0.022 | Accuracy drop when masking the top-2 features (60.68% → 58.48%) |
| Stability | 0.878 (SD 0.142) | Spearman correlation between attributions of an input and the same input perturbed with σ = 0.05 noise, over 300 comparisons |

Read together these are mutually explanatory: the model identifies the same features consistently (high stability), yet removing them costs little accuracy (low faithfulness). Both follow from redundancy — `nni_mean`, `nni_min` and `nni_max` are different summaries of the same interval distribution.

---

## Results

Trained on 5,276 samples, evaluated on 1,320 held-out samples.

**Model D confusion matrix:**

|  | Predicted No Stress | Predicted Stress |
|--|---------------------|------------------|
| **Actual No Stress** | 230 (39.8%) | 348 (60.2%) |
| **Actual Stress** | 171 (23.0%) | 571 (77.0%) |

The bias toward stress detection is deliberate (see [class weighting](#class-weighting)). The cost is a specificity of 39.8% — a false alarm for three in five non-stressed windows — which is accepted for a wellness application but would not be acceptable clinically.

Inference: ~63 ms per sample on CPU, negligible against a 30-second classification window.

---

## Why ~60% accuracy?

Studies reporting substantially higher figures operate under conditions a consumer wearable cannot reproduce: longer analysis windows, multiple physiological modalities, or evaluation protocols in which the same participant contributes to both training and test data.

The most directly relevant comparison is **Castaldo et al. (2019)**, [DOI: 10.1186/s12911-019-0742-y](https://doi.org/10.1186/s12911-019-0742-y), which extracted 23 HRV features from excerpts of 5, 3, 2, 1 and 0.5 minutes across 42 subjects. Two findings from that paper frame this project:

1. **The 30-second excerpts were excluded from their classification analysis**, on the grounds that too few features retained coherent behaviour with the 5-minute benchmark at that length.
2. Their conclusion is that automatic stress detection from ultra-short HRV is feasible with excerpts **no shorter than one minute**. The ~88% accuracy they report is obtained at one minute, using MeanNN, StdHR and **HF** — the last a frequency-domain measure not computable below one minute at all.

HRV30 therefore operates below the floor that the most methodologically rigorous study in this area identifies, and excludes by construction one of the three features that study's classifier depends on. That is a consequence of the deployment target, not an oversight: 30 seconds is what a smartwatch gives you.

**Known limitation.** The train/test split is random rather than subject-wise. No per-subject calibration is performed, but windows from the same participant may appear in both partitions. A leave-one-subject-out evaluation would give a stricter — and probably lower — estimate of generalisation.

---

## License

Thesis document: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).
Code: please cite this work if you use the code or methodology.

---

## Acknowledgements

- SWELL-KW dataset — Koldijk et al., Radboud University
- WESAD dataset — Schmidt et al., University of Siegen
- [pyHRV](https://pyhrv.readthedocs.io/) for HRV feature extraction
- [BioSPPy](https://biosppy.readthedocs.io/) for ECG signal processing
- Anthropic Claude API for the feedback agent
