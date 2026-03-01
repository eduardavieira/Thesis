# Stress Detection from Ultra-Short HRV using Deep Learning

> Master's Thesis — Eduarda | NOVA School of Science and Technology

A deep learning system for real-time binary stress classification using ultra-short (30-second) Heart Rate Variability (HRV) features extracted from ECG signals. The system combines a CNN-Transformer hybrid architecture with explainability analysis and an AI-powered feedback agent.

---

## Overview

Mental stress is one of the leading causes of cardiovascular disease, cognitive dysfunction, and reduced quality of life. This project develops and evaluates four deep learning architectures for stress detection using only 8 time-domain HRV features extracted from 30-second ECG windows — a window length compatible with real-time wearable monitoring.

The best-performing model (Model D — CNN-Transformer) achieves **60.68% accuracy** and **76.95% recall** on a combined SWELL-KW and WESAD dataset, consistent with the state of the art for HRV-only, subject-independent, ultra-short-term stress detection. The architecture provides full explainability through attention mechanisms and gradient-based feature attribution.

---

## Repository Structure

```
├── main.py                     # Main training and evaluation pipeline
├── Models.py                   # All four model architectures
├── Data.py                     # Data loading, preprocessing, normalization
├── explainability.py           # Attention, Integrated Gradients, SHAP analysis
├── stress_feedback_agent.py    # AI-powered feedback agent (Claude API)
├── saved_models/
│   └── used_model.h5           # Saved Model D weights
├── Chapters/Figures/           # Generated explainability figures
└── case_studies/               # Case study visualizations
```

---

## Models

Four deep learning architectures were implemented and evaluated:

| Model | Architecture | Accuracy | Recall | F1 | Parameters |
|-------|-------------|----------|--------|----|------------|
| Model A | Simple LSTM | 60.00% | 72.24% | 0.67 | ~50K |
| Model B | Deep LSTM | 60.00% | 77.90% | 0.69 | ~100K |
| Model C | CNN-LSTM | 59.70% | 62.94% | 0.64 | ~80K |
| **Model D** | **CNN-Transformer** | **60.68%** | **76.95%** | **0.69** | **~150K** |

**Model D** is the recommended model due to its highest accuracy, competitive recall, and built-in explainability through multi-head self-attention.

---

## Datasets

Two publicly available datasets were used:

- **SWELL-KW** — 25 subjects performing knowledge work under email interruptions and time pressure stressors. ECG recorded at 2048 Hz.
  - [Download](https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:58624)

- **WESAD** — 15 subjects exposed to the Trier Social Stress Test (TSST). ECG recorded at 700 Hz.
  - [Download](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)

---

## Pipeline

```
ECG Signal
    ↓ Bandpass filter (0.5–40 Hz)
    ↓ QRS detection (BiospPy)
    ↓ 30-second windowing
    ↓ RR interval extraction (pyHRV)
    ↓ Outlier removal (500–1200 ms)
    ↓ Feature extraction (8 time-domain features)
    ↓ IQR clipping + StandardScaler normalization
    ↓ Reshape to (samples, 8, 1)
    ↓ CNN-Transformer classification
    ↓ AI Feedback Agent (Claude API)
```

### HRV Features Used

| # | Feature | Description |
|---|---------|-------------|
| 1 | hr\_mean | Mean heart rate |
| 2 | nni\_mean | Mean NN interval |
| 3 | nni\_min | Minimum NN interval |
| 4 | nni\_max | Maximum NN interval |
| 5 | sdnn | Standard deviation of NN intervals |
| 6 | rmssd | Root mean square of successive differences |
| 7 | nn50 | Count of successive differences > 50ms |
| 8 | pnn50 | Percentage of nn50 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pyhrv biosppy scipy requests
```

### Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- pyHRV
- BiospPy
- requests (for feedback agent)

---

## Usage

### Training from scratch

```python
python3 main.py
```

This will:
1. Load and preprocess `ProcessedECGData.csv`
2. Train Model D (CNN-Transformer)
3. Evaluate on the held-out test set
4. Save the model to `saved_models/used_model.h5`

To save the model, uncomment these lines in `main.py`:
```python
os.makedirs('saved_models', exist_ok=True)
model.save('saved_models/used_model.h5')
```

### Running on a saved model

```bash
# Using a saved model on a specific test sample
python3 stress_feedback_agent.py \
  --model saved_models/used_model.h5 \
  --data /path/to/ProcessedECGData.csv \
  --sample 3
```

### Running the feedback agent (manual mode)

```bash
# Test without loading a model
python3 stress_feedback_agent.py --prediction stress
python3 stress_feedback_agent.py --prediction no_stress
```

---

## AI Feedback Agent

The feedback agent integrates Model D with the Claude API to generate personalised lifestyle interventions based on the stress prediction.

### Setup

1. Get an API key from [https://console.anthropic.com](https://console.anthropic.com)
2. Set it as an environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Or add to ~/.bashrc for persistence
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

### Example Output

**Stress detected (85.0% confidence):**
```
WELLNESS FEEDBACK:
I can see you're experiencing some stress right now, and that's completely
normal — let's help you find your calm again.

• Take 5 deep breaths — breathe in for 4 counts, hold for 4, exhale for 6.
• Ground yourself with the 5-4-3-2-1 technique.
• Relax your shoulders and jaw — we often hold tension here without realising it.
• Step outside or look out a window for 2-3 minutes.
• Remind yourself: this feeling will pass.
```

**No stress detected (73.2% confidence):**
```
WELLNESS FEEDBACK:
Your body is showing wonderful signs of being relaxed and balanced right now.

• Keep moving gently — a short walk maintains this calm energy.
• Stay hydrated — hydration supports your body's stress resilience.
• This is a great moment for focused work or meaningful activities.
• Notice what led to this state and keep nurturing these healthy patterns.
```

---

## Explainability

Model D provides three complementary explainability methods:

- **Gradient Saliency** — measures output sensitivity to each input feature
- **Integrated Gradients** — directional attributions along a path from baseline to input
- **SHAP** — game-theoretic feature attribution (validation)

Top features identified across all methods:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | nni\_mean | 1.000 |
| 2 | nni\_min | 0.941 |
| 3 | nn50 | 0.910 |
| 4 | sdnn | 0.901 |

These findings align with the known physiology of sympathetic stress activation — shorter NN intervals and reduced variability are hallmarks of the fight-or-flight response.

---

## Results

All models were trained on 5,276 samples and evaluated on 1,320 held-out test samples.

**Model D Confusion Matrix:**

|  | Predicted No Stress | Predicted Stress |
|--|---------------------|------------------|
| **Actual No Stress** | 230 (39.8%) | 348 (60.2%) |
| **Actual Stress** | 171 (23.0%) | 571 (77.0%) |

The model is intentionally biased toward stress detection (high recall), which is appropriate for a wellness application where missing a genuine stress event is more costly than a false alarm.

---

## Why ~60% Accuracy?

Studies that report higher accuracy typically use:
- 5-minute HRV windows (vs our 30 seconds)
- Multiple physiological sensors (EDA, temperature, respiration)
- Person-specific training (not generalizable)

For subject-independent, HRV-only, ultra-short-term stress detection, 60% is consistent with the state of the art:

- Castaldo et al. (2018) — accuracy above 60% for ultra-short HRV classifiers. DOI: [10.1007/978-981-10-5122-7_161](https://doi.org/10.1007/978-981-10-5122-7_161)
- Benchekroun et al. (2023) — 65–68% accuracy using real-life HRV only. DOI: [10.3390/s23041807](https://doi.org/10.3390/s23041807)
- Heimerl et al. (2024) — SWELL-KW models achieve 60–70% F1. DOI: [10.1145/3678957.3685738](https://doi.org/10.1145/3678957.3685738)

---

## License

This project was developed as part of a Master's thesis. Please cite appropriately if you use this code or methodology in your own work.

---

## Acknowledgements

- SWELL-KW dataset: Koldijk et al., Radboud University
- WESAD dataset: Schmidt et al., University of Siegen
- pyHRV library for HRV feature extraction
- BiospPy for ECG signal processing
- Anthropic Claude API for the feedback agent
