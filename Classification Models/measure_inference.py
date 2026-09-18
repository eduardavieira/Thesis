import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.models import load_model
from Data import normalize_data
import pandas as pd
from Models import PositionalEncoding
from sklearn.model_selection import train_test_split

# Load model
model = load_model(
    'saved_models/wesad_model.h5',
    custom_objects={'PositionalEncoding': PositionalEncoding}
)

# Load data (V5, filtrada ao WESAD, por NOME e nao por posicao)
path = "/home/eduarda/Desktop/ProcessedECGData_v5.csv"
numeric_features = ['hr_mean','nni_mean','nni_min','nni_max',
                    'sdnn','rmssd','nn50','pnn50']
_d = pd.read_csv(path).dropna(subset=numeric_features + ['label'])
_d = _d[_d['file'].str.contains('ECG_values')].reset_index(drop=True)  # WESAD
x_features = _d[numeric_features].to_numpy(float)
y_label = _d['label'].to_numpy().astype(int)
print(f"WESAD: {len(_d)} janelas")

x_train, x_test, y_train, y_test = train_test_split(
    x_features, y_label,
    test_size=0.2,
    random_state=42,
    stratify=y_label
)

x_train_norm, x_test_norm = normalize_data(x_train, x_test)
x_test_reshape = x_test_norm.reshape(x_test_norm.shape[0], -1, 1)

# Measure inference time
single_sample = x_test_reshape[0:1]

# Warm-up run
model.predict(single_sample, verbose=0)

# Timed runs
n_runs = 1000
start = time.time()
for _ in range(n_runs):
    model.predict(single_sample, verbose=0)
avg_ms = (time.time() - start) / n_runs * 1000

print(f"Average inference time per sample: {avg_ms:.2f} ms")