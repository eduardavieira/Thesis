"""
Comprehensive Explainability Analysis for CNN-Transformer Model
Can be run standalone (loads saved model) or after training (uses model in memory)
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Try to use model from memory first, otherwise load from disk
try:
    # Check if model exists in current scope
    model
    x_test_reshape
    y_test
    print("Using model from memory (just trained)")
    USING_SAVED_MODEL = False
except NameError:
    print("Model not in memory - will load from saved file")
    USING_SAVED_MODEL = True

# If model not in memory, load it
if USING_SAVED_MODEL:
    from tensorflow.keras.models import load_model
    from Data import load_features, normalize_data
    from Models import PositionalEncoding
    from sklearn.model_selection import train_test_split
    
    # Check if model file exists
    model_path = 'saved_models/used_model.h5'
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        print("\nOptions:")
        print("  1. Train the model first (it will auto-save)")
        print("  2. Or run this script right after training (while model is in memory)")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, custom_objects={'PositionalEncoding': PositionalEncoding})
    print("Model loaded!")
    
    # Load and prepare data
    print("Loading and preparing data...")
    path = "/home/eduarda/Desktop/ProcessedECGData.csv"
    x_features, y_label, numeric_features = load_features(path)

    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_label,
        test_size=0.2, 
        random_state=42, 
        stratify=y_label
    )
    
    x_train_norm, x_test_norm = normalize_data(x_train, x_test)
    
    x_train_reshape = x_train_norm.reshape(len(x_train), -1, 1)
    x_test_reshape = x_test_norm.reshape(len(x_test), -1, 1)
    
    print(f"Data ready: Train {x_train_reshape.shape}, Test {x_test_reshape.shape}")

# Create results directory
os.makedirs('results', exist_ok=True)
os.makedirs('results/case_studies', exist_ok=True)

# Import explainability modules
from Explainability import (
    AttentionExtractor,
    SHAPExplainer,
    IntegratedGradientsExplainer,
    create_case_studies,
    evaluate_explainability_metrics
)

# Feature names
feature_names = numeric_features

print("COMPREHENSIVE EXPLAINABILITY ANALYSIS")

# ============================================================
# 1. GRADIENT ATTRIBUTION
# ============================================================
print("1. GRADIENT ATTRIBUTION")

attention_extractor = AttentionExtractor(model)

# Find interesting samples
y_pred_proba = model.predict(x_test_reshape, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# True positive (correctly detected stress)
tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
if len(tp_indices) > 0:
    tp_idx = tp_indices[0]
    print(f"Found True Positive example at index {tp_idx}")
    attention_extractor.visualize_attention_heatmap(
        x_test_reshape[tp_idx:tp_idx+1],
        y_test[tp_idx],
        y_pred_proba[tp_idx],
        feature_names,
        save_path='results/gradient_true_positive.png'
    )
else:
    print("No true positive examples found")

# True negative (correctly detected no stress)
tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]
if len(tn_indices) > 0:
    tn_idx = tn_indices[0]
    print(f"Found True Negative example at index {tn_idx}")
    attention_extractor.visualize_attention_heatmap(
        x_test_reshape[tn_idx:tn_idx+1],
        y_test[tn_idx],
        y_pred_proba[tn_idx],
        feature_names,
        save_path='results/gradient_true_negative.png'
    )
    
    # Compare patterns
    if len(tp_indices) > 0:
        print("Creating comparison plot...")
        attention_extractor.compare_attention_patterns(
            x_test_reshape[tp_idx:tp_idx+1],
            x_test_reshape[tn_idx:tn_idx+1],
            feature_names,
            save_path='results/gradient_by_class.png'
        )
else:
    print("⚠️  No true negative examples found")


# ============================================================
# 2. SHAP ANALYSIS
# ============================================================
print("2. GRADIENT-BASED FEATURE IMPORTANCE")

print("Computing feature importance using gradients...")
num_samples = min(200, len(x_test_reshape))
print(f"  Analyzing {num_samples} samples...")

feature_importance = np.zeros(len(feature_names))

for i in range(num_samples):
    with tf.GradientTape() as tape:
        x_tensor = tf.constant(x_test_reshape[i:i+1], dtype=tf.float32)
        tape.watch(x_tensor)
        pred = model(x_tensor)
    grad = tape.gradient(pred, x_tensor)
    feature_importance += np.abs(grad.numpy()[0, :, 0])

feature_importance /= len(x_test_reshape)

# Normalize
feature_importance_norm = feature_importance / feature_importance.max()

# Create visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Sort by importance
sorted_idx = np.argsort(feature_importance_norm)
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importance = feature_importance_norm[sorted_idx]

colors = plt.cm.RdYlGn_r(sorted_importance)
bars = ax.barh(sorted_features, sorted_importance, color=colors, alpha=0.8)

ax.set_xlabel('Relative Importance', fontsize=12, fontweight='bold')
ax.set_title('Global Feature Importance (Gradient-Based)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/feature_importance_global.png', dpi=300, bbox_inches='tight')
print(" Saved global feature importance: results/feature_importance_global.png")
plt.close()

# Print rankings
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance_norm
}).sort_values('Importance', ascending=False)

print("\n" + "="*60)
print("GLOBAL FEATURE IMPORTANCE")
print("="*60)
for i, row in importance_df.iterrows():
    bar = "█" * int(row['Importance'] * 40)
    print(f"  {row['Feature']:15s}: {bar} {row['Importance']:.4f}")
print("="*60 + "\n")

# Single prediction explanations (replacing SHAP waterfall plots)
print("Creating individual prediction explanations...")

def explain_single_prediction_gradient(x_sample, y_pred_prob, feature_names, title, save_path):
    """Explain using gradients instead of SHAP"""
    with tf.GradientTape() as tape:
        x_tensor = tf.constant(x_sample, dtype=tf.float32)
        tape.watch(x_tensor)
        pred = model(x_tensor)
    grad = tape.gradient(pred, x_tensor)
    
    importance = np.abs(grad.numpy()[0, :, 0])
    importance_norm = importance / (importance.max() + 1e-8)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Feature contributions
    sorted_idx = np.argsort(importance_norm)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_vals = importance_norm[sorted_idx]
    
    colors = plt.cm.RdYlGn_r(sorted_vals)
    ax1.barh(sorted_features, sorted_vals, color=colors, alpha=0.8)
    ax1.set_xlabel('Contribution to Prediction', fontsize=11, fontweight='bold')
    ax1.set_title(f'{title}\nPrediction: {"STRESS" if y_pred_prob>0.5 else "NO STRESS"} ({y_pred_prob:.1%})', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Right: Feature values
    feature_vals = x_sample[0, :, 0]
    ax2.scatter(range(len(feature_names)), feature_vals,
               s=importance_norm*500+100, c=importance_norm, cmap='RdYlGn_r',
               alpha=0.7, edgecolors='black', linewidth=1.5)
    ax2.set_xticks(range(len(feature_names)))
    ax2.set_xticklabels(feature_names, rotation=45, ha='right')
    ax2.set_ylabel('Normalized Feature Value', fontsize=11, fontweight='bold')
    ax2.set_title('Feature Values (size = importance)', fontsize=12, fontweight='bold')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Saved explanation: {save_path}")
    plt.close()

if len(tp_indices) > 0:
    explain_single_prediction_gradient(
        x_test_reshape[tp_idx:tp_idx+1],
        y_pred_proba[tp_idx],
        feature_names,
        'STRESS DETECTION EXAMPLE',
        'results/gradient_stress_example.png'
    )

if len(tn_indices) > 0:
    explain_single_prediction_gradient(
        x_test_reshape[tn_idx:tn_idx+1],
        y_pred_proba[tn_idx],
        feature_names,
        'NO STRESS DETECTION EXAMPLE',
        'results/gradient_nostress_example.png'
    )


# ============================================================
# 3. INTEGRATED GRADIENTS
# ============================================================
print("3. INTEGRATED GRADIENTS ANALYSIS")

ig_explainer = IntegratedGradientsExplainer(model)

if len(tp_indices) > 0:
    print(f"Computing IG for stress example (index {tp_idx})...")
    ig_explainer.visualize_attributions(
        tf.constant(x_test_reshape[tp_idx:tp_idx+1], dtype=tf.float32),
        feature_names,
        y_pred_proba[tp_idx],
        save_path='results/ig_stress_example.png'
    )

if len(tn_indices) > 0:
    print(f"Computing IG for no-stress example (index {tn_idx})...")
    ig_explainer.visualize_attributions(
        tf.constant(x_test_reshape[tn_idx:tn_idx+1], dtype=tf.float32),
        feature_names,
        y_pred_proba[tn_idx],
        save_path='results/ig_nostress_example.png'
    )


# ============================================================
# 4. COMPREHENSIVE CASE STUDIES
# ============================================================
print("4. CASE STUDY GENERATION")

create_case_studies(
    model,
    x_test_reshape,
    y_test,
    feature_names,
    output_dir='results/case_studies'
)


# ============================================================
# 5. QUANTITATIVE METRICS
# ============================================================
print("5. QUANTITATIVE EVALUATION")

metrics = evaluate_explainability_metrics(
    model,
    x_test_reshape,
    y_test,
    feature_names
)


# ============================================================
# 6. SUMMARY REPORT
# ============================================================
print("EXPLAINABILITY ANALYSIS COMPLETE")
print("\nGenerated Files:")
print("  results/")
print("  ├── gradient_true_positive.png")
print("  ├── gradient_true_negative.png")
print("  ├── gradient_by_class.png")
print("  ├── shap_global_importance.png")
print("  ├── shap_stress_example.png")
print("  ├── shap_nostress_example.png")
print("  ├── ig_stress_example.png")
print("  ├── ig_nostress_example.png")
print("  └── case_studies/")
print("       ├── true_positive_high_conf_gradient.png")
print("       ├── true_positive_high_conf_ig.png")
print("       ├── true_negative_high_conf_gradient.png")
print("       ├── true_negative_high_conf_ig.png")
print("       ├── false_positive_gradient.png")
print("       ├── false_positive_ig.png")
print("       ├── false_negative_gradient.png")
print("       └── false_negative_ig.png")

print("\nKey Findings:")
print(f"  • Top 3 Important Features:")
for i in range(min(3, len(importance_df))):
    feat = importance_df.iloc[i]
    print(f"    {i+1}. {feat['Feature']:15s}: {feat['Importance']:.4f}")

print(f"\n  • Faithfulness Score: {metrics['faithfulness']:.4f}")
print(f"\n  • Stability Score: {metrics['stability']:.4f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("\nUse these visualizations in your thesis:")
print("  • Chapter 4.2: Explainability Analysis")
print("  • Chapter 4.3: Feature Importance")
print("  • Chapter 4.4: Case Studies")
print("  • Chapter 5: Discussion & Interpretation")
print("="*60 + "\n")

if USING_SAVED_MODEL:
    print("TIP: To run this faster next time, execute right after training")
    print("   (model will already be in memory, no need to reload)")