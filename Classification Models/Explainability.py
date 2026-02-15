import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score

# Set style for better plots
#plt.style.use('seaborn-darkgrid')
#sns.set_palette("husl")


# ============================================================
# 1. ATTENTION VISUALIZATION
# ============================================================

class AttentionExtractor:
    """
    Extract and visualize attention weights from CNN-Transformer model
    """
    def __init__(self, model):
        self.model = model
        # Find attention layer
        self.attention_layer = None
        for layer in model.layers:
            if 'attention' in layer.name.lower():
                self.attention_layer = layer
                print(f" Found attention layer: {layer.name}")
                break
        
        if self.attention_layer is None:
            print("⚠️  No attention layer found in model")
    
    def visualize_attention_heatmap(self, x_sample, y_true, y_pred, feature_names, 
                                     save_path='attention_heatmap.png'):
        """
        Create attention heatmap for a single sample
        
        Note: TensorFlow's MultiHeadAttention doesn't expose weights directly,
        so we'll use gradient-based attention as a proxy
        """
        
        # Get model prediction
        prediction = self.model.predict(x_sample, verbose=0)[0][0]
        
        # Compute gradients to see which inputs matter most
        with tf.GradientTape() as tape:
            x_tensor = tf.constant(x_sample, dtype=tf.float32)
            tape.watch(x_tensor)
            predictions = self.model(x_tensor)
        
        # Get gradients
        gradients = tape.gradient(predictions, x_tensor)
        
        # Compute attention-like scores (absolute gradient values)
        attention_scores = np.abs(gradients.numpy()[0, :, 0])  # Shape: (8,)
        
        # Normalize to [0, 1]
        attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Bar plot of attention scores
        colors = plt.cm.RdYlGn_r(attention_scores)
        ax1.barh(feature_names, attention_scores, color=colors, alpha=0.8)
        ax1.set_xlabel('Attention Score (Importance)', fontsize=12)
        ax1.set_title(f'Feature Importance\nPrediction: {"STRESS" if prediction > 0.5 else "NO STRESS"} ({prediction:.1%})', 
                     fontsize=13, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Right: Feature values with attention overlay
        feature_values = x_sample[0, :, 0]
        
        # Create scatter plot with size based on attention
        sizes = attention_scores * 500 + 100
        scatter = ax2.scatter(range(len(feature_names)), feature_values, 
                             s=sizes, c=attention_scores, cmap='RdYlGn_r', 
                             alpha=0.7, edgecolors='black', linewidth=1.5)
        
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=45, ha='right')
        ax2.set_ylabel('Normalized Feature Value', fontsize=12)
        ax2.set_title('Feature Values (bubble size = attention)', fontsize=13, fontweight='bold')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Attention Score', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization: {save_path}")
        plt.close()
        
        return attention_scores
    
    def compare_attention_patterns(self, x_stress, x_no_stress, feature_names,
                                    save_path='attention_comparison.png'):
        """
        Compare attention patterns between stress and no-stress samples
        """
        
        # Get attention scores for both
        with tf.GradientTape() as tape:
            x_tensor = tf.constant(x_stress, dtype=tf.float32)
            tape.watch(x_tensor)
            pred = self.model(x_tensor)
        grad_stress = np.abs(tape.gradient(pred, x_tensor).numpy()[0, :, 0])
        
        with tf.GradientTape() as tape:
            x_tensor = tf.constant(x_no_stress, dtype=tf.float32)
            tape.watch(x_tensor)
            pred = self.model(x_tensor)
        grad_no_stress = np.abs(tape.gradient(pred, x_tensor).numpy()[0, :, 0])
        
        # Normalize
        grad_stress = (grad_stress - grad_stress.min()) / (grad_stress.max() - grad_stress.min() + 1e-8)
        grad_no_stress = (grad_no_stress - grad_no_stress.min()) / (grad_no_stress.max() - grad_no_stress.min() + 1e-8)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(feature_names))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, grad_stress, width, label='Stress Sample', 
                      color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, grad_no_stress, width, label='No Stress Sample', 
                      color='#3498db', alpha=0.8)
        
        ax.set_xlabel('HRV Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Score', fontsize=12, fontweight='bold')
        ax.set_title('Attention Pattern Comparison: Stress vs No Stress', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved attention comparison: {save_path}")
        plt.close()


# ============================================================
# 2. SHAP ANALYSIS
# ============================================================

class SHAPExplainer:
    """
    SHAP-based explanations for feature importance
    """
    def __init__(self, model, x_background):
        """
        Args:
            model: Trained model
            x_background: Background dataset (100-200 samples)
        """
        try:
            import shap
            self.shap = shap
        except ImportError:
            print("⚠️  SHAP not installed. Run: pip install shap")
            return
        
        self.model = model
        
        print("Initializing SHAP explainer (this may take 2-3 minutes)...")
        self.explainer = shap.DeepExplainer(
            model,
            x_background[:100]  # Use subset as background
        )
        print(" SHAP explainer ready!")
    
    def global_feature_importance(self, x_samples, feature_names, 
                                   save_path='shap_global_importance.png'):
        """
        Global feature importance across multiple samples
        """
        print("Computing SHAP values (this may take a minute)...")
        shap_values = self.explainer.shap_values(x_samples)
        
        # Reshape (samples, features)
        shap_reshaped = shap_values[0].reshape(len(x_samples), -1)
        x_reshaped = x_samples.reshape(len(x_samples), -1)
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: SHAP summary plot
        plt.sca(ax1)
        self.shap.summary_plot(
            shap_reshaped,
            x_reshaped,
            feature_names=feature_names,
            show=False,
            plot_type='dot'
        )
        ax1.set_title('SHAP Feature Importance (Detailed)', fontsize=13, fontweight='bold')
        
        # Right: Bar plot
        mean_abs_shap = np.abs(shap_reshaped).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)
        
        colors = plt.cm.RdYlGn_r(mean_abs_shap[sorted_idx] / mean_abs_shap.max())
        ax2.barh(np.array(feature_names)[sorted_idx], mean_abs_shap[sorted_idx], 
                color=colors, alpha=0.8)
        ax2.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=12)
        ax2.set_title('Global Feature Importance Ranking', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved SHAP global importance: {save_path}")
        plt.close()
        
        # Print rankings
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_abs_shap
        }).sort_values('Importance', ascending=False)
        
        print("GLOBAL FEATURE IMPORTANCE (SHAP)")
        for i, row in importance_df.iterrows():
            print(f"  {row['Feature']:20s}: {row['Importance']:.4f}")
        
        return importance_df
    
    def explain_single_prediction(self, x_sample, feature_names, prediction,
                                   save_path='shap_single_explanation.png'):
        """
        Explain a single prediction with waterfall plot
        """
        shap_values = self.explainer.shap_values(x_sample)
        
        # Reshape
        shap_flat = shap_values[0].reshape(-1)
        x_flat = x_sample.reshape(-1)
        
        # Create waterfall-style plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(shap_flat))[::-1]
        
        # Create data
        features_sorted = [feature_names[i] for i in sorted_idx]
        shap_sorted = shap_flat[sorted_idx]
        values_sorted = x_flat[sorted_idx]
        
        # Bar colors based on contribution
        colors = ['#e74c3c' if s > 0 else '#3498db' for s in shap_sorted]
        
        # Create bars
        bars = ax.barh(features_sorted, shap_sorted, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (feat, shap_val, val) in enumerate(zip(features_sorted, shap_sorted, values_sorted)):
            label = f"value={val:.2f}"
            x_pos = shap_val + (0.01 if shap_val > 0 else -0.01)
            ha = 'left' if shap_val > 0 else 'right'
            ax.text(x_pos, i, label, va='center', ha=ha, fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
        ax.set_title(f'Feature Contributions\nPrediction: {"STRESS" if prediction > 0.5 else "NO STRESS"} ({prediction:.1%})', 
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.8, label='Increases stress probability'),
            Patch(facecolor='#3498db', alpha=0.8, label='Decreases stress probability')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved SHAP single explanation: {save_path}")
        plt.close()


# ============================================================
# 3. INTEGRATED GRADIENTS
# ============================================================

class IntegratedGradientsExplainer:
    """
    Integrated Gradients for faithful feature attribution
    """
    def __init__(self, model):
        self.model = model
    
    @tf.function
    def compute_gradients(self, inputs):
        """Compute gradients of prediction w.r.t. inputs"""
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = self.model(inputs)
        return tape.gradient(predictions, inputs)
    
    def integrated_gradients(self, x_sample, baseline=None, steps=50):
        """
        Compute integrated gradients
        """
        if baseline is None:
            baseline = tf.zeros_like(x_sample)
        
        # Generate interpolated inputs
        alphas = tf.linspace(0.0, 1.0, steps + 1)
        interpolated_inputs = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (x_sample - baseline)
            interpolated_inputs.append(interpolated)
        
        interpolated_inputs = tf.concat(interpolated_inputs, axis=0)
        
        # Compute gradients
        gradients = self.compute_gradients(interpolated_inputs)
        
        # Approximate integral using trapezoidal rule
        avg_gradients = tf.reduce_mean(gradients, axis=0, keepdims=True)
        
        # Multiply by input difference
        attributions = (x_sample - baseline) * avg_gradients
        
        return attributions.numpy()
    
    def visualize_attributions(self, x_sample, feature_names, prediction,
                               save_path='ig_attributions.png'):
        """
        Visualize integrated gradients
        """
        # Compute attributions
        attributions = self.integrated_gradients(x_sample)
        
        # Flatten
        attr_flat = attributions.reshape(-1)
        x_flat = x_sample.numpy().reshape(-1)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top: Attributions
        colors = ['#e74c3c' if a > 0 else '#3498db' for a in attr_flat]
        bars = ax1.barh(feature_names, attr_flat, color=colors, alpha=0.8)
        ax1.set_xlabel('Attribution (Contribution to Stress Prediction)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Integrated Gradients Explanation\nPrediction: {"STRESS" if prediction > 0.5 else "NO STRESS"} ({prediction:.1%})', 
                     fontsize=13, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax1.grid(axis='x', alpha=0.3)
        
        # Bottom: Feature values
        ax2.barh(feature_names, x_flat, color='gray', alpha=0.6)
        ax2.set_xlabel('Normalized Feature Value', fontsize=12, fontweight='bold')
        ax2.set_title('Input Feature Values', fontsize=13, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved IG visualization: {save_path}")
        plt.close()
        
        return attributions


# ============================================================
# 4. COMPREHENSIVE CASE STUDIES
# ============================================================

def create_case_studies(model, x_test, y_test, feature_names, output_dir='case_studies'):
    """
    Create detailed case study explanations for different prediction scenarios
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    y_pred_proba = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Find interesting cases
    cases = {
        'true_positive_high_conf': np.where((y_test == 1) & (y_pred == 1) & (y_pred_proba > 0.8))[0],
        'true_negative_high_conf': np.where((y_test == 0) & (y_pred == 0) & (y_pred_proba < 0.2))[0],
        'false_positive': np.where((y_test == 0) & (y_pred == 1))[0],
        'false_negative': np.where((y_test == 1) & (y_pred == 0))[0],
    }
    
    print("\n" + "="*60)
    print("CREATING CASE STUDIES")
    print("="*60)
    
    # Initialize explainers
    attention_extractor = AttentionExtractor(model)
    ig_explainer = IntegratedGradientsExplainer(model)
    
    for case_name, indices in cases.items():
        if len(indices) == 0:
            continue
        
        # Pick first example
        idx = indices[0]
        x_sample = x_test[idx:idx+1]
        y_true = y_test[idx]
        y_pred_val = y_pred_proba[idx]
        
        print(f"\n{case_name.replace('_', ' ').title()}:")
        print(f"   True: {'Stress' if y_true == 1 else 'No Stress'}")
        print(f"   Predicted: {'Stress' if y_pred_val > 0.5 else 'No Stress'} ({y_pred_val:.1%})")
        
        # Attention
        attention_extractor.visualize_attention_heatmap(
            x_sample, y_true, y_pred_val, feature_names,
            save_path=f'{output_dir}/{case_name}_attention.png'
        )
        
        # Integrated Gradients
        ig_explainer.visualize_attributions(
            tf.constant(x_sample, dtype=tf.float32),
            feature_names,
            y_pred_val,
            save_path=f'{output_dir}/{case_name}_ig.png'
        )
    
    print("\n Case studies complete!")
    print(f"   Saved to: {output_dir}/")
    print("="*60 + "\n")


# ============================================================
# 5. QUANTITATIVE EVALUATION
# ============================================================

def evaluate_explainability_metrics(model, x_test, y_test, feature_names):
    """
    Compute quantitative metrics for explainability
    """
    print("\n" + "="*60)
    print("EXPLAINABILITY METRICS EVALUATION")
    print("="*60)
    
    # 1. Faithfulness: Performance drop when removing top features
    print("\n Computing Faithfulness...")
    
    # Get baseline accuracy
    y_pred_baseline = model.predict(x_test, verbose=0).flatten()
    y_pred_baseline_binary = (y_pred_baseline > 0.5).astype(int)
    acc_baseline = accuracy_score(y_test, y_pred_baseline_binary)
    
    # Compute feature importance via gradients
    feature_importance = np.zeros(len(feature_names))
    
    for i in range(len(x_test)):
        with tf.GradientTape() as tape:
            x_tensor = tf.constant(x_test[i:i+1], dtype=tf.float32)
            tape.watch(x_tensor)
            pred = model(x_tensor)
        grad = tape.gradient(pred, x_tensor)
        feature_importance += np.abs(grad.numpy()[0, :, 0])
    
    feature_importance /= len(x_test)
    
    # Remove top 2 features
    top_k = 2
    top_indices = np.argsort(feature_importance)[-top_k:]
    
    x_test_masked = x_test.copy()
    x_test_masked[:, top_indices, :] = 0
    
    y_pred_masked = model.predict(x_test_masked, verbose=0).flatten()
    y_pred_masked_binary = (y_pred_masked > 0.5).astype(int)
    acc_masked = accuracy_score(y_test, y_pred_masked_binary)
    
    faithfulness = acc_baseline - acc_masked
    
    print(f"   Baseline accuracy: {acc_baseline:.4f}")
    print(f"   Accuracy after removing top-{top_k} features: {acc_masked:.4f}")
    print(f"   Faithfulness (drop): {faithfulness:.4f}")
    print(f"   → {'Good' if faithfulness > 0.05 else 'Weak'} (>0.05 is good)")
    
    # 2. Stability: Consistency across similar inputs
    print("\n  Computing Stability...")
    
    n_pairs = 20
    similarities = []
    
    from scipy.stats import spearmanr
    
    for i in range(min(n_pairs, len(x_test)-1)):
        # Get two consecutive samples (often similar)
        with tf.GradientTape() as tape:
            x1 = tf.constant(x_test[i:i+1], dtype=tf.float32)
            tape.watch(x1)
            pred1 = model(x1)
        grad1 = np.abs(tape.gradient(pred1, x1).numpy()[0, :, 0])
        
        with tf.GradientTape() as tape:
            x2 = tf.constant(x_test[i+1:i+2], dtype=tf.float32)
            tape.watch(x2)
            pred2 = model(x2)
        grad2 = np.abs(tape.gradient(pred2, x2).numpy()[0, :, 0])
        
        # Compute correlation
        corr, _ = spearmanr(grad1, grad2)
        if not np.isnan(corr):
            similarities.append(corr)
    
    stability = np.mean(similarities)
    
    print(f"   Mean explanation similarity: {stability:.4f}")
    print(f"   → {'Good' if stability > 0.6 else 'Moderate' if stability > 0.4 else 'Weak'} (>0.6 is good)")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Faithfulness: {faithfulness:.4f} ({' Good' if faithfulness > 0.05 else '⚠️  Weak'})")
    print(f"  Stability:    {stability:.4f} ({' Good' if stability > 0.6 else '⚠️  Moderate'})")
    print("="*60 + "\n")
    
    return {
        'faithfulness': faithfulness,
        'stability': stability,
        'feature_importance': dict(zip(feature_names, feature_importance))
    }