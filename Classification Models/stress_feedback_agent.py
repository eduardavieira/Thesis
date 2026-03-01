"""
Stress Feedback Agent
=====================
Uses the Claude API to generate personalized lifestyle intervention
suggestions based on the output of the HRV stress detection model.

Usage:
    # Run with a manual prediction:
    python stress_feedback_agent.py --prediction stress
    python stress_feedback_agent.py --prediction no_stress

    # Run directly from saved model + CSV data:
    python stress_feedback_agent.py --model saved_models/used_model.h5 --data /path/to/ProcessedECGData.csv

    Or use programmatically:
        from stress_feedback_agent import StressFeedbackAgent
        agent = StressFeedbackAgent()
        response = agent.get_feedback("stress")
        print(response)
"""

import argparse
import requests
import json
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 512

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# System prompt — defines the agent's persona and behaviour
SYSTEM_PROMPT = """You are a compassionate wellness assistant embedded in a smartwatch stress monitoring application. 
Your role is to provide brief, practical, and evidence-based lifestyle intervention suggestions based on the user's current stress state detected from their heart rate variability (HRV).

Guidelines:
- Keep responses concise and actionable (3-5 suggestions maximum)
- Use a warm, supportive, non-alarming tone
- Suggestions should be realistic for someone using a smartwatch (i.e. they may be at work, commuting, or at home)
- Never diagnose or provide medical advice
- Focus on immediate, short-term interventions the user can do right now
- For stress: focus on grounding, breathing, and short breaks
- For no-stress: reinforce healthy habits and encourage maintenance
- Always end with one encouraging sentence
"""

# Prompts sent to the model for each prediction case
STRESS_PROMPT = """The HRV stress detection model has classified the user's current physiological state as STRESSED.

Please provide 3 to 5 practical lifestyle intervention suggestions the user can act on right now to help manage and reduce their stress. 
Focus on interventions that are quick (under 10 minutes), require no equipment, and are suitable for any environment (office, home, public transport).
Keep the tone warm, calm and encouraging."""

NO_STRESS_PROMPT = """The HRV stress detection model has classified the user's current physiological state as NOT STRESSED.

Please provide 3 to 5 positive reinforcement messages and wellness tips to help the user maintain this healthy state.
Encourage them to keep up good habits such as regular breaks, hydration, movement, and sleep hygiene.
Keep the tone warm, upbeat and motivating."""


# ============================================================
# AGENT CLASS
# ============================================================

class StressFeedbackAgent:
    """
    Agent that calls the Claude API to generate lifestyle feedback
    based on the HRV model's stress/no-stress prediction.
    """

    def __init__(self):
        self.api_url = ANTHROPIC_API_URL
        self.model = MODEL
        self.headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": ANTHROPIC_API_KEY
        }

    def get_feedback(self, prediction: str) -> str:
        """
        Get lifestyle feedback from Claude based on the stress prediction.

        Args:
            prediction: Either "stress" or "no_stress"

        Returns:
            String containing Claude's feedback message
        """
        prediction = prediction.strip().lower()

        if prediction not in ("stress", "no_stress"):
            raise ValueError(f"Invalid prediction: '{prediction}'. Must be 'stress' or 'no_stress'.")

        # Select the appropriate user prompt
        user_prompt = STRESS_PROMPT if prediction == "stress" else NO_STRESS_PROMPT

        # Build the API request body
        payload = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        }

        # Call the Claude API
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

        except requests.exceptions.Timeout:
            return "[Error] The request to the Claude API timed out. Please try again."
        except requests.exceptions.ConnectionError:
            return "[Error] Could not connect to the Claude API. Check your internet connection."
        except requests.exceptions.HTTPError as e:
            return f"[Error] API request failed: {e}"

        # Parse the response
        data = response.json()

        # Extract the text content from the response
        try:
            feedback = data["content"][0]["text"]
        except (KeyError, IndexError):
            return "[Error] Unexpected response format from Claude API."

        return feedback


# ============================================================
# DEMO / SIMULATION
# ============================================================

def simulate_pipeline(prediction: str):
    """
    Simulates the full pipeline:
    HRV Model Prediction → Stress Feedback Agent → User Feedback
    """
    print("=" * 60)
    print("  HRV STRESS DETECTION — FEEDBACK AGENT")
    print("=" * 60)
    print(f"\n  Model Prediction : {'🔴 STRESS DETECTED' if prediction == 'stress' else '🟢 NO STRESS DETECTED'}")
    print(f"  Sending to agent...\n")
    print("=" * 60)

    agent = StressFeedbackAgent()
    feedback = agent.get_feedback(prediction)

    print("\n  WELLNESS FEEDBACK:")
    print("-" * 60)
    print(feedback)
    print("=" * 60)


# ============================================================
# INTEGRATION EXAMPLE — with live model object
# ============================================================

def integrate_with_model(model, x_sample):
    """
    Example of how to integrate the feedback agent directly
    with your trained Keras model from Models.py

    Args:
        model: Your trained Keras model (e.g. Model D)
        x_sample: A single reshaped HRV sample (1, 8, 1)

    Returns:
        prediction_label: "stress" or "no_stress"
        feedback: Claude's feedback string
    """
    import numpy as np

    # Get model prediction
    probability = model.predict(x_sample, verbose=0)[0][0]
    prediction_label = "stress" if probability > 0.5 else "no_stress"
    confidence = probability if probability > 0.5 else 1 - probability

    print(f"Model confidence: {confidence:.1%} ({prediction_label})")

    # Get feedback from agent
    agent = StressFeedbackAgent()
    feedback = agent.get_feedback(prediction_label)

    return prediction_label, feedback


# ============================================================
# LOAD SAVED MODEL AND RUN
# ============================================================

def load_and_run(model_path: str, data_path: str, sample_index: int = 0):
    """
    Loads a saved .h5 model and runs the full pipeline on a sample
    from your processed CSV — no retraining needed.

    Args:
        model_path:   Path to saved model, e.g. 'saved_models/used_model.h5'
        data_path:    Path to your ProcessedECGData.csv
        sample_index: Which test sample to run (default: 0 = first)

    Usage:
        python stress_feedback_agent.py --load_model saved_models/used_model.h5 \
                                        --data /home/eduarda/Desktop/ProcessedECGData.csv \
                                        --sample 0
    """
    import numpy as np
    from tensorflow.keras.models import load_model
    from Data import load_features, normalize_data

    # ---- Load model ----
    print(f"\nLoading model from: {model_path}")
    try:
        # custom_object_scope handles PositionalEncoding layer in Model D
        from tensorflow.keras.utils import custom_object_scope
        from Models import PositionalEncoding
        with custom_object_scope({'PositionalEncoding': PositionalEncoding}):
            model = load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[Error] Could not load model: {e}")
        return

    # ---- Load and preprocess data ----
    print(f"Loading data from: {data_path}")
    x_features, y_label, _ = load_features(data_path)

    # Split exactly as in main.py to get the same test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_label,
        test_size=0.2,
        random_state=42,       # same seed as main.py — reproducible split
        stratify=y_label
    )

    # Normalize using training set parameters (no data leakage)
    x_train_norm, x_test_norm = normalize_data(x_train, x_test)

    # Reshape to (samples, timesteps, channels)
    nrows_test, ncols = x_test_norm.shape
    x_test_reshape = x_test_norm.reshape(nrows_test, ncols, 1)

    # ---- Select sample ----
    if sample_index >= len(x_test_reshape):
        print(f"[Error] sample_index {sample_index} out of range. Test set has {len(x_test_reshape)} samples.")
        return

    x_sample = x_test_reshape[sample_index:sample_index + 1]  # shape (1, 8, 1)
    true_label = "stress" if y_test[sample_index] == 1 else "no_stress"

    # ---- Predict ----
    probability = model.predict(x_sample, verbose=0)[0][0]
    prediction_label = "stress" if probability > 0.5 else "no_stress"
    confidence = probability if probability > 0.5 else 1 - probability

    # ---- Display prediction ----
    print("\n" + "=" * 60)
    print("  HRV STRESS DETECTION — FEEDBACK AGENT")
    print("=" * 60)
    print(f"  Sample index     : {sample_index}")
    print(f"  True label       : {'🔴 STRESS' if true_label == 'stress' else '🟢 NO STRESS'}")
    print(f"  Model prediction : {'🔴 STRESS' if prediction_label == 'stress' else '🟢 NO STRESS'}")
    print(f"  Confidence       : {confidence:.1%}")
    print(f"  Correct          : {'✅ Yes' if prediction_label == true_label else '❌ No'}")
    print("=" * 60)

    # ---- Get feedback from Claude ----
    print("\n  Generating wellness feedback...\n")
    agent = StressFeedbackAgent()
    feedback = agent.get_feedback(prediction_label)

    print("  WELLNESS FEEDBACK:")
    print("-" * 60)
    print(feedback)
    print("=" * 60)

    return prediction_label, confidence, feedback


# ============================================================
# LOAD SAVED MODEL AND RUN
# ============================================================

def load_and_run(model_path: str, data_path: str, sample_index: int = 0):
    """
    Load the saved Model D from .h5 file, preprocess a sample
    from the CSV, run prediction, and get feedback.

    This replicates the exact same preprocessing pipeline from
    Data.py and main.py so results are consistent with training.

    Args:
        model_path:   Path to saved model, e.g. 'saved_models/used_model.h5'
        data_path:    Path to ProcessedECGData.csv
        sample_index: Which test sample to run (default: 0 = first)
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    print("=" * 60)
    print("  LOADING SAVED MODEL")
    print("=" * 60)

    # ---- 1. Load saved model ----
    # PositionalEncoding is a custom layer defined in Models.py
    # We must redefine it here so Keras can deserialize the .h5 file
    class PositionalEncoding(tf.keras.layers.Layer):
        def __init__(self, d_model, max_len=100, **kwargs):
            super(PositionalEncoding, self).__init__(**kwargs)
            self.d_model = d_model
            self.max_len = max_len

        def build(self, input_shape):
            position = np.arange(self.max_len)[:, np.newaxis]
            div_term = np.exp(
                np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model)
            )
            pos_encoding = np.zeros((self.max_len, self.d_model))
            pos_encoding[:, 0::2] = np.sin(position * div_term)
            pos_encoding[:, 1::2] = np.cos(position * div_term)
            self.pos_encoding = tf.constant(
                pos_encoding[np.newaxis, :, :], dtype=tf.float32
            )

        def call(self, x):
            seq_len = tf.shape(x)[1]
            return x + self.pos_encoding[:, :seq_len, :]

        def get_config(self):
            config = super().get_config()
            config.update({"d_model": self.d_model, "max_len": self.max_len})
            return config

    print(f"\n  Loading model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"PositionalEncoding": PositionalEncoding}
    )
    print("  Model loaded successfully!")
    model.summary()

    # ---- 2. Load and preprocess data (mirrors Data.py exactly) ----
    print(f"\n  Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    data = data.dropna(subset=[data.columns[1]])
    data = data.dropna(subset=data.columns[2:10])

    x_features = data.iloc[:, 2:10].to_numpy()
    y_label    = data.iloc[:, 1].to_numpy()

    print(f"  Loaded {len(x_features)} samples")

    # ---- 3. Replicate train/test split from main.py ----
    # Same random_state=42 and stratify as during training
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_label,
        test_size=0.2,
        random_state=42,
        stratify=y_label
    )

    # ---- 4. Replicate normalization from Data.py ----
    Q1 = np.percentile(x_train, 25, axis=0)
    Q3 = np.percentile(x_train, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    x_train_clipped = np.clip(x_train, lower_bound, upper_bound)
    x_test_clipped  = np.clip(x_test,  lower_bound, upper_bound)

    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train_clipped)
    x_test_norm  = scaler.transform(x_test_clipped)

    x_train_norm = np.clip(x_train_norm, -5, 5)
    x_test_norm  = np.clip(x_test_norm,  -5, 5)

    # ---- 5. Reshape to (samples, timesteps, channels) ----
    ncols = x_test_norm.shape[1]  # 8 features
    x_test_reshape = x_test_norm.reshape(x_test_norm.shape[0], ncols, 1)

    # ---- 6. Pick sample and predict ----
    sample_index = min(sample_index, len(x_test_reshape) - 1)
    x_sample  = x_test_reshape[sample_index:sample_index + 1]  # shape (1, 8, 1)
    true_label = int(y_test[sample_index])

    probability = model.predict(x_sample, verbose=0)[0][0]
    predicted   = "stress" if probability > 0.5 else "no_stress"
    confidence  = probability if probability > 0.5 else 1 - probability

    print(f"\n  Sample index    : {sample_index}")
    print(f"  True label      : {'Stress' if true_label == 1 else 'No Stress'}")
    print(f"  Predicted       : {'🔴 STRESS' if predicted == 'stress' else '🟢 NO STRESS'}")
    print(f"  Confidence      : {confidence:.1%}")

    # ---- 7. Get feedback from agent ----
    print("\n  Fetching feedback from Claude API...")
    agent = StressFeedbackAgent()
    feedback = agent.get_feedback(predicted)

    print("\n" + "=" * 60)
    print("  WELLNESS FEEDBACK:")
    print("-" * 60)
    print(feedback)
    print("=" * 60)

    return predicted, confidence, feedback


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stress Feedback Agent — generates lifestyle suggestions based on HRV model output."
    )

    # Option A: manual prediction
    parser.add_argument(
        "--prediction",
        type=str,
        choices=["stress", "no_stress"],
        default=None,
        help="Manually specify prediction: 'stress' or 'no_stress'"
    )

    # Option B: load saved model + CSV
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to saved .h5 model, e.g. 'saved_models/used_model.h5'"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to ProcessedECGData.csv"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Index of test sample to predict on (default: 0)"
    )

    args = parser.parse_args()

    # Route to correct mode
    if args.model and args.data:
        # Load saved model and run on real data
        load_and_run(args.model, args.data, args.sample)
    elif args.prediction:
        # Manual prediction mode
        simulate_pipeline(args.prediction)
    else:
        parser.error("Provide either --prediction OR both --model and --data")