"""
Stress Feedback Agent
=====================
Uses the Claude API to generate lifestyle intervention suggestions from the
label predicted by the HRV stress detection model (Model G, thesis Section 3.8
and Appendix "Demonstration of the Feedback Agent").

The agent class depends only on an HTTP client. It never sees the model, the
HRV values or the confidence score: it receives the predicted label alone.

Usage:
    # Agent alone, with a manually supplied label:
    python3 stress_feedback_agent.py --prediction stress
    python3 stress_feedback_agent.py --prediction no_stress

    # Full pipeline: Model G classifies one test window of the seed-42 random
    # partition of WESAD, and its label is passed to the agent. Windows 3
    # (subject S8, stress) and 0 (subject S7, baseline) are the two windows
    # shown in the thesis.
    python3 stress_feedback_agent.py --csv ProcessedECGData_v6_wesad.csv \
        --series NNSeries_v6_wesad.npz --sample 3

    Or programmatically:
        from stress_feedback_agent import StressFeedbackAgent
        agent = StressFeedbackAgent()
        print(agent.get_feedback("stress"))
"""

import argparse
import os
import sys

import requests


# ============================================================
# CONFIGURATION
# ============================================================

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Model used in the thesis; set CLAUDE_MODEL to use another one.
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS = 512

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Model G weights behind the thesis figures and the agent demonstration.
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "saved_models", "wesad_model_G_v6.weights.h5")

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

# Local fallback content, used when the API cannot be reached.
# Mirrors the structure the system prompt requests (3-5 actionable
# suggestions, warm register, closing encouragement) so that degraded
# output is consistent in form with generated output.
FALLBACK_MESSAGES = {
    "stress": (
        "Your readings suggest you may be under some stress right now, and that is a "
        "normal response.\n\n"
        "- Take five slow breaths: in for four counts, hold for four, out for six.\n"
        "- Ground yourself: name five things you can see, four you can touch, three "
        "you can hear.\n"
        "- Release your shoulders and unclench your jaw, where tension often sits "
        "unnoticed.\n"
        "- If you can, step away for two minutes and look out of a window.\n\n"
        "This will pass, and taking a moment now makes a difference."
    ),
    "no_stress": (
        "Your readings look balanced and relaxed right now.\n\n"
        "- Keep moving gently: a short walk or some light stretching maintains this state.\n"
        "- Drink some water, as hydration supports your body's stress resilience.\n"
        "- This is a good moment for focused work or something that matters to you.\n"
        "- Notice how this feels, and what led to it.\n\n"
        "Whatever you are doing is working. Keep it up."
    ),
}


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
        self.last_error = None
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

        if not ANTHROPIC_API_KEY:
            return self._fallback(prediction, "ANTHROPIC_API_KEY is not set")

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

        # Call the Claude API. Any failure degrades to local content rather than
        # propagating an exception or returning a diagnostic string to the user.
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            # Join the text blocks. Newer models may also return other block
            # types (e.g. thinking) before the text, so the first block is not
            # necessarily text.
            feedback = "".join(block["text"] for block in data["content"]
                               if block.get("type") == "text").strip()

        except requests.exceptions.Timeout:
            return self._fallback(prediction, "request timed out")
        except requests.exceptions.ConnectionError:
            return self._fallback(prediction, "could not connect to the API")
        except requests.exceptions.HTTPError as e:
            return self._fallback(prediction, f"API returned an error: {e}")
        except requests.exceptions.RequestException as e:
            return self._fallback(prediction, f"request failed: {e}")
        except ValueError as e:
            # Covers JSON decoding failures from response.json()
            return self._fallback(prediction, f"malformed response body: {e}")
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            return self._fallback(prediction, f"unexpected response structure: {e}")

        if not feedback:
            return self._fallback(prediction, "unexpected response structure: no text "
                                  f"(stop_reason={data.get('stop_reason')})")

        self.last_error = None
        return feedback

    def _fallback(self, prediction: str, reason: str) -> str:
        """
        Return locally held content for the predicted class.

        The failure reason is recorded on the instance and written to stderr so
        that it remains visible to a developer, while the user still receives an
        appropriate intervention message rather than an error string.
        """
        self.last_error = reason
        print(f"[stress_feedback_agent] API call failed ({reason}); "
              f"using local fallback.", file=sys.stderr)
        return FALLBACK_MESSAGES[prediction]


# ============================================================
# AGENT ALONE: manually supplied label
# ============================================================

def simulate_pipeline(prediction: str):
    """Pass a manually supplied label to the agent and print its message."""
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
# FULL PIPELINE: Model G classifies a window, the agent gets the label
# ============================================================

def classify_window(csv_path: str, series_path: str, weights_path: str = DEFAULT_WEIGHTS,
                    sample_index: int = 0, seed: int = 42):
    """
    Classify one test window of WESAD with Model G.

    Reproduces the data selection, the stratified 80/20 split (seed 42) and the
    normalisation of explain_model_g.py, so that sample_index refers to the same
    test windows as the explainability analysis and the thesis.

    Returns:
        (label, probability, true_label, file) with label "stress" or "no_stress"
    """
    # Deep learning imports stay here, so that the agent class above can be
    # used without TensorFlow.
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from train_sequence import FEATURES, build

    raw = pd.read_csv(csv_path)
    nni = np.load(series_path)["nni"]
    if len(raw) != len(nni):
        raise SystemExit(f"CSV has {len(raw)} rows but the series file has {len(nni)}")
    keep = raw[FEATURES + ["label"]].notna().all(axis=1).to_numpy()
    keep &= raw["file"].str.contains("ECG_values").to_numpy()          # WESAD only
    raw, nni = raw[keep].reset_index(drop=True), nni[keep]
    x = raw[FEATURES].to_numpy(float)
    y = raw["label"].to_numpy().astype(int)

    itr, ite = train_test_split(np.arange(len(y)), test_size=0.2,
                                random_state=seed, stratify=y)
    if not 0 <= sample_index < len(ite):
        raise SystemExit(f"--sample must be between 0 and {len(ite) - 1}")
    i = ite[sample_index]

    # Normalisation fitted on the training windows only
    real = nni[itr] > 0
    mu, sd = nni[itr][real].mean(), nni[itr][real].std()
    fmu, fsd = x[itr].mean(0), x[itr].std(0)
    s = np.where(nni[i] > 0, (nni[i] - mu) / sd, 0.0)[None, :, None].astype("float32")
    k = (nni[i] > 0).astype("float32")[None, :, None]
    f = np.clip((x[i] - fmu) / fsd, -5, 5)[None, :].astype("float32")

    model = build("both", nni.shape[1], len(FEATURES))
    model.load_weights(weights_path)
    probability = float(model([s, k, f], training=False).numpy()[0, 0])

    label = "stress" if probability > 0.5 else "no_stress"
    true_label = "stress" if y[i] == 1 else "no_stress"
    return label, probability, true_label, raw["file"].iloc[i]


def load_and_run(csv_path: str, series_path: str, weights_path: str = DEFAULT_WEIGHTS,
                 sample_index: int = 0):
    """Classify one window with Model G and pass only its label to the agent."""
    label, probability, true_label, file = classify_window(
        csv_path, series_path, weights_path, sample_index)

    print("\n" + "=" * 60)
    print("  HRV STRESS DETECTION — FEEDBACK AGENT (Model G)")
    print("=" * 60)
    print(f"  Test window      : {sample_index} ({file})")
    print(f"  True label       : {'🔴 STRESS' if true_label == 'stress' else '🟢 NO STRESS'}")
    print(f"  Model prediction : {'🔴 STRESS' if label == 'stress' else '🟢 NO STRESS'}")
    print(f"  P(stress)        : {probability:.3f}")
    print("=" * 60)

    # Only the label crosses to the agent
    print("\n  Generating wellness feedback...\n")
    feedback = StressFeedbackAgent().get_feedback(label)

    print("  WELLNESS FEEDBACK:")
    print("-" * 60)
    print(feedback)
    print("=" * 60)

    return label, probability, feedback


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

    # Option B: Model G classifies a window of WESAD
    parser.add_argument("--csv", default=None,
                        help="Processed WESAD features (ProcessedECGData_v6_wesad.csv)")
    parser.add_argument("--series", default=None,
                        help="NN-interval series aligned with the CSV (NNSeries_v6_wesad.npz)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Model G weights (default: saved_models/wesad_model_G_v6.weights.h5)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Index of the test window, as in the explainability analysis (default: 0)")

    args = parser.parse_args()

    if args.csv and args.series:
        load_and_run(args.csv, args.series, args.weights, args.sample)
    elif args.prediction:
        simulate_pipeline(args.prediction)
    else:
        parser.error("Provide either --prediction, or both --csv and --series")
