import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Ensure project root is on PYTHONPATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.evaluation import evaluate_models

# --------------------------------------------------
# Load evaluation results
# --------------------------------------------------
df = evaluate_models()

errors = {
    "TF-IDF": df["TF-IDF Error"].mean(),
    "BERT": df["BERT Error"].mean()
}

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure()
plt.bar(errors.keys(), errors.values())
plt.title("Average Error vs Human Judgment")
plt.ylabel("Mean Absolute Error")
plt.xlabel("Model")
plt.tight_layout()
plt.show()