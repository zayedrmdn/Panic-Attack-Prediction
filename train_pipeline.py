import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "panic_disorder_dataset_training.csv")

# ─── Features & Target ────────────────────────────────────────────────────
FEATURES = [
    "Current Stressors",
    "Symptoms",
    "Severity",
    "Psychiatric History",
    "Substance Use",
    "Coping Mechanisms",
    "Impact on Life",
    "Lifestyle Factors"
]
TARGET = "Panic Disorder Diagnosis"

# ─── Load Data ────────────────────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV)
X = df[FEATURES]
y = df[TARGET]

# ─── Preprocessing (One-Hot Encode Categories) ────────────────────────────
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)
])

# ─── Model ────────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
])

# ─── Train & Save ─────────────────────────────────────────────────────────
print("🚀 Training pipeline...")
pipeline.fit(X, y)

feat_count = pipeline.named_steps["preprocessor"].get_feature_names_out().shape[0]
print(f"✅ Trained on {feat_count} features")

MODEL_PATH = os.path.join("app", "model_pipeline.pkl")
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ Pipeline saved to {MODEL_PATH}")

# ─── Learning Curve ───────────────────────────────────────────────────────
print("\n📊 Generating Learning Curve...")

train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, cv=5, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10)
)

# Compute means and standard deviations
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# ─── Plot Learning Curve ──────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", label="Training Score", color="blue")
plt.plot(train_sizes, test_mean, "s-", label="Validation Score", color="red")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (Random Forest)")
plt.legend()
plt.grid()
plt.savefig("learning_curve.png")  # Save instead of show
print("📊 Learning Curve saved as learning_curve.png")
