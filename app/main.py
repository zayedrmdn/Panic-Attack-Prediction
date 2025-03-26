import os
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__, template_folder="templates")

# Load the exact pipeline you just trained
BASE = os.path.dirname(__file__)
pipeline = joblib.load(os.path.join(BASE, "model_pipeline.pkl"))

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

@app.route("/")
def home():
    # Grab the fitted OneHotEncoder from your pipeline
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_cols = FEATURES

    # Build a dict: { feature_name: [allowed values...] }
    categories = dict(zip(cat_cols, encoder.categories_))

    return render_template("index.html", categories=categories)

@app.route("/predict_form", methods=["POST"])
def predict_form():
    data = {f: request.form.get(f, "") for f in FEATURES}
    df = pd.DataFrame([data])

    # Ensure no empty strings
    for col in FEATURES:
        df[col] = df[col].astype(str).str.strip().replace("", "Unknown")

    pred = pipeline.predict(df)[0]

    # Convert probability to percentage
    prob = pipeline.predict_proba(df)[0]
    prob_percent = [f"{p * 100:.2f}%" for p in prob]

    return render_template("result.html", prediction=int(pred), probability=prob_percent)

if __name__ == "__main__":
    app.run(debug=True)
