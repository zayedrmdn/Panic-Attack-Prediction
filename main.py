from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get user input
            features = [
                float(request.form["cat__Lifestyle Factors_Sleep quality"]),
                float(request.form["cat__Current Stressors_High"]),
                float(request.form["cat__Symptoms_Panic attacks"]),
                float(request.form["cat__Impact on Life_Significant"]),
                float(request.form["cat__Severity_Severe"]),
                float(request.form["cat__Lifestyle Factors_Exercise"]),
                float(request.form["cat__Lifestyle Factors_Diet"]),
            ]

            # Convert input to NumPy array
            input_array = np.array([features])

            # Make prediction
            prediction = model.predict(input_array)[0]
            result = "Panic Attack Detected" if prediction == 1 else "No Panic Attack"

            return render_template("result.html", result=result)

        except Exception as e:
            return render_template("result.html", result=f"Error: {e}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
