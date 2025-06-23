from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data and convert to float
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])[0]
        result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'
    except Exception as e:
        result = f"Error in prediction: {str(e)}"
    return render_template('result.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)