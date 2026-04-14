from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)

# ==================== MODEL LOADING ====================
model = None
try:
    model_path = 'copd_risk_model.pkl'
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            print("✅ Model loaded from Dictionary!")
        else:
            model = model_data
            print("✅ Model loaded directly!")
    else:
        print("⚠️ Model file not found! Using dummy model for testing.")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ==================== ROUTES ====================
@app.route('/')
def home():
    return render_template('prediction.html')

@app.route('/<page>')
def render_page(page):
    """Dynamic page rendering"""
    if page.endswith('.html'):
        return render_template(page)
    return render_template(f'{page}.html')

# Static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# ==================== PREDICTION API ====================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("📩 Data Received:", data)

        # Validation
        required_fields = ['age', 'bmi', 'mmrc', 'packYears', 'exacerbations', 
                          'occupationalExposure', 'fev1Fvc', 'fev1', 'oxygen', 
                          'eosinophil', 'dlco', 'aat']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Features array
        features = [
            float(data['age']),
            float(data['bmi']),
            float(data['mmrc']),
            float(data['packYears']),
            float(data['exacerbations']),
            float(data['occupationalExposure']),
            float(data['fev1Fvc']),
            float(data['fev1']),
            float(data['oxygen']),
            float(data['eosinophil']),
            float(data['dlco']),
            float(data['aat'])
        ]

        # Dummy prediction if model not loaded
        if model is None:
            risk_score = calculate_dummy_risk(features)
            prediction = 1 if risk_score > 50 else 0
            result = "High Risk (COPD Detected)" if prediction == 1 else "Low Risk (Healthy)"
            
            return jsonify({
                'prediction': str(prediction),
                'risk_score': risk_score,
                'result': result,
                'mode': 'demo'
            })

        # Actual model prediction
        final_features = [np.array(features)]
        prediction = model.predict(final_features)[0]
        
        try:
            risk_probability = model.predict_proba(final_features)[0][1] * 100
            risk_score = round(risk_probability, 2)
        except:
            risk_score = 72.5 if prediction == 1 else 27.5

        result = "High Risk (COPD Detected)" if prediction == 1 else "Low Risk (Healthy)"
        
        return jsonify({
            'prediction': str(prediction),
            'risk_score': risk_score,
            'result': result
        })

    except Exception as e:
        print("❌ Prediction Error:", e)
        return jsonify({'error': str(e)}), 500

def calculate_dummy_risk(features):
    """Dummy risk calculation for testing"""
    risk_score = 0
    if features[6] < 0.7: risk_score += 30  # fev1Fvc
    if features[7] < 80: risk_score += 20    # fev1
    if features[0] > 60: risk_score += 15    # age
    if features[3] > 20: risk_score += 25    # packYears
    if features[4] > 2: risk_score += 10     # exacerbations
    return min(risk_score, 98)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)