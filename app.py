"""
Crop Recommendation Web Application
==================================
A professional Flask web application for crop recommendation
with white theme and dark text UI.

Features:
- Clean, modern white background with dark text
- Input form for soil and weather parameters
- Display crop recommendations with probabilities
- Farming tips section
"""

from flask import Flask, render_template_string, request, jsonify
import joblib
import numpy as np
import os

# ============================================================================
# Configuration
# ============================================================================
app = Flask(__name__)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths - use absolute path based on script location
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# ============================================================================
# Load Model and Label Encoder
# ============================================================================
print("Loading model and label encoder...")
model = None
le = None

try:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    print(f"Model loaded successfully!")
    print(f"Number of crops: {len(le.classes_)}")
except FileNotFoundError as e:
    print(f"Error: Model files not found! {e}")
    print("Please run 'python train.py' first to train and save the model.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting to retrain model...")

# ============================================================================
# HTML Template with White Theme and Dark Text
# ============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Crop Advisory System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #ffffff; color: #1a1a1a; line-height: 1.6; }
        .header { background: linear-gradient(135deg, #1a472a 0%, #2d5a3f 100%); color: white; padding: 30px 20px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; font-weight: 600; }
        .header p { font-size: 1.1rem; opacity: 0.9; }
        .container { max-width: 1200px; margin: 40px auto; padding: 0 20px; }
        .info-section { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 12px; padding: 20px; margin-bottom: 30px; }
        .info-section h3 { color: #1a472a; margin-bottom: 15px; font-size: 1.3rem; }
        .info-section p { color: #444; font-size: 0.95rem; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .input-group { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; transition: all 0.3s ease; }
        .input-group:hover { border-color: #1a472a; box-shadow: 0 4px 12px rgba(26, 71, 42, 0.1); }
        .input-group label { display: block; color: #1a1a1a; font-weight: 600; margin-bottom: 10px; font-size: 0.95rem; }
        .input-group input { width: 100%; padding: 12px 15px; border: 2px solid #ddd; border-radius: 8px; font-size: 1rem; color: #1a1a1a; background: #fff; transition: border-color 0.3s ease; }
        .input-group input:focus { outline: none; border-color: #1a472a; }
        .input-group input::placeholder { color: #999; }
        .input-hint { font-size: 0.8rem; color: #666; margin-top: 8px; }
        .button-group { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 30px; }
        .btn { padding: 14px 30px; font-size: 1rem; font-weight: 600; border: none; border-radius: 8px; cursor: pointer; transition: all 0.3s ease; display: inline-flex; align-items: center; gap: 8px; }
        .btn-primary { background: #1a472a; color: white; }
        .btn-primary:hover { background: #2d5a3f; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(26, 71, 42, 0.3); }
        .btn-secondary { background: #f0f0f0; color: #1a1a1a; border: 2px solid #ddd; }
        .btn-secondary:hover { background: #e0e0e0; border-color: #1a472a; }
        .btn-tertiary { background: #fff; color: #1a472a; border: 2px solid #1a472a; }
        .btn-tertiary:hover { background: #1a472a; color: white; }
        .result-section { display: none; margin-top: 30px; }
        .result-section.active { display: block; }
        .result-box { background: #f8f9fa; border: 2px solid #1a472a; border-radius: 12px; padding: 30px; margin-bottom: 20px; }
        .result-box h3 { color: #1a472a; font-size: 1.5rem; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #1a472a; }
        .crop-list { list-style: none; padding: 0; }
        .crop-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 15px; margin-bottom: 10px; background: white; border-radius: 8px; border: 1px solid #e0e0e0; }
        .crop-item:first-child { background: #1a472a; color: white; border-color: #1a472a; }
        .crop-name { font-weight: 600; font-size: 1rem; }
        .crop-prob { font-weight: 600; font-size: 1rem; }
        .crop-item:first-child .crop-name, .crop-item:first-child .crop-prob { color: white; }
        .recommended { background: linear-gradient(135deg, #1a472a 0%, #2d5a3f 100%); color: white; padding: 25px; border-radius: 10px; margin-top: 20px; text-align: center; }
        .recommended h4 { font-size: 1rem; opacity: 0.9; margin-bottom: 10px; }
        .recommended .crop-name { font-size: 2rem; font-weight: 700; }
        .tips-section { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 12px; padding: 25px; margin-top: 30px; }
        .tips-section h3 { color: #1a472a; margin-bottom: 20px; font-size: 1.3rem; }
        .tips-list { list-style: none; padding: 0; }
        .tips-list li { padding: 10px 0; padding-left: 30px; position: relative; color: #333; border-bottom: 1px solid #eee; }
        .tips-list li:last-child { border-bottom: none; }
        .tips-list li::before { content: "🌱"; position: absolute; left: 0; }
        .footer { text-align: center; padding: 30px 20px; margin-top: 50px; border-top: 1px solid #e0e0e0; color: #666; }
        .loading { display: none; text-align: center; padding: 40px; }
        .loading.active { display: block; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #1a472a; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error-message { background: #fee; border: 1px solid #fcc; color: #c00; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: none; }
        .error-message.active { display: block; }
        @media (max-width: 768px) { .header h1 { font-size: 1.8rem; } .form-grid { grid-template-columns: 1fr; } .button-group { flex-direction: column; } .btn { width: 100%; justify-content: center; } }
    </style>
</head>
<body>
    <header class="header">
        <h1>🌾 Smart Crop Advisory System</h1>
        <p>Enter your soil and environmental parameters to get AI-powered crop recommendations</p>
    </header>
    <div class="container">
        <div class="info-section">
            <h3>📊 About This System</h3>
            <p>This intelligent system uses Machine Learning to analyze your soil nutrients (Nitrogen, Phosphorus, Potassium) along with environmental factors like temperature, humidity, pH level, and rainfall to recommend the most suitable crops for your conditions.</p>
        </div>
        <div id="errorMessage" class="error-message"></div>
        <form id="cropForm">
            <div class="form-grid">
                <div class="input-group">
                    <label for="N">🔬 Nitrogen (N)</label>
                    <input type="number" id="N" name="N" step="0.1" placeholder="Enter Nitrogen value" required>
                    <div class="input-hint">Range: 0 - 140</div>
                </div>
                <div class="input-group">
                    <label for="P">🔬 Phosphorus (P)</label>
                    <input type="number" id="P" name="P" step="0.1" placeholder="Enter Phosphorus value" required>
                    <div class="input-hint">Range: 0 - 145</div>
                </div>
                <div class="input-group">
                    <label for="K">🔬 Potassium (K)</label>
                    <input type="number" id="K" name="K" step="0.1" placeholder="Enter Potassium value" required>
                    <div class="input-hint">Range: 0 - 205</div>
                </div>
                <div class="input-group">
                    <label for="temperature">🌡️ Temperature (°C)</label>
                    <input type="number" id="temperature" name="temperature" step="0.1" placeholder="Enter Temperature" required>
                    <div class="input-hint">Range: 0 - 50°C</div>
                </div>
                <div class="input-group">
                    <label for="humidity">💧 Humidity (%)</label>
                    <input type="number" id="humidity" name="humidity" step="0.1" placeholder="Enter Humidity" required>
                    <div class="input-hint">Range: 0 - 100%</div>
                </div>
                <div class="input-group">
                    <label for="ph">🧪 pH Level</label>
                    <input type="number" id="ph" name="ph" step="0.1" placeholder="Enter pH value" required>
                    <div class="input-hint">Range: 3 - 9</div>
                </div>
                <div class="input-group">
                    <label for="rainfall">🌧️ Rainfall (mm)</label>
                    <input type="number" id="rainfall" name="rainfall" step="0.1" placeholder="Enter Rainfall" required>
                    <div class="input-hint">Range: 0 - 300mm</div>
                </div>
            </div>
            <div class="button-group">
                <button type="submit" class="btn btn-primary">🌾 Get Recommendation</button>
                <button type="button" class="btn btn-secondary" onclick="clearForm()">🧹 Clear</button>
                <button type="button" class="btn btn-tertiary" onclick="showTips()">💡 Farming Tips</button>
            </div>
        </form>
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px; color: #1a1a1a;">Analyzing your parameters...</p>
        </div>
        <div id="resultSection" class="result-section">
            <div class="result-box">
                <h3>📈 Crop Probability Analysis</h3>
                <ul id="cropList" class="crop-list"></ul>
                <div id="recommendedCrop" class="recommended"></div>
            </div>
        </div>
        <div id="tipsSection" class="tips-section" style="display: none;">
            <h3>💡 Essential Farming Tips</h3>
            <ul class="tips-list">
                <li><strong>Soil Testing:</strong> Test your soil regularly to understand nutrient levels and pH balance.</li>
                <li><strong>Organic Fertilizers:</strong> Use compost and organic matter to improve soil structure and fertility.</li>
                <li><strong>Crop Rotation:</strong> Rotate crops seasonally to prevent soil exhaustion and control pests.</li>
                <li><strong>Water Management:</strong> Use drip irrigation or mulching to conserve water and maintain moisture.</li>
                <li><strong>Weather Monitoring:</strong> Stay updated with weather forecasts to plan sowing and harvesting.</li>
                <li><strong>Pest Control:</strong> Use integrated pest management (IPM) and avoid excessive chemical pesticides.</li>
                <li><strong>Choose Right Crops:</strong> Select crops based on your soil type, climate, and water availability.</li>
                <li><strong>Harvest Timing:</strong> Harvest at the right maturity stage for maximum yield and quality.</li>
            </ul>
        </div>
    </div>
    <footer class="footer">
        <p>🌿 Smart Crop Advisory System | Powered by Machine Learning</p>
    </footer>
    <script>
        document.getElementById('cropForm').addEventListener('submit', function(e) {
            e.preventDefault();
            document.getElementById('resultSection').classList.remove('active');
            document.getElementById('errorMessage').classList.remove('active');
            document.getElementById('loading').classList.add('active');
            const formData = {
                N: parseFloat(document.getElementById('N').value),
                P: parseFloat(document.getElementById('P').value),
                K: parseFloat(document.getElementById('K').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                humidity: parseFloat(document.getElementById('humidity').value),
                ph: parseFloat(document.getElementById('ph').value),
                rainfall: parseFloat(document.getElementById('rainfall').value)
            };
            fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(formData) })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').classList.remove('active');
                if (data.error) { showError(data.error); } else { displayResults(data); }
            })
            .catch(error => {
                document.getElementById('loading').classList.remove('active');
                showError('An error occurred. Please try again.');
            });
        });
        function displayResults(data) {
            const resultSection = document.getElementById('resultSection');
            const cropList = document.getElementById('cropList');
            const recommendedDiv = document.getElementById('recommendedCrop');
            cropList.innerHTML = '';
            data.probabilities.forEach((item, index) => {
                const li = document.createElement('li');
                li.className = 'crop-item';
                li.innerHTML = `<span class="crop-name">${item.crop}</span><span class="crop-prob">${item.probability}%</span>`;
                cropList.appendChild(li);
            });
            recommendedDiv.innerHTML = `<h4>✅ Recommended Crop</h4><div class="crop-name">${data.recommended}</div>`;
            resultSection.classList.add('active');
            resultSection.scrollIntoView({ behavior: 'smooth' });
        }
        function showError(message) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.classList.add('active');
        }
        function clearForm() {
            document.getElementById('cropForm').reset();
            document.getElementById('resultSection').classList.remove('active');
            document.getElementById('errorMessage').classList.remove('active');
            document.getElementById('tipsSection').style.display = 'none';
        }
        function showTips() {
            const tipsSection = document.getElementById('tipsSection');
            if (tipsSection.style.display === 'none') { tipsSection.style.display = 'block'; tipsSection.scrollIntoView({ behavior: 'smooth' }); }
            else { tipsSection.style.display = 'none'; }
        }
    </script>
</body>
</html>
"""

# ============================================================================
# Routes
# ============================================================================
@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors."""
    return '', 204

@app.route('/')
def home():
    """Render the main page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if model is None or le is None:
            return jsonify({'error': 'Model not loaded. Please run train.py first.'}), 500
        
        data = request.get_json()
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))
        
        if not (0 <= N <= 140): return jsonify({'error': 'Nitrogen value should be between 0 and 140'})
        if not (0 <= P <= 145): return jsonify({'error': 'Phosphorus value should be between 0 and 145'})
        if not (0 <= K <= 205): return jsonify({'error': 'Potassium value should be between 0 and 205'})
        if not (0 <= temperature <= 50): return jsonify({'error': 'Temperature should be between 0 and 50°C'})
        if not (0 <= humidity <= 100): return jsonify({'error': 'Humidity should be between 0 and 100%'})
        if not (3 <= ph <= 9): return jsonify({'error': 'pH level should be between 3 and 9'})
        if not (0 <= rainfall <= 300): return jsonify({'error': 'Rainfall should be between 0 and 300mm'})
        
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        probabilities = model.predict_proba(input_data)[0]
        
        crop_probabilities = { crop: round(prob * 100, 2) for crop, prob in zip(le.classes_, probabilities) }
        sorted_crops = sorted(crop_probabilities.items(), key=lambda x: x[1], reverse=True)
        top_crops = sorted_crops[:5]
        
        response = {
            'probabilities': [{'crop': crop.title(), 'probability': prob} for crop, prob in top_crops],
            'recommended': top_crops[0][0].title()
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting Smart Crop Advisory System...")
    print("=" * 60)
    print("\nOpen your browser and navigate to:")
    print("  http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
