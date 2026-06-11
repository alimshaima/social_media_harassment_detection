from flask import Flask, render_template, request, jsonify
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.prediction import HarassmentPredictor

app = Flask(__name__)

try:
    predictor = HarassmentPredictor()
    print("✅ Harassment Predictor loaded successfully!")
except Exception as e:
    print(f"❌ Error loading predictor: {e}")
    predictor = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if predictor is None:
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        result = predictor.predict(text)
        
        # Return in format expected by frontend
        return jsonify({
            'success': True,
            'result': {
                'text': result.get('text', text),
                'cleaned_text': result.get('normalized_text', text),
                'is_harassment': result.get('is_harassment', False),
                'confidence': result.get('confidence', 0),
                'harassment_probability': result.get('harassment_probability', 0),
                'non_harassment_probability': 1 - result.get('harassment_probability', 0),
                'detection_method': result.get('detection_method', 'unknown')
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'predictor_loaded': predictor is not None})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Harassment Detection API")
    print("="*50)
    print("📍 Server running at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)