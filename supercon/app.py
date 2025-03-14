from flask import Flask, request, jsonify, render_template
import numpy as np
from .supercon_run import load_model_and_predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
            
        # Convert input to array of 81 features
        try:
            features = np.array([data['features']])
            if len(data['features']) != 81:
                return jsonify({'error': 'Please provide exactly 81 numerical values'}), 400
        except (ValueError, IndexError):
            return jsonify({'error': 'Invalid input. Please provide 81 numerical values'}), 400
            
        prediction = load_model_and_predict(features)
        return jsonify({
            'input': data['features'],
            'prediction': float(prediction[0][0])
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
