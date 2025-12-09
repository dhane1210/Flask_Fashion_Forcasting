import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import TOPIC_MAP
from utils import load_ai_assets, clean_text

app = Flask(__name__)
CORS(app)

# Load assets (Models & Dashboard JSON Cache)
assets = load_ai_assets()

# --- FIX: Add a Home Route to stop 404 Errors ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "active",
        "service": "Fashion Trend AI Microservice",
        "endpoints": {
            "GET /get_all_trends": "Get Dashboard Report",
            "POST /predict": "Predict Segment for New User"
        }
    })

@app.route('/get_all_trends', methods=['GET'])
def get_trends():
    # FIXED: Check for 'report_data' (JSON) instead of 'data' (CSV)
    if 'report_data' in assets and assets['report_data']:
        return jsonify(assets['report_data'])
    
    # Fallback error if pipeline hasn't been run
    return jsonify({
        "error": "Dashboard data not found.", 
        "solution": "Please run 'python master_pipeline.py' to generate the dashboard cache."
    }), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'vectorizer' not in assets: return jsonify({"error": "Models not loaded"}), 500
    try:
        data = request.json
        cleaned = clean_text(data['text'])
        tfidf = assets['vectorizer'].transform([cleaned])
        topic_id = np.argmax(assets['nmf'].transform(tfidf))
        
        user_df = pd.DataFrame([{
            'age': data['age'], 'gender': data['gender'], 
            'region': data['region'], 'topic_id': topic_id,
            'region_clean': data['region']
        }])
        
        segment_id = assets['pipeline'].predict(user_df)[0]
        
        return jsonify({
            "predicted_topic": TOPIC_MAP.get(topic_id, "Unknown"),
            "assigned_segment": int(segment_id),
            "message": "Prediction Successful"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)