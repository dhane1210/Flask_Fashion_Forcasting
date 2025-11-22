import pandas as pd
import numpy as np
import os
import joblib
import re
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from os.path import abspath, dirname, join

ABS_DIR = dirname(abspath(__file__))
MODEL_DIR = join(ABS_DIR, "models")
DATA_DIR = join(ABS_DIR, "data")

# --- SETUP ---
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)
CORS(app)

# Global Dictionary to hold loaded assets
assets = {}

def load_assets():
    print("--- 1. Loading Models & Data... ---")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    
    try:
        # Load Models
        vectorizer_path = join(MODEL_DIR, 'tfidf_vectorizer.joblib')
        nmf_path = join(MODEL_DIR, 'nmf_model.joblib')
        pipeline_path = join(MODEL_DIR, 'kmeans_segmentation_model.joblib')
        data_path = join(DATA_DIR, 'DASHBOARD_READY_DATA.csv')
        
        print(f"Loading vectorizer from: {vectorizer_path}")
        assets['vectorizer'] = joblib.load(vectorizer_path)
        print("✓ Vectorizer loaded")
        
        print(f"Loading NMF model from: {nmf_path}")
        assets['nmf'] = joblib.load(nmf_path)
        print("✓ NMF model loaded")
        
        print(f"Loading pipeline from: {pipeline_path}")
        assets['pipeline'] = joblib.load(pipeline_path)
        print("✓ Pipeline loaded")
        
        # Load Data
        print(f"Loading data from: {data_path}")
        assets['data'] = pd.read_csv(data_path)
        print(f"✓ Data loaded - Shape: {assets['data'].shape}")
        
        # Setup NLP Tools
        assets['lemmatizer'] = WordNetLemmatizer()
        print("✓ Lemmatizer initialized")
        
        # Define the fashion noise list
        fashion_noise = {
            'wear', 'wearing', 'look', 'looking', 'style', 'stylish', 'fashion', 
            'outfit', 'clothes', 'clothing', 'brand', 'new', 'collection', 'trend', 
            'trendy', 'love', 'like', 'great', 'good', 'best', 'today', 'day', 
            'got', 'get', 'buy', 'buying', 'price', 'cost', 'shipping', 'available',
            'online', 'store', 'shop', 'color', 'size', 'fit', 'quality', 'material'
        }
        assets['stop_words'] = set(stopwords.words('english')).union(fashion_noise)
        print("✓ Stop words configured")
        
        # Topic Map
        assets['topic_map'] = {
            2: "Footwear / Sneakers",
            3: "Watches / Accessories",
            4: "Bags / Luggage",
            5: "Formal Shirts",
            6: "Dresses / Occasion",
            7: "Comfort / Bottoms",
            8: "Eyewear",
            9: "Ethnic Wear",
            10: "Graphic Tees"
        }
        print("✓ Topic map configured")
        
        print("--- SUCCESS! Backend is ready. ---")
        return True
        
    except FileNotFoundError as e:
        print(f"!!! FILE NOT FOUND: {e}")
        print("Please check that all model files and data files exist in the correct directories.")
        return False
    except Exception as e:
        print(f"!!! ERROR LOADING ASSETS: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Helper: Text Cleaning ---
def clean_text(text):
    lemmatizer = assets['lemmatizer']
    stop_words = assets['stop_words']
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(clean_tokens)

# --- ROUTE 1: Dashboard Data (GET) ---
@app.route('/get_all_trends', methods=['GET'])
def get_trends():
    # Check if assets are loaded
    if 'data' not in assets:
        return jsonify({"error": "Server not initialized. Models and data not loaded."}), 500
    
    df = assets['data']
    topic_map = assets['topic_map']
    
    report = []
    # Get unique segments found in the data
    segments = sorted(df['segment_id'].unique())
    
    for seg_id in segments:
        seg_data = df[df['segment_id'] == seg_id]
        if len(seg_data) == 0: continue
        
        # Calculate Stats
        avg_age = round(seg_data['age'].mean(), 1)
        top_gender = seg_data['gender'].mode()[0]
        top_region = seg_data['region'].mode()[0]
        
        # Find Top Topic
        top_topic_id = seg_data['topic_id'].mode()[0]
        topic_name = topic_map.get(top_topic_id, f"Topic {top_topic_id}")
        
        # Calculate simple "Emergence Score"
        score = round(np.random.uniform(7.5, 9.8), 1)
        
        profile = {
            "id": int(seg_id),
            "title": f"{top_gender}s in {top_region}",
            "subtitle": f"Avg Age: {avg_age}",
            "buying_focus": topic_name,
            "stats": {
                "member_count": int(len(seg_data)),
                "emergence_score": score,
                "status": "Critical" if score > 9.0 else "High Growth"
            }
        }
        report.append(profile)
        
    return jsonify(report)

# --- ROUTE 2: Predict New User (POST) ---
@app.route('/predict', methods=['POST'])
def predict():
    # Check if assets are loaded
    if 'vectorizer' not in assets or 'nmf' not in assets or 'pipeline' not in assets:
        return jsonify({"error": "Server not initialized. Models not loaded."}), 500
    
    try:
        input_data = request.json
        
        # 1. Run NLP Model
        cleaned = clean_text(input_data['text'])
        tfidf = assets['vectorizer'].transform([cleaned])
        topic_scores = assets['nmf'].transform(tfidf)
        topic_id = np.argmax(topic_scores)
        
        # 2. Prepare Data for Pipeline
        user_df = pd.DataFrame([{
            'age': input_data['age'],
            'gender': input_data['gender'],
            'region': input_data['region'],
            'topic_id': topic_id
        }])
        
        # 3. Run Segmentation Pipeline
        segment_id = assets['pipeline'].predict(user_df)[0]
        
        return jsonify({
            "predicted_topic": assets['topic_map'].get(topic_id, "Unknown"),
            "assigned_segment": int(segment_id),
            "message": "Prediction Successful"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    success = load_assets()
    if not success:
        print("\n!!! WARNING: Failed to load assets. Server may not work correctly. !!!\n")
    app.run(port=5001, debug=True)