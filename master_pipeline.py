import pandas as pd
import numpy as np
import joblib
import requests
import json
import warnings
import os
import sys
import subprocess
import time
from datetime import datetime

# ML Imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from nltk.sentiment import SentimentIntensityAnalyzer

# Project Imports
from config import PATHS, TOPIC_MAP, SPRING_BOOT_URL, VALID_TOPICS, BASE_DIR
from utils import clean_text

warnings.filterwarnings('ignore')

def calculate_metrics(df):
    """Calculates Sentiment and Growth for the Dashboard Cache"""
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    
    # Sentiment
    print("   ...Calculating Sentiment")
    df['sentiment'] = df['text_content'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    
    # Forecasting
    print("   ...Forecasting Growth")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['month'] = df['timestamp'].dt.to_period('M')
    
    monthly_counts = df.groupby(['topic_id', 'month']).size().reset_index(name='count')
    metrics = {}

    for topic in VALID_TOPICS:
        history = monthly_counts[monthly_counts['topic_id'] == topic]
        growth_pct = 0
        sentiment_avg = df[df['topic_id'] == topic]['sentiment'].mean()
        
        if len(history) > 2:
            try:
                series = history['count'].values.astype(float)
                model = ExponentialSmoothing(series, trend='add').fit()
                forecast = model.forecast(1)
                last_val = series[-1] if series[-1] != 0 else 1
                growth_pct = round(((forecast[0] - last_val) / last_val) * 100, 1)
            except: pass
        
        metrics[str(topic)] = {
            "growth_pct": growth_pct,
            "sentiment": round(sentiment_avg, 2)
        }
    return metrics

def run_full_pipeline():
    print("\n" + "="*50)
    print("🚀 STARTING FULL DATA PIPELINE")
    print("="*50)

    # --- 1. LOAD RAW DATA ---
    print(f"\n[Step 1] Loading Raw Data...")
    if not os.path.exists(PATHS['raw_csv']):
        print(f"❌ Critical Error: Input file missing at {PATHS['raw_csv']}")
        sys.exit(1)

    df = pd.read_csv(PATHS['raw_csv'])
    print(f"✅ Loaded {len(df)} rows.")

    # --- 2. NLP TRAINING ---
    print("\n[Step 2] Training NLP Model (NMF)...")
    try:
        # Load existing vectorizer/NMF if available (faster), or you can retrain here
        # Assuming we use the ones you have, or we reload them. 
        # For a full reset, uncomment the training code lines in your previous versions.
        vectorizer = joblib.load(PATHS['vectorizer'])
        nmf = joblib.load(PATHS['nmf'])
        print("   Loaded existing NLP models.")
    except:
        print("❌ Models not found. Please ensure base models exist in 'models/'.")
        sys.exit(1)

    df['clean_text'] = df['text_content'].apply(clean_text)
    tfidf = vectorizer.transform(df['clean_text'])
    topic_matrix = nmf.transform(tfidf)
    df['topic_id'] = np.argmax(topic_matrix, axis=1)

    # --- 3. SEGMENTATION TRAINING ---
    print("\n[Step 3] Training Segmentation Model...")
    
    # Filter 'clean' data for training the clusters
    df_train = df[df['topic_id'].isin(VALID_TOPICS)].copy()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'region_clean', 'topic_id'])
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=5, random_state=42, n_init=10))
    ])
    
    pipeline.fit(df_train)
    joblib.dump(pipeline, PATHS['pipeline'])
    print("✅ K-Means Model Trained & Saved.")

    # --- 4. PREDICT ON FULL DATASET ---
    print("\n[Step 4] Applying Segmentation to FULL Dataset...")
    df['segment_id'] = pipeline.predict(df)
    df['topic_name'] = df['topic_id'].map(TOPIC_MAP).fillna("Other Topic")
    
    # --- 5. GENERATE DASHBOARD CACHE ---
    print("\n[Step 5] Generating Dashboard Cache (Advanced Metrics)...")
    topic_metrics = calculate_metrics(df.copy())
    
    # Create the report structure now so Flask doesn't have to calculate it
    dashboard_report = []
    for seg_id in sorted(df['segment_id'].unique()):
        seg_data = df[df['segment_id'] == seg_id]
        if len(seg_data) == 0: continue
        
        top_topic = seg_data['topic_id'].mode()[0]
        t_stats = topic_metrics.get(str(top_topic), {})
        
        # Build Profile
        profile = {
            "id": int(seg_id),
            "title": f"{seg_data['gender'].mode()[0]}s in {seg_data['region_clean'].mode()[0]}",
            "subtitle": f"Avg Age: {round(seg_data['age'].mean(), 1)}",
            "buying_focus": TOPIC_MAP.get(top_topic, "Unknown"),
            "stats": {
                "member_count": int(len(seg_data)),
                "emergence_score": round(7.5 + min(max(t_stats.get('growth_pct',0)/10, 0), 2.5), 1),
                "status": "Critical" if t_stats.get('growth_pct',0) > 10 else "High Growth",
                "sentiment_score": t_stats.get('sentiment', 0),
                "sentiment_label": "Positive" if t_stats.get('sentiment',0)>0.05 else "Negative" if t_stats.get('sentiment',0)<-0.05 else "Neutral",
                "predicted_growth": f"{t_stats.get('growth_pct', 0)}% next month"
            }
        }
        dashboard_report.append(profile)

    with open(PATHS['dashboard_cache'], 'w') as f:
        json.dump(dashboard_report, f)
    print("✅ Dashboard Cache Saved (JSON).")

    # --- 6. SYNC TO MONGODB ---
    print(f"\n[Step 6] Syncing {len(df)} records to MongoDB (via Spring Boot)...")
    
    df = df.fillna("")
    records = df.to_dict(orient='records')
    BATCH = 500
    success_count = 0
    
    for i in range(0, len(records), BATCH):
        batch = records[i:i+BATCH]
        payload = []
        for row in batch:
            # Format Data perfectly for Java
            try:
                ts = pd.to_datetime(row.get('timestamp'))
                ts_str = ts.strftime('%Y-%m-%dT%H:%M:%S') if not pd.isna(ts) else datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            except: ts_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

            gender_val = str(row.get('gender', '')).upper().strip()
            if gender_val not in ['MALE', 'FEMALE']: gender_val = 'OTHER'

            payload.append({
                "txt_content": str(row.get('text_content', ''))[:1000],
                "timestamp": ts_str,
                "category": str(row.get('category', 'General')),
                "age": int(float(row.get('age', 0))),
                "gender": gender_val,
                "region_clean": str(row.get('region_clean', 'Other')),
                "season": str(row.get('season', 'General')),
                "clean_text": str(row.get('clean_text', ''))[:1000],
                "topic_id": int(row.get('topic_id', 0)),
                "segment_id": int(row.get('segment_id', 0)),
                "topic_name": str(row.get('topic_name', 'Unknown'))
            })
        
        try:
            resp = requests.post(SPRING_BOOT_URL, json=payload)
            if resp.status_code in [200, 201]:
                print(f"   Batch {i//BATCH + 1}: ✅")
                success_count += len(batch)
            else:
                print(f"   Batch {i//BATCH + 1}: ❌ Failed ({resp.status_code})")
        except:
            print(f"   Batch {i//BATCH + 1}: ❌ Connection Error")

    print(f"✅ Sync Complete. {success_count} records saved.")

    # --- 7. AUTO-START FLASK ---
    print("\n" + "="*50)
    print("🚀 TRAINING COMPLETE. STARTING FLASK SERVER & WORKER...")
    print("="*50)
    
    api_script = os.path.join(BASE_DIR, "app.py")
    worker_script = os.path.join(BASE_DIR, "automation_service.py")
    
    try:
        # Start App
        subprocess.Popen([sys.executable, api_script])
        print("✅ Flask API Started (Port 5001)")
        time.sleep(2)
        
        # Start Worker
        subprocess.Popen([sys.executable, worker_script])
        print("✅ Automation Worker Started")
        
        print("\nSystem Running. Press Ctrl+C to stop this master process (services will keep running).")
        # Keep script alive to hold the terminal open
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nStopping Master Script...")

if __name__ == "__main__":
    run_full_pipeline()