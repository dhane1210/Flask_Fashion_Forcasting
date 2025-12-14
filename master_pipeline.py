import pandas as pd
import numpy as np
import joblib
import requests
import json
import warnings
import os
import sys
import subprocess
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from config import PATHS, TOPIC_MAP, CATEGORY_TO_ID, SPRING_SYNC_URL, SPRING_FETCH_URL, VALID_TOPICS, BASE_DIR
from utils import clean_text
from generate_metrics import generate_advanced_metrics, calculate_simple_metrics

warnings.filterwarnings('ignore')

def smart_assign_topic(row):

    cat = str(row.get('category', '')).title()
    if cat in CATEGORY_TO_ID: return CATEGORY_TO_ID[cat]
    
    text = str(row.get('text_content', '')).lower()
    if any(x in text for x in ['shoe', 'sneaker', 'boot', 'heel']): return 1
    if any(x in text for x in ['watch', 'dial', 'strap']): return 2
    if any(x in text for x in ['bag', 'purse', 'wallet', 'backpack']): return 3
    if any(x in text for x in ['gym', 'run', 'yoga', 'fitness']): return 4
    if any(x in text for x in ['glass', 'lens', 'frame']): return 5
    if any(x in text for x in ['suit', 'formal', 'blazer']): return 6
    if any(x in text for x in ['dress', 'gown']): return 7
    if any(x in text for x in ['shirt', 'tee', 'top']): return 0
    
    return np.random.choice(VALID_TOPICS)

def balance_dataset(df):
    """
    STRICT BALANCING: Caps the max rows per topic to 600.
    This ensures dominant categories (T-Shirts) don't drown out niche ones (Watches).
    """
    print("Balancing Dataset (Strict Mode)")
    df_balanced = pd.DataFrame()
    
    TARGET_CAP = 600 
    
    for topic in VALID_TOPICS:
        subset = df[df['topic_id'] == topic]
        if len(subset) > TARGET_CAP:
            # Downsample if too big (Randomly pick 600)
            subset = subset.sample(n=TARGET_CAP, random_state=42)
        
        df_balanced = pd.concat([df_balanced, subset])
    
    print(f"   Reduced data from {len(df)} to {len(df_balanced)} to ensure maximum variety.")
    return df_balanced

def run_pipeline():
    print("\n" + "="*50)
    print("STARTING BALANCED PIPELINE (8 SEGMENTS)")
    print("="*50)

    # --- 1. LOAD RAW DATA ---
    print(f"\n[Step 1] Loading Raw Data...")
    if not os.path.exists(PATHS['raw_csv']):
        print(f"Error: Raw CSV missing."); return

    df_raw = pd.read_csv(PATHS['raw_csv'])
    print(f"Loaded {len(df_raw)} rows.")

    # --- 2. TRAIN NLP (Dummy for API) ---
    print("\n[Step 2] Preparing NLP Models...")
    df_raw['clean_text'] = df_raw['text_content'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf = vectorizer.fit_transform(df_raw['clean_text'])
    nmf = NMF(n_components=10, init='nndsvda', random_state=42)
    nmf.fit(tfidf)
    joblib.dump(vectorizer, PATHS['vectorizer'])
    joblib.dump(nmf, PATHS['nmf'])
    
    # --- 3. ASSIGN TOPICS & BALANCE ---
    print("\n[Step 3] Assigning & Balancing Topics...")
    df_raw['topic_id'] = df_raw.apply(smart_assign_topic, axis=1)
    
    # APPLY STRICT BALANCING HERE
    df_raw = balance_dataset(df_raw)
    
    df_raw['topic_name'] = df_raw['topic_id'].map(TOPIC_MAP)
    print("   Topic Distribution (After Balancing):")
    print(df_raw['topic_name'].value_counts().head(8))

    # --- 4. SYNC TO MONGODB ---
    print(f"\n[Step 4] Syncing {len(df_raw)} records to MongoDB...")
    df_raw = df_raw.fillna("")
    records = df_raw.to_dict(orient='records')
    BATCH = 500
    
    for i in range(0, len(records), BATCH):
        batch = records[i:i+BATCH]
        payload = []
        for row in batch:
            try: ts = pd.to_datetime(row.get('timestamp')).strftime('%Y-%m-%dT%H:%M:%S')
            except: ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            
            gen = str(row.get('gender', '')).upper()
            if gen not in ['MALE', 'FEMALE']: gen = 'OTHER'

            payload.append({
                "txt_content": str(row.get('text_content', ''))[:500],
                "timestamp": ts,
                "category": str(row.get('category', 'General')),
                "age": int(float(row.get('age', 0))),
                "gender": gen,
                "region_clean": str(row.get('region_clean', 'Other')),
                "season": str(row.get('season', 'General')),
                "clean_text": str(row.get('clean_text', ''))[:500],
                "topic_id": int(row.get('topic_id', 0)),
                "segment_id": 0,
                "topic_name": str(row.get('topic_name', 'Unknown'))
            })
        try:
            requests.post(SPRING_SYNC_URL, json=payload)
            print(f"   Batch {i//BATCH + 1}: Sent")
        except:
            print(f"   Batch {i//BATCH + 1}: Failed")

    print("Data Sync Complete.")

    # --- 5. FETCH FROM MONGODB ---
    print(f"\n[Step 5] Fetching Live Data from MongoDB...")
    try:
        resp = requests.get(SPRING_FETCH_URL)
        if resp.status_code == 200:
            db_data = resp.json()
            df_db = pd.DataFrame(db_data)
            print(f"Fetched {len(df_db)} rows.")
        else:
            print("Failed fetch."); return
    except: return

    # --- 6. TRAIN SEGMENTATION (K-Means) ---
    print("\n[Step 6] Training Segmentation Model...")
    
    df_train = df_db[df_db['topic_id'].isin(VALID_TOPICS)].copy()
    print(f"   Training on {len(df_train)} balanced rows.")

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'region_clean', 'topic_id'])
    ])
    
    # INCREASED CLUSTERS TO 8 to force more variety
    N_CLUSTERS = 8 
    kmeans_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10))
    ])
    
    kmeans_pipeline.fit(df_train)
    df_db['segment_id'] = kmeans_pipeline.predict(df_db)
    
    joblib.dump(kmeans_pipeline, PATHS['pipeline'])
    print(f"K-Means Model Trained ({N_CLUSTERS} Segments).")

    # --- 7. GENERATE METRICS ---
    print("\n[Step 7] Generating Dashboard Cache...")
    generate_advanced_metrics() 
    
    dashboard_report = calculate_simple_metrics(df_db, TOPIC_MAP)
    with open(PATHS['dashboard_cache'], 'w') as f:
        json.dump(dashboard_report, f)
    print("Dashboard Cache Updated.")

    # --- 8. AUTO-START ---
    print("\n" + "="*50)
    print("PIPELINE FINISHED. LAUNCHING SERVERS...")
    print("="*50)
    
    subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "start_services.py")])

if __name__ == "__main__":
    run_pipeline()