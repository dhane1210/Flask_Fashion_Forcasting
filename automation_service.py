import time
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from config import SPRING_BOOT_URL, TOPIC_MAP
from utils import load_ai_assets, clean_text

print("--- 🤖 Starting Automation Worker... ---")
assets = load_ai_assets()

if not assets.get('vectorizer'):
    print("❌ Models missing. Worker exiting.")
    exit()

def simulate_new_data():
    # In real life: Call Twitter/Reddit API here
    print("\n[Scraper] Looking for fresh trends...")
    data = [
        {"text": "Obsessed with these new running shoes", "age": 24, "gender": "MALE", "region": "Europe"},
        {"text": "Looking for a vintage leather bag", "age": 45, "gender": "FEMALE", "region": "North America"},
        {"text": "Best cotton t-shirts for summer", "age": 19, "gender": "MALE", "region": "Europe"}
    ]
    df = pd.DataFrame(data)
    df['timestamp'] = datetime.now().isoformat()
    df['category'] = "Social Media"
    df['season'] = "Winter"
    return df

def run_pipeline():
    df = simulate_new_data()
    
    # NLP
    df['clean_text'] = df['text'].apply(clean_text)
    tfidf = assets['vectorizer'].transform(df['clean_text'])
    df['topic_id'] = np.argmax(assets['nmf'].transform(tfidf), axis=1)
    df['topic_name'] = df['topic_id'].map(TOPIC_MAP).fillna("Other")

    # Segmentation
    input_df = df[['age', 'gender', 'region', 'topic_id']].copy()
    input_df['region_clean'] = input_df['region']
    df['segment_id'] = assets['pipeline'].predict(input_df)

    # Sync
    json_payload = []
    for _, row in df.iterrows():
        # Java Format Compliance
        gender = row['gender'].upper() if row['gender'].upper() in ['MALE', 'FEMALE'] else 'OTHER'
        ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        json_payload.append({
            "txt_content": row['text'],
            "timestamp": ts,
            "category": row['category'],
            "age": int(row['age']),
            "gender": gender,
            "region_clean": row['region'],
            "season": row['season'],
            "clean_text": row['clean_text'],
            "topic_id": int(row['topic_id']),
            "segment_id": int(row['segment_id']),
            "topic_name": row['topic_name']
        })

    try:
        resp = requests.post(SPRING_BOOT_URL, json=json_payload, headers={"Content-Type": "application/json"})
        if resp.status_code in [200, 201]: print(f"✅ Synced {len(json_payload)} new items.")
        else: print(f"❌ Sync Failed: {resp.status_code}")
    except:
        print("❌ Connection Error to Spring Boot")

if __name__ == "__main__":
    try:
        while True:
            run_pipeline()
            time.sleep(30)
    except KeyboardInterrupt:
        print("Stopped.")