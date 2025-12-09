import pandas as pd
import numpy as np
import joblib
import json
import nltk
import os  # <--- THIS WAS MISSING
from nltk.sentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from config import PATHS, VALID_TOPICS

def generate_advanced_metrics():
    print("--- 🚀 Generating Advanced Metrics (Sentiment & Forecasts) ---")
    
    # 1. Load Data & Models
    if not os.path.exists(PATHS['raw_csv']):
        print(f"❌ Error: Raw CSV not found at {PATHS['raw_csv']}")
        return

    df = pd.read_csv(PATHS['raw_csv'])
    # Ensure text is clean before processing
    df = df.dropna(subset=['text_content'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Load Models
    try:
        vectorizer = joblib.load(PATHS['vectorizer'])
        nmf = joblib.load(PATHS['nmf'])
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # 2. Get Topics
    print("Assigning topics...")
    # We convert text to string to avoid errors
    tfidf_matrix = vectorizer.transform(df['text_content'].astype(str))
    df['topic_id'] = np.argmax(nmf.transform(tfidf_matrix), axis=1)

    # 3. Sentiment Analysis
    print("Calculating Sentiment...")
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    
    # Use a lambda function to handle non-string text safely
    df['sentiment'] = df['text_content'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    
    # Calculate average sentiment per topic
    topic_sentiment = df.groupby('topic_id')['sentiment'].mean().to_dict()

    # 4. Forecasting
    print("Forecasting Trends...")
    df = df.dropna(subset=['timestamp'])
    
    # Group by Month
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_counts = df.groupby(['topic_id', 'month']).size().reset_index(name='count')
    
    topic_forecasts = {}
    
    for topic in VALID_TOPICS:
        history = monthly_counts[monthly_counts['topic_id'] == topic]
        growth_pct = 0
        
        # We need at least 3 data points (months) to forecast
        if len(history) > 2:
            try:
                series = history['count'].values.astype(float)
                # Exponential Smoothing is robust for simple trends
                model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
                forecast = model.forecast(1)
                
                # Avoid division by zero
                last_val = series[-1] if series[-1] != 0 else 1
                growth_pct = round(((forecast[0] - last_val) / last_val) * 100, 1)
            except: 
                pass
            
        topic_forecasts[topic] = {"growth_pct": growth_pct}

    # 5. Save Results
    final_metrics = {}
    for topic in VALID_TOPICS:
        final_metrics[int(topic)] = {
            "sentiment": round(topic_sentiment.get(topic, 0), 2),
            "forecast": topic_forecasts.get(topic, {})
        }

    with open(PATHS['metrics_json'], 'w') as f:
        json.dump(final_metrics, f)
    
    print(f"✅ Metrics saved to {PATHS['metrics_json']}")

if __name__ == "__main__":
    generate_advanced_metrics()