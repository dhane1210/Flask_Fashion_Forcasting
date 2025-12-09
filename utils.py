import re
import joblib
import pandas as pd
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import PATHS

# Setup NLTK (Runs once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
fashion_noise = {
    'wear', 'wearing', 'look', 'looking', 'style', 'stylish', 'fashion', 
    'outfit', 'clothes', 'clothing', 'brand', 'new', 'collection', 'trend', 
    'trendy', 'love', 'like', 'great', 'good', 'best', 'today', 'day', 
    'got', 'get', 'buy', 'buying', 'price', 'cost', 'shipping', 'available',
    'online', 'store', 'shop', 'color', 'size', 'fit', 'quality', 'material'
}
stop_words = set(stopwords.words('english')).union(fashion_noise)

def clean_text(text):
    """Standard text cleaning for all models."""
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(clean_tokens)

def load_ai_assets():
    """Loads models and dashboard cache."""
    assets = {}
    print(f"--- Loading AI Assets ---")
    
    try:
        # Load Models
        if os.path.exists(PATHS['pipeline']):
            assets['vectorizer'] = joblib.load(PATHS['vectorizer'])
            assets['nmf'] = joblib.load(PATHS['nmf'])
            assets['pipeline'] = joblib.load(PATHS['pipeline'])
            print("✅ Models Loaded")
        else:
            print("⚠️ Models not found. Pipeline will fail if predicting.")

        # Load Dashboard Cache (JSON) instead of CSV
        if os.path.exists(PATHS['dashboard_cache']):
            with open(PATHS['dashboard_cache'], 'r') as f:
                assets['report_data'] = json.load(f)
            print("✅ Dashboard Cache Loaded")
        else:
            print("⚠️ Dashboard cache missing. Run master_pipeline.py first.")
            assets['report_data'] = []
            
    except Exception as e:
        print(f"❌ Error loading assets: {e}")
    
    return assets