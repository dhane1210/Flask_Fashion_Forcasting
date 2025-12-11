import re
import joblib
import pandas as pd
import json
import os
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import PATHS, FASHION_NOISE

# --- FIX: BYPASS SSL CHECKS (Linux Support) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- NLTK SETUP ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# Merge Standard English Stopwords with our Aggressive Fashion Noise
stop_words = set(stopwords.words('english')).union(FASHION_NOISE)

def clean_text(text):
    """Standard text cleaning for all models."""
    if pd.isna(text): return ""
    text = str(text).lower()
    
    # Remove special chars but keep spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    tokens = text.split()
    # Lemmatize and remove the aggressive stopwords
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    
    return " ".join(clean_tokens)

def load_ai_assets():
    """Loads models and dashboard cache."""
    assets = {}
    print(f"--- Loading AI Assets ---")
    
    try:
        # 1. Load Models
        if os.path.exists(PATHS['pipeline']):
            assets['vectorizer'] = joblib.load(PATHS['vectorizer'])
            assets['nmf'] = joblib.load(PATHS['nmf'])
            assets['pipeline'] = joblib.load(PATHS['pipeline'])
            print("✅ Models Loaded")
        else:
            print("⚠️ Models not found. Pipeline will fail if predicting.")

        # 2. Load Dashboard Cache (Safely)
        cache_path = PATHS.get('dashboard_cache')
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                assets['report_data'] = json.load(f)
            print("✅ Dashboard Cache Loaded")
        else:
            print(f"⚠️ Dashboard cache not found. Run master_pipeline.py first.")
            assets['report_data'] = []
            
    except Exception as e:
        print(f"❌ Error loading assets: {e}")
    
    return assets