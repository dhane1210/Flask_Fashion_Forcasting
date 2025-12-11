import os
from os.path import abspath, dirname, join

# --- DIRECTORY CONFIGURATION ---
BASE_DIR = dirname(abspath(__file__))
DATA_DIR = join(BASE_DIR, "data")
MODEL_DIR = join(BASE_DIR, "models")

# --- FILE PATHS ---
PATHS = {
    "raw_csv": join(DATA_DIR, "FINAL_segmentation_15k_rows.csv"),
    "dashboard_cache": join(DATA_DIR, "dashboard_stats_cache.json"),
    "metrics_json": join(DATA_DIR, "topic_advanced_metrics.json"),
    "vectorizer": join(MODEL_DIR, "tfidf_vectorizer.joblib"),
    "nmf": join(MODEL_DIR, "nmf_model.joblib"),
    "pipeline": join(MODEL_DIR, "kmeans_segmentation_model.joblib")
}

# --- EXTERNAL SERVICES ---
SPRING_BOOT_URL = "http://localhost:8080/segments/"
SPRING_SYNC_URL = "http://localhost:8080/segments/"
SPRING_FETCH_URL = "http://localhost:8080/segments/all"

# --- CONSTANTS ---
# 1. Output Labels (What shows on Dashboard)
TOPIC_MAP = {
    0: "T-Shirts & Tops",
    1: "Footwear",
    2: "Watches",
    3: "Bags & Luggage",
    4: "Activewear",
    5: "Eyewear",
    6: "Formal Wear",
    7: "Dresses",
    99: "General Trends"
}

# 2. Input Map (Category Column -> Topic ID)
CATEGORY_TO_ID = {
    'Tshirts': 0, 'Shirts': 0, 'Blouses': 0, 'Tops': 0,
    'Shoes': 1, 'Sneakers': 1, 'Heels': 1, 'Sandals': 1,
    'Watches': 2,
    'Bags': 3, 'Handbags': 3, 'Wallets': 3,
    'Activewear': 4, 'Sportswear': 4, 'Leggings': 4,
    'Eyewear': 5, 'Sunglasses': 5,
    'Formal': 6, 'Suits': 6, 'Blazers': 6,
    'Dresses': 7
}

# Topics to use for clustering (Exclude 99)
VALID_TOPICS = [0, 1, 2, 3, 4, 5, 6, 7] 

# --- NOISE FILTER ---
FASHION_NOISE = {
    'wear', 'wearing', 'look', 'looking', 'style', 'stylish', 'fashion', 
    'outfit', 'clothes', 'clothing', 'brand', 'new', 'collection', 'trend', 
    'trendy', 'love', 'like', 'great', 'good', 'best', 'today', 'day', 
    'got', 'get', 'buy', 'buying', 'price', 'cost', 'shipping', 'available',
    'online', 'store', 'shop', 'color', 'size', 'fit', 'quality', 'material'
}