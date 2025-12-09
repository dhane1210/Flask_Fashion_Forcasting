import os
from os.path import abspath, dirname, join

# --- DIRECTORY CONFIGURATION ---
BASE_DIR = dirname(abspath(__file__))
DATA_DIR = join(BASE_DIR, "data")
MODEL_DIR = join(BASE_DIR, "models")

# --- FILE PATHS ---
PATHS = {
    # 1. RAW INPUT (The only CSV we strictly need)
    "raw_csv": join(DATA_DIR, "FINAL_segmentation_15k_rows.csv"),
    
    # 2. OUTPUT CACHE (Small JSON for Dashboard API speed)
    "dashboard_cache": join(DATA_DIR, "dashboard_stats_cache.json"),
    
    # 3. MODELS
    "vectorizer": join(MODEL_DIR, "tfidf_vectorizer.joblib"),
    "nmf": join(MODEL_DIR, "nmf_model.joblib"),
    "pipeline": join(MODEL_DIR, "kmeans_segmentation_model.joblib")
}

# --- EXTERNAL SERVICES ---
SPRING_BOOT_URL = "http://localhost:8080/segments/"

# --- CONSTANTS ---
TOPIC_MAP = {
    0: "T-Shirts/Cotton",
    1: "Shoes/Footwear",
    3: "Watches",
    5: "Activewear",
    8: "Eyewear",
    9: "Bags"
}

# Topics to use for clustering training (Excluding noise)
VALID_TOPICS = [0, 1, 3, 5, 8, 9]