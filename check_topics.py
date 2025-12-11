import joblib
from config import PATHS

def print_topic_keywords():
    print("--- 🔍 INSPECTING CURRENT TOPIC MODEL ---")
    
    try:
        vectorizer = joblib.load(PATHS['vectorizer'])
        nmf = joblib.load(PATHS['nmf'])
        feature_names = vectorizer.get_feature_names_out()
    except:
        print("❌ Models not found. Run master_pipeline.py first.")
        return

    # Print top 10 keywords for each topic
    for topic_idx, topic in enumerate(nmf.components_):
        top_features_ind = topic.argsort()[:-11:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        print(f"Topic {topic_idx}: {', '.join(top_features)}")

if __name__ == "__main__":
    print_topic_keywords()