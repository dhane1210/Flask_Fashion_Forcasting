🧠 AI Trend Forecasting Microservice (Flask)

This repository contains the Machine Learning Intelligence Engine for the Fashion Trend Forecasting System.

Unlike standard backends that just retrieve data, this microservice actively trains AI models, discovers hidden market segments, and predicts future trends using raw social media and product data.

🚀 Key Capabilities

This system implements a sophisticated Two-Stage ML Pipeline:

NLP Topic Discovery (Model 1):

Algorithm: Non-Negative Matrix Factorization (NMF) with TF-IDF.

Input: 15,000+ raw text descriptions from social media/reviews.

Output: Automatically discovers hidden fashion themes (e.g., "Activewear", "Vintage Leather", "Smart Watches") without manual labeling.

Customer Segmentation (Model 2):

Algorithm: K-Means Clustering.

Input: User Demographics (Age, Gender, Region) + Interest Topic (from Model 1).

Output: Identifies 5 distinct "High Value" customer personas (e.g., "Young Men in Europe interested in Sneakers").

Advanced Predictive Metrics:

Sentiment Analysis: Uses VADER to determine if the trend is Positive (😍) or Negative (😡).

Growth Forecasting: Uses Exponential Smoothing (Time Series) to predict trend volume for the next month (e.g., "+15% Growth").

Automated Database Sync:

The system pushes processed insights directly to the Spring Boot Backend, which stores them in MongoDB. This ensures the main application always serves cached, high-performance data.

📂 Project Structure (Modular Architecture)

We follow a production-grade modular structure to ensure scalability and clean code.

flask_server/
│
├── config.py                # Single Source of Truth (Paths, URLs, Topic Maps)
├── utils.py                 # Shared utilities (Text Cleaning, Asset Loading)
│
├── app.py                   # 🌐 The API (Serves JSON to Spring Boot)
├── automation_service.py    # 🤖 The Worker (Simulates live data & Syncs to DB)
├── master_pipeline.py       # 🧠 The Trainer (Retrains models & Populates DB)
├── start_services.py        # 🚀 The Launcher (Runs API + Worker together)
│
├── data/                    # Contains raw CSVs and generated JSON cache
└── models/                  # Stores trained .joblib models (Vectorizers, Pipelines)


⚡ Quick Start Guide

Prerequisites

Python 3.8+

Spring Boot Backend running on Port 8080 (for Database Sync)

1. Installation

Install dependencies using pip.

pip install -r requirements.txt


2. Initialization (Run Once)

Before starting the server, you must train the models and populate the database.

python master_pipeline.py


What this does:

Loads FINAL_segmentation_15k_rows.csv.

Trains NMF (NLP) & K-Means models.

Calculates Sentiment & Growth forecasts.

Syncs all 15,000+ records to MongoDB via Spring Boot.

Saves .joblib models for the API.

3. Start the System (Live Mode)

Run the master launcher to start the API and the Background Worker simultaneously.

python start_services.py


API: Running at http://localhost:5001

Worker: Running in background (Simulating live data every 30s)

🔌 API Documentation

This Flask service is designed to be consumed by the Spring Boot Backend, not the Frontend directly.

1. Get Dashboard Intelligence

Endpoint: GET /get_all_trends

Description: Returns a high-level summary of the 5 discovered market segments.

Response:

[
  {
    "id": 0,
    "title": "Females in Europe",
    "buying_focus": "Bags",
    "stats": {
      "member_count": 2915,
      "emergence_score": 8.8,
      "sentiment_label": "Positive",
      "predicted_growth": "15.2% next month"
    }
  },
  ...
]


2. Predict Segment (Real-Time Inference)

Endpoint: POST /predict

Description: Takes raw user data and predicts which segment they belong to.

Payload:

{
  "text": "I love these new running shoes",
  "age": 25,
  "gender": "Male",
  "region": "Europe"
}


🛠 Tech Stack

Framework: Flask

Machine Learning: Scikit-Learn (NMF, K-Means, TF-IDF), Statsmodels (Forecasting)

NLP: NLTK (VADER Sentiment, Lemmatization)

Data Processing: Pandas, NumPy

Serialization: Joblib (Model persistence)

🔄 The Data Pipeline Flow

Raw Data (FINAL_segmentation_15k_rows.csv) is loaded.

Preprocessing: Text is cleaned, lemmatized, and stopwords are removed.

Model 1 (NLP): Converts text to vectors (TF-IDF) and assigns a Topic ID.

Model 2 (Clustering): Combines Topic ID with Demographics to assign a Segment ID.

Metrics Engine: Calculates historical growth rates and sentiment scores.

Sync: The final enriched dataset is POSTed to the Spring Boot API for storage in MongoDB.
