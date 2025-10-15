# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an Amazon Alexa Reviews Sentiment Analysis project that provides both web and API interfaces for sentiment prediction. The system uses machine learning models trained on Amazon Alexa review data to classify text sentiment as positive or negative.

## Architecture

### Core Components

**Flask API Backend (`api.py`)**
- Serves the main prediction endpoint at `/predict`
- Handles both single text predictions and bulk CSV file processing
- Uses pre-trained XGBoost model with CountVectorizer and MinMaxScaler
- Returns sentiment predictions with visualization graphs for bulk data

**Streamlit Frontend (`main.py`)**
- Provides alternative UI for the sentiment prediction service
- Connects to Flask API running on localhost:5000
- Supports both individual text input and CSV file upload

**Web Interface (`templates/`)**
- `landing.html`: Primary web interface with modern styling
- `index.html`: Simple alternative interface
- Both connect to Flask API via JavaScript fetch calls

**Pre-trained Models (`Models/`)**
- `model_xgb.pkl`: XGBoost classifier (primary model)
- `model_rf.pkl`, `model_dt.pkl`: Alternative Random Forest and Decision Tree models
- `countVectorizer.pkl`: Text vectorization preprocessing
- `scaler.pkl`: Feature scaling transformation

### Data Flow

1. Text input → NLP preprocessing (stemming, stopword removal, regex cleaning)
2. CountVectorizer transformation → MinMaxScaler normalization
3. XGBoost prediction → Probability conversion to binary sentiment
4. For bulk processing: CSV generation + matplotlib visualization

## Development Commands

### Environment Setup
```powershell
conda create -n amazonreview python=3.10
conda activate amazonreview
pip install -r requirements.txt
```

### Running the Application
```powershell
# Start Flask API (required for both interfaces)
flask --app api.py run

# Alternative: Run with debugging
python api.py

# Start Streamlit interface (in separate terminal)
streamlit run main.py
```

### Testing the API
```powershell
# Test API health
curl http://localhost:5000/test

# Test single prediction
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"I love this product!\"}"
```

## Key Implementation Details

### Text Preprocessing Pipeline
- Regex cleaning: `[^a-zA-Z]` pattern removes non-alphabetic characters
- NLTK stopwords removal using English stopwords corpus
- Porter stemming for word normalization
- CountVectorizer converts to sparse matrix representation

### Model Selection
The system defaults to XGBoost (`model_xgb.pkl`) but contains trained alternatives. The notebook `Data Exploration & Modelling.ipynb` shows the complete training pipeline including:
- Data exploration of 3,150 Amazon Alexa reviews
- Feature engineering with text preprocessing
- Model comparison between Decision Tree, Random Forest, and XGBoost
- Cross-validation and hyperparameter tuning

### API Response Formats
- Single prediction: `{"prediction": "Positive"|"Negative"}`
- Bulk prediction: CSV download with added "Predicted sentiment" column
- Bulk processing includes base64-encoded pie chart visualization in headers

## File Structure Context

- `Data/`: Contains original dataset (`amazon_alexa.tsv`) and sample prediction outputs
- Data expects CSV format with "Sentence" column for bulk predictions
- Models directory contains all serialized scikit-learn/XGBoost objects
- Templates use localhost:5000 hardcoded endpoints for API calls

## Development Notes

- Flask API runs on port 5000 by default
- NLTK stopwords corpus must be downloaded (handled in notebook)
- All models expect the same preprocessing pipeline (CountVectorizer + MinMaxScaler)
- Bulk processing generates matplotlib graphs in BytesIO format
- JavaScript frontends handle file downloads via Blob objects