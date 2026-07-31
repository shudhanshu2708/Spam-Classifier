# 📩 Spam Classifier

A machine learning–powered SMS spam detection API, built with FastAPI. Takes a message (or a batch of messages via file upload) and predicts whether it's **spam** or **ham** (normal message).

## Features

- 🔍 Predict spam/ham for a single message
- 📂 Bulk prediction via CSV/TXT file upload
- 📊 Live model accuracy endpoint
- ⚡ Fast, lightweight REST API built with FastAPI

## Dataset

- SMS Spam Collection — 5,572 messages
- 4,825 Ham | 747 Spam

## How it works

1. Loaded and cleaned the CSV data using Pandas
2. Converted text to numeric features using TF-IDF Vectorizer
3. Trained a Multinomial Naive Bayes model on 80% of the data
4. Evaluated on the remaining 20% — achieved **97.3% accuracy**
5. Wrapped the trained model in a FastAPI service with prediction, batch upload, and metrics endpoints

## API Endpoints

| Method | Endpoint          | Description                                      |
|--------|-------------------|---------------------------------------------------|
| GET    | `/`               | Health check                                       |
| POST   | `/predict`        | Predict spam/ham for a single message              |
| POST   | `/upload-predict` | Upload a `.csv` or `.txt` file for bulk prediction |
| GET    | `/accuracy`       | Returns the current model's accuracy               |

### Example — `/predict`

**Request:**
```json
{
  "text": "Congratulations! You've won a free prize, claim now!"
}
```

**Response:**
```json
{
  "message": "Congratulations! You've won a free prize, claim now!",
  "prediction": "spam"
}
```

### Example — `/upload-predict`

Upload a `.txt` file (one message per line) or a `.csv` file (message in the last column). Returns per-message predictions along with total spam/ham counts.

## Results

|                 | Predicted Ham | Predicted Spam |
| --------------- | ------------- | -------------- |
| **Actual Ham**  | 965 ✅         | 0 ✅            |
| **Actual Spam** | 30 ❌          | 120 ✅          |

**Accuracy: 97.3%**

## Tech Stack

- ⚡ FastAPI — REST API
- 🐍 Python
- 🐼 Pandas — data cleaning
- 🔢 NumPy
- 🤖 Scikit-learn — TF-IDF Vectorizer, Multinomial Naive Bayes
- 💾 Joblib — model persistence

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirement.txt

# 2. Train the model (generates model.pkl, tfidf.pkl, metrics.json)
python main.py

# 3. Start the API
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation (Swagger UI).

## What I learned

- How to clean and prepare text data for ML
- What TF-IDF does and why it's used for text classification
- How Naive Bayes works for spam detection
- How to serve an ML model as a production-style REST API with FastAPI
- Handling file uploads and batch inference in FastAPI
