<img width="24" height="24" alt="image" src="https://github.com/user-attachments/assets/eae1d8f3-8d98-43b4-b4a6-90b9aaf61cb1" /># 📩 Spam Classifier

A simple SMS spam detection project built while learning Machine Learning.

## What it does
Takes an SMS message and predicts whether it's **spam** or **ham** (normal message).

## Dataset
- SMS Spam Collection — 5,572 messages
- 4,825 Ham | 747 Spam

## How it works
1. Loaded and cleaned the CSV data using Pandas
2. Converted text to numbers using TF-IDF Vectorizer
3. Trained a Naive Bayes model on 80% of the data
4. Tested on remaining 20% — got **96.2% accuracy**

## Results
| | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 965 ✅ | 0 ✅ |
| **Actual Spam** | 30 ❌ | 120 ✅ |

## Tech Used
- ⚡ FastAPI
- 🐍 Python
- 🐼 Pandas
- 🔢 NumPy
- 🤖 Scikit-learn — Naive Bayes, TF-IDF

## What I learned
- How to clean and prepare text data
- What TF-IDF does and why it's used
- How Naive Bayes works for text classification
- How to save a trained model using joblib
