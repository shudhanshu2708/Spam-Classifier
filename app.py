from fastapi import FastAPI , UploadFile , File 
from pydantic import BaseModel
import joblib
import io
import csv
import json

app = FastAPI()


#models
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

#input 
class Message(BaseModel):
    text: str

@app.get("/")
def home():
    return {"Message": "Spam Classifier API"}

@app.post("/predict")
def predict(msg: Message):
    clean_text = msg.text.lower()
    transformed = tfidf.transform([clean_text])
    result = model.predict(transformed)
    label = str(result[0])
    return {"message": msg.text , "prediction": label}

@app.post("/upload-predict")
async def upload_predict(file: UploadFile = File(...)):
    contents = await file.read()
    decoded = contents.decode("utf-8",errors="ignore")

    messages = []

    if file.filename.endswith(".csv"):
        reader = csv.reader(io.StringIO(decoded))

        for row in reader:
            if row and row[-1].strip().lower() not in ("message", "text", "sms" ):
                messages.append(row[-1].strip())
    else:
        messages = [line.strip() for line in decoded.splitlines() if line.strip()]

    results = []
    for msg in messages:
        clean_text = msg.lower()
        transformed = tfidf.transform([clean_text])
        pred = model.predict(transformed)
        label =str(pred[0])
        results.append({"messages" : msg , "prediction" : label})

    spam_count = sum(1 for r in results if r["prediction"] == "spam")

    return {
        "total_messages": len(results),
        "spam_count": spam_count,
        "ham_count": len(results) - spam_count,
        "results" : results
    }     

@app.get("/accuracy")
def get_accuracy():
    with open("metrics.json") as f:
        metrics = json.load(f)
    return metrics     