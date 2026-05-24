from fastapi import FastAPI
from pydantic import BaseModel
import joblib

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
    return {"message": msg.text , "prediction": result[0]}
    