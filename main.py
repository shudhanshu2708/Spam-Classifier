
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib

df = pd.read_csv("spam.csv" , encoding="latin-1")

df=df[["v1" , "v2"]]

df.columns =[ "label" ,"message"]

X = df["message"]
y = df["label"]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state =42)

tfidf = TfidfVectorizer(stop_words="english" , max_features=5000)
df["message"] = df["message"].str.lower()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf , y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy: ", accuracy_score(y_test , y_pred))

print(confusion_matrix(y_test , y_pred))
joblib.dump(model ," model.pkl")
joblib.dump(tfidf, "tfidf.pkl")















