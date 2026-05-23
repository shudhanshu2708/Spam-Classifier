import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("spam.csv" , encoding="latin-1")



df=df[["v1" , "v2"]]

df.columns =[ "label" ,"message"]

#print(df.head())


def check_spam(message):
    if "free" in message or "FREE" in message or "claim" in message or "CLAIM" in message:
        return "Spam!"
    else:
        return "ham!"
    
#for message in df["message"]:
    #print(message,check_spam(message))

df["prediction"] = df["message"].apply(check_spam)
print(df["prediction"].value_counts())



#df["label"].value_counts().plot(kind='bar')
#plt.title("Spam classifier")
#plt.xlabel("Label")
#plt.ylabel("Message Count")
#plt.tight_layout()

#plt.show()














