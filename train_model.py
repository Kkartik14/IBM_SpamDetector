import pandas as pd
import numpy as np
import pickle
import os
import urllib.request
import zipfile
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def download_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    filepath = "data/smsspamcollection.zip"
    
    if not os.path.exists(filepath):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, filepath)
    
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall("data")
    
    return "data/SMSSpamCollection"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_spam_model():
    print("Loading data...")
    dataset_path = download_dataset()
    
    data = []
    with open(dataset_path, 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                label, message = parts
                data.append([label, message])
    
    df = pd.DataFrame(data, columns=['label', 'message'])
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    df['message'] = df['message'].apply(preprocess_text)
    
    print("Preparing features...")
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    with open("models/spam_model.pkl", "wb") as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'model': model
        }, f)
    
    print("Model saved to models/spam_model.pkl")

if __name__ == "__main__":
    train_spam_model() 