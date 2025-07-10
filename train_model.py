import pandas as pd
import numpy as np
import pickle
import os
import urllib.request
import tarfile
import re
import email
from email.parser import Parser
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def download_email_dataset():
    url = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"
    spam_url = "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2"
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    ham_path = "data/20030228_easy_ham.tar.bz2"
    spam_path = "data/20030228_spam.tar.bz2"
    
    if not os.path.exists(ham_path):
        print("Downloading ham emails...")
        urllib.request.urlretrieve(url, ham_path)
    
    if not os.path.exists(spam_path):
        print("Downloading spam emails...")
        urllib.request.urlretrieve(spam_url, spam_path)
    
    if not os.path.exists("data/easy_ham"):
        print("Extracting ham emails...")
        with tarfile.open(ham_path, 'r:bz2') as tar:
            tar.extractall("data")
    
    if not os.path.exists("data/spam"):
        print("Extracting spam emails...")
        with tarfile.open(spam_path, 'r:bz2') as tar:
            tar.extractall("data")
    
    return "data/easy_ham", "data/spam"

def extract_email_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            email_content = f.read()
        
        msg = email.message_from_string(email_content)
        
        subject = msg.get('Subject', '')
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        return subject + " " + body
    except:
        return ""

def preprocess_email_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_email_data():
    ham_dir, spam_dir = download_email_dataset()
    
    data = []
    
    print("Processing ham emails...")
    for filename in os.listdir(ham_dir):
        if filename != 'cmds':
            file_path = os.path.join(ham_dir, filename)
            content = extract_email_content(file_path)
            if content:
                data.append([0, content])
    
    print("Processing spam emails...")
    for filename in os.listdir(spam_dir):
        if filename != 'cmds':
            file_path = os.path.join(spam_dir, filename)
            content = extract_email_content(file_path)
            if content:
                data.append([1, content])
    
    return data

def train_email_spam_model():
    print("Loading email data...")
    data = load_email_data()
    
    df = pd.DataFrame(data, columns=['label', 'content'])
    df['content'] = df['content'].apply(preprocess_email_text)
    df = df[df['content'].str.len() > 10]
    
    print(f"Dataset size: {len(df)} emails")
    print(f"Spam emails: {len(df[df['label'] == 1])}")
    print(f"Ham emails: {len(df[df['label'] == 0])}")
    
    print("Preparing features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(df['content'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training email spam model...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    with open("models/email_spam_model.pkl", "wb") as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'model': model
        }, f)
    
    print("Email spam model saved to models/email_spam_model.pkl")

if __name__ == "__main__":
    train_email_spam_model() 