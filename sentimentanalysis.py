import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# 🔹 Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt') 
nltk.download('omw-1.4')

# 🔹 Load dataset
file_path = 'C:/Users/HP/Downloads/sentiment_annotated_with_texts.csv'  # Ensure this file exists in the same directory
df = pd.read_csv(file_path)

# 🔹 Display basic info about dataset
print(df.head())
print("\nDataset Shape:", df.shape)

# 🔹 Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 🔹 Rename columns if necessary (based on the dataset structure)
df.rename(columns={'true_sentiment': 'sentiment', 'text': 'headline'}, inplace=True)

# 🔹 Text Preprocessing Function
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = nltk.word_tokenize(text)  # Tokenization
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)

# 🔹 Apply preprocessing
df['cleaned_text'] = df['headline'].astype(str).apply(preprocess_text)

# 🔹 Convert sentiment labels to numerical values
sentiment_mapping = {"Positive": 2, "Neutral": 1, "Negative": 0}  # Adjusted for dataset
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# 🔹 Splitting dataset
X = df['cleaned_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🔹 Model Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 🔹 Predictions
y_pred = model.predict(X_test_tfidf)

# 🔹 Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 🔹 Visualization - Confusion Matrix
plt.figure(figsize=(5,5))
plt.matshow(confusion_matrix(y_test, y_pred), cmap=plt.cm.Blues, fignum=1)
plt.title("Confusion Matrix")
plt.colorbar()
plt.ylabel("Actual Sentiment")
plt.xlabel("Predicted Sentiment")
plt.show()

# 🔹 Streamlit App for Deployment
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment_dict = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_dict[prediction]

st.title("📰 News Headline Sentiment Analysis")
headline = st.text_input("Enter a news headline:")
if st.button("Analyze"):
    result = predict_sentiment(headline)
    st.write(f"**Predicted Sentiment:** {result}")
