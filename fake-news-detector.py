"""
Fake News Detection using NLP and ML
Created by Sujal Sariya
Part of the RISE Internship - Tamizhan Skills
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("🔁 Loading dataset...")

# Load dataset
df = pd.read_csv('FakeNewsNet.csv')

print("✅ Sample data:")
print(df.head())

# Features and labels
# Some datasets use a `text` column, others use `title` or `content`.
if 'text' in df.columns:
	X = df['text']
elif 'title' in df.columns:
	X = df['title']
elif 'content' in df.columns:
	X = df['content']
else:
	raise KeyError(f"No text column found. Available columns: {list(df.columns)}")

# label column may be named `label` or `real` in some datasets
if 'label' in df.columns:
	y = df['label']
elif 'real' in df.columns:
	y = df['real']
else:
	raise KeyError(f"No label column found. Available columns: {list(df.columns)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Passive Aggressive Classifier
print("🧠 Training model...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📈 Classification Report:\n", classification_report(y_test, y_pred))
