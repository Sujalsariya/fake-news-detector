"""
Improved training script for Fake News Detection
Uses TF-IDF (unigrams + bigrams) + Logistic Regression with GridSearchCV
"""
import re
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_columns(df: pd.DataFrame):
    if 'text' in df.columns:
        X = df['text']
    elif 'title' in df.columns:
        X = df['title']
    elif 'content' in df.columns:
        X = df['content']
    else:
        raise KeyError(f"No text column found. Available columns: {list(df.columns)}")

    if 'label' in df.columns:
        y = df['label']
    elif 'real' in df.columns:
        y = df['real']
    else:
        raise KeyError(f"No label column found. Available columns: {list(df.columns)}")

    return X, y


def main():
    df_path = 'FakeNewsNet.csv'
    print(f"Loading {df_path}...")
    df = pd.read_csv(df_path)
    print("Columns:", list(df.columns))

    X, y = get_columns(df)
    X = X.fillna("").astype(str).map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.85, min_df=5, max_features=50000)),
        ('clf', LogisticRegression(solver='saga', max_iter=2000, class_weight='balanced', n_jobs=-1))
    ])

    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.1, 1.0, 5.0]
    }

    print("Running GridSearchCV (this may take a while)...")
    gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)

    y_pred = gs.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    model_path = 'best_model.joblib'
    joblib.dump(gs.best_estimator_, model_path)
    print(f"Saved best model to {model_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
