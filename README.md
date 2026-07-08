# Fake News Detection 📰🧠

This repository provides a lightweight pipeline for detecting fake news using classical NLP + ML techniques. It includes a baseline script and an improved training script that produces a saved model.

Key points:
- Author: Sujal Sariya
- Dataset (example): `FakeNewsNet.csv` (a small CSV is included for experimentation)
- Saved model: `best_model.joblib` (created by `train_improved.py`)

---

## Files

- `fake-news-detector.py` - baseline training/evaluation script (TF-IDF + PassiveAggressiveClassifier). Updated to handle multiple common column names.
- `train_improved.py` - improved pipeline: TF-IDF (unigrams+bigrams) + `LogisticRegression` using `GridSearchCV`. Saves best model to `best_model.joblib`.
- `FakeNewsNet.csv` - example dataset (committed here for demo). Columns in this dataset: `title`, `news_url`, `source_domain`, `tweet_num`, `real`.
- `best_model.joblib` - exported trained model from the improved pipeline.
- `requirements.txt` - Python dependencies.

---

## Requirements

Python 3.10+ and the packages listed in `requirements.txt`.

Install:

```bash
pip install -r requirements.txt
```

---

## Dataset expectations

- The scripts look for a text column (in order of preference): `text`, `title`, `content`.
- The label column can be named `label` or `real`.
- If your CSV uses different column names, either rename them or edit the script to point to the correct columns.

---

## Quick usage

1. Run the baseline script (small / fast):

```bash
python3 fake-news-detector.py
```

2. Run the improved training (GridSearch; may take longer):

```bash
python3 train_improved.py
```

This will print results and save the best estimator to `best_model.joblib`.

---

## Example results (from `train_improved.py` run)

- Accuracy: ~0.818
- Confusion matrix (example): [[880, 271], [573, 2916]]

Interpret these numbers alongside precision/recall for each class — dataset is imbalanced so accuracy alone can be misleading.

---

## Next steps / improvements

- Try class rebalancing (`class_weight='balanced'`, SMOTE) to improve minority-class recall.
- Try stronger models: XGBoost / LightGBM.
- For large accuracy gains, fine-tune a transformer (e.g., `bert-base-uncased`) using `transformers` + `accelerate`.
- Add metadata features (source domain, tweet count) to improve predictions.

---

## Notes

- The repository currently includes a small dataset file for convenience. In production, keep datasets out of the repository and load them from a data storage or mount.
- The code is intended as a starting point — treat the saved model as an experimental artifact, not production-ready.

---
