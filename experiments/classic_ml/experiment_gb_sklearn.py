
## Experiment: Classical models vs. Neural Networks on GB dataset
# 
# This file contains the following models:
# - Linear SVC word, chars
# - Logistic regression words, chars
# - SGD classifier words, chars

# %%
# Initial setup
import sys
from pathlib import Path
from dotenv import dotenv_values

# Make packages in projects directory available for importing
env = dotenv_values(".env")
PROJECT_DIR = Path(env["PROJECT_DIR"])
sys.path.append(str(PROJECT_DIR))

# Import
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

import pandas as pd

from lib.data_processing import load_data, split_data
from lib.evaluation import evaluate
from lib.logger import CustomLogger
from pathlib import Path


RESULTS_DIR = PROJECT_DIR / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set up logger
logger = CustomLogger("experiment_gb_sklearn", log_to_local=False)

## Load data
logger.info(f"Load file: power-gb-train.tsv")
data = load_data(file_path_list=[PROJECT_DIR / "data/train/power/power-gb-train.tsv"],text_head="text_en")
train_raw, test_raw = split_data(data, test_size=0.2, random_state=0)
logger.info(f"Data size: {len(data)}, % positive class: {sum(data.labels) / len(data) * 100:.2f}%")

## Prepare feature encoders

logger.info("Prepare words_encoder")
words_encoder = TfidfVectorizer(max_features=50000)
words_encoder.fit(train_raw.texts)

logger.info("Prepare chars_encoder")
chars_encoder = TfidfVectorizer(max_features=50000, analyzer="char", ngram_range=(3,5), use_idf=True, sublinear_tf=True)
chars_encoder.fit(train_raw.texts)


## Classifiers from sklearn
# Models tested: 
# - LinearSVC
# - LogisticRegression
# - SGDClassifier

# %%
# Prepare train & test set
X_train_skl_words = words_encoder.transform(train_raw.texts)
X_test_skl_words = words_encoder.transform(test_raw.texts)

X_train_skl_chars = chars_encoder.transform(train_raw.texts)
X_test_skl_chars = chars_encoder.transform(test_raw.texts)

## Linear SVC
#### Word feature

#%%
logger.info("Fit LinearSVC model, word feature")
base_svc = LinearSVC()
model_LinearSVC_words = CalibratedClassifierCV(estimator=base_svc, cv=5)
model_LinearSVC_words.fit(X_train_skl_words, train_raw.labels)

pred_LinearSVC_words = model_LinearSVC_words.predict(X_test_skl_words)
logits_linearSVC_words = model_LinearSVC_words.predict_proba(X_test_skl_words)

result_linearSVC_words = evaluate(test_raw.labels, pred_LinearSVC_words, logits_linearSVC_words[:, 1])

#### Char features

# %%
logger.info("Fit LinearSVC model, char feature")
base_svc = LinearSVC()
model_LinearSVC_chars = CalibratedClassifierCV(estimator=base_svc, cv=5)
model_LinearSVC_chars.fit(X_train_skl_chars, train_raw.labels)

pred_LinearSVC_chars = model_LinearSVC_chars.predict(X_test_skl_chars)
logits_linearSVC_chars = model_LinearSVC_chars.predict_proba(X_test_skl_chars)

result_linearSVC_chars = evaluate(test_raw.labels, pred_LinearSVC_chars, logits_linearSVC_chars[:, 1])

# %%
## Logistic Regression
# Word features

logger.info("Fit LogisticRegression model, word feature")
model_logreg_words = LogisticRegression()
model_logreg_words.fit(X_train_skl_words, train_raw.labels)

pred_logreg_words = model_logreg_words.predict(X_test_skl_words)
logits_logreg_words = model_logreg_words.predict_proba(X_test_skl_words)

result_linearlogreg_words = evaluate(test_raw.labels, pred_logreg_words, logits_logreg_words[:, 1])

# %%
#### Char features

logger.info("Fit Logistic Regression model, char feature")
model_logreg_chars = LogisticRegression()
model_logreg_chars.fit(X_train_skl_chars, train_raw.labels)

pred_logreg_chars = model_logreg_chars.predict(X_test_skl_chars)
logits_logreg_chars = model_logreg_chars.predict_proba(X_test_skl_chars)

result_linearlogreg_chars = evaluate(test_raw.labels, pred_logreg_chars, logits_logreg_chars[:, 1])

## SGDClassifier
# SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.
# SGD is sensitive to feature scaling.

# %%
# word feature

logger.info("Fit SGDClassifier model, word feature")
model_sgd_words = SGDClassifier(loss="log_loss")
model_sgd_words.fit(X_train_skl_words, train_raw.labels)

pred_sgd_words = model_sgd_words.predict(X_test_skl_words)
logits_sgd_words = model_sgd_words.predict_proba(X_test_skl_words)

result_linearsgd_words = evaluate(test_raw.labels, pred_sgd_words, logits_sgd_words[:, 1])

#### Char features

# %%
# chars features

logger.info("Fit SGDClassifier model, char feature")
model_sgd_chars = SGDClassifier(loss="log_loss")
model_sgd_chars.fit(X_train_skl_chars, train_raw.labels)

pred_sgd_chars = model_sgd_chars.predict(X_test_skl_chars)
logits_sgd_chars = model_sgd_chars.predict_proba(X_test_skl_chars)

result_linearsgd_chars = evaluate(test_raw.labels, pred_sgd_chars, logits_sgd_chars[:, 1])


#%%
# Final results
logger.info("Write result")

results_dict = {
    "LinearSVC_words": result_linearSVC_words,
    "LinearSVC_chars": result_linearSVC_chars,
    "LogisticRegression_words": result_linearlogreg_words,
    "LogisticRegression_chars": result_linearlogreg_chars,
    "SGDClassifier_words": result_linearsgd_words,
    "SGDClassifier_chars": result_linearsgd_chars
}

results_df = pd.DataFrame.from_dict(results_dict, orient="index") \
    .reset_index().rename(columns={"index": "model"})

# Write result to JSON, CSV, LaTeX
with open(RESULTS_DIR / "results_sklearn.json", "w") as f:
    json.dump(results_dict, f)

results_df.to_csv(RESULTS_DIR / "results_sklearn.csv", index=False)

results_df.to_latex(RESULTS_DIR / "results_sklearn.tex", index=False)