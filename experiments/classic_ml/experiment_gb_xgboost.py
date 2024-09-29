
## Experiment: Classical models vs. Neural Networks on GB dataset
# 
# We will compare the following models:
# - XGBoost Words, chars

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
import xgboost as xgb
import pandas as pd

from lib.data_processing import load_data, split_data
from lib.evaluation import evaluate
from lib.logger import CustomLogger


RESULTS_DIR = PROJECT_DIR / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set up logger
logger = CustomLogger("experiment_gb_xgboost", log_to_local=False)


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

# %%
# Prepare train & test set
X_train_skl_words = words_encoder.transform(train_raw.texts)
X_test_skl_words = words_encoder.transform(test_raw.texts)

X_train_skl_chars = chars_encoder.transform(train_raw.texts)
X_test_skl_chars = chars_encoder.transform(test_raw.texts)

# 

# %%

# Prepare data for xgboost
train_dmat_words = xgb.DMatrix(X_train_skl_words, pd.array(train_raw.labels).astype("category"))
test_dmat_words = xgb.DMatrix(X_test_skl_words, pd.array(test_raw.labels).astype("category"))

train_dmat_chars = xgb.DMatrix(X_train_skl_chars, pd.array(train_raw.labels).astype("category"))
test_dmat_chars = xgb.DMatrix(X_test_skl_chars, pd.array(test_raw.labels).astype("category"))


# %%
# Config for Xgboost
params = {
    "booster": "gbtree",
    "objective": "binary:logistic",  # there is also binary:hinge but hinge does not output probability
    "tree_method": "hist",  # default to hist
    "device": "cuda",

    # Params for tree booster
    "eta": 0.3,
    "gamma": 0.0,  # Min loss achieved to split the tree
    "max_depth": 6,
    "reg_alpha": 0,
    "reg_lambda": 1,

}

ITERATIONS = 2000


#### Word features
evals_words = [(train_dmat_words, "train")]

model_xgb_words = xgb.train(
    params = params,
    dtrain = train_dmat_words,
    num_boost_round = ITERATIONS,
    evals = evals_words,
    verbose_eval = 100
)

pred_xgb_words_probs = model_xgb_words.predict(test_dmat_words)
result_xgb_words = evaluate(test_raw.labels, pred_xgb_words_probs > 0.5, pred_xgb_words_probs)

#### Char features

# %%
# Can use only half of the original max features
xgb_chars_encoder = TfidfVectorizer(max_features=50000, analyzer="char", ngram_range=(3,5), use_idf=True, sublinear_tf=True)
xgb_chars_encoder.fit(train_raw.texts)

# Prepare train & test set
X_train_xgb_chars = xgb_chars_encoder.transform(train_raw.texts)
X_test_xgb_chars = xgb_chars_encoder.transform(test_raw.texts)

import xgboost as xgb

train_dmat_chars = xgb.DMatrix(X_train_xgb_chars, pd.array(train_raw.labels).astype("category"))
test_dmat_chars = xgb.DMatrix(X_test_xgb_chars, pd.array(test_raw.labels).astype("category"))

evals_chars = [(train_dmat_chars, "train")]
ITERATIONS = 2000

model_xgb_chars = xgb.train(
    params = params,
    dtrain = train_dmat_chars,
    num_boost_round = ITERATIONS,
    evals = evals_chars,
    verbose_eval = 100
)

pred_xgb_chars_probs = model_xgb_chars.predict(test_dmat_chars)
result_xgb_chars = evaluate(test_raw.labels, pred_xgb_chars_probs > 0.5, pred_xgb_chars_probs)

# Write results
logger.info("Write result")
results_dict = {
    "XGBoost_words": result_xgb_words,
    "XGBoost_chars": result_xgb_chars
}

with open(RESULTS_DIR / "experiment_gb_xgboost.json", "w") as f:
    json.dump(results_dict, f)

results_df = pd.DataFrame.from_dict(results_dict, orient="index")\
    .reset_index().rename(columns={"index": "model"})

results_df.to_csv(RESULTS_DIR / "experiment_gb_xgboost.csv", index=False)

results_df.to_latex(RESULTS_DIR / "experiment_gb_xgboost.tex", index=False)
