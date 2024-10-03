
## result: Classical models vs. Neural Networks on GB dataset
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

from sklearn.model_selection import KFold

from lib.data_processing import load_data, split_data, RawDataset
from lib.evaluation import evaluate
from lib.logger import CustomLogger
from lib.utils import write_ndjson_file


RESULTS_DIR = PROJECT_DIR / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set up logger
logger = CustomLogger("result_gb_xgboost", log_to_local=False)

# Helper functions
def get_average_metrics(result_list: list[dict]) -> dict:
    accuracy = np.mean([[result['accuracy'] for result in result_list]])
    precision = np.mean([[result['precision'] for result in result_list]])
    recall = np.mean([[result['recall'] for result in result_list]])
    f1 = np.mean([[result['f1'] for result in result_list]])
    auc = np.mean([[result['auc'] for result in result_list]])
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

def train_evaluate_xgboost(data: RawDataset, nfolds: int, feature_type: str):
    result_list = []
    kfold = KFold(n_splits=nfolds, shuffle=False, random_state=None)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(data), start=1):
        logger.info(f"CV fold {fold_idx}")

        # Prepare encoders
        if feature_type == "word": 
            encoder = TfidfVectorizer(max_features=50000)
        elif feature_type == "char":
            encoder = TfidfVectorizer(max_features=50000, analyzer="char", ngram_range=(3,5), use_idf=True, sublinear_tf=True)
        else:
            raise ValueError(f"Not supported feature type: {feature_type}")
        
        encoder.fit(data.subset(train_idx).texts)

        X_train = encoder.transform(data.subset(train_idx).texts)
        X_test = encoder.transform(data.subset(test_idx).texts)
        label_train = data.subset(train_idx).labels
        label_test = data.subset(test_idx).labels

        train_dmat = xgb.DMatrix(X_train, pd.array(label_train).astype("category"))
        test_dmat = xgb.DMatrix(X_test, pd.array(label_test).astype("category"))

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

        evals_words = [(train_dmat, "train")]

        model = xgb.train(
            params = params,
            dtrain = train_dmat_words,
            num_boost_round = ITERATIONS,
            evals = evals_words,
            verbose_eval = False
        )

        probs = model.predict(test_dmat)
        result = evaluate(label_test, probs > 0.5, probs)

        # Fit model
        result_list.append({"fold": str(fold_idx), **result})

    return model, result_list

## Load data
logger.info(f"Load file: power-gb-train.tsv")
data = load_data(file_path_list=[PROJECT_DIR / "data/train/power/power-gb-train.tsv"],text_head="text_en")
train_raw, test_raw = split_data(data, test_size=0.2, random_state=0)
logger.info(f"Data size: {len(data)}, % positive class: {sum(data.labels) / len(data) * 100:.2f}%")

NFOLDS = 5

#%%
## Word feature
logger.info("Fit XGBoost model, word feature")

# Train, test, evalute
model_XGBoost_word, result_XGBoost_word = train_evaluate_xgboost(data, NFOLDS, "word")
avg_XGBoost_word = get_average_metrics(result_XGBoost_word)
result_XGBoost_word.append({"fold": "average", **avg_XGBoost_word})

logger.info([f"{key}: {value:.3f}" for key, value in avg_XGBoost_word.items()])
write_ndjson_file(result_XGBoost_word, RESULTS_DIR / "results_XGBoost_word.json")

#%%
## char feature

logger.info("Fit XGBoost model, char feature")

# Train, test, evalute
model_XGBoost_char, result_XGBoost_char = train_evaluate_xgboost(data, NFOLDS, "char")
avg_XGBoost_char = get_average_metrics(result_XGBoost_char)
result_XGBoost_char.append({"fold": "average", **avg_XGBoost_char})

logger.info([f"{key}: {value:.3f}" for key, value in avg_XGBoost_char.items()])
write_ndjson_file(result_XGBoost_char, RESULTS_DIR / "results_XGBoost_char.json")



# Write results
logger.info("Write result")
results_aggr = [
    {"model": "XGBoost_word", **result_XGBoost_word},
    {"model": "XGBoost_char", **result_XGBoost_char}
]

write_ndjson_file(results_aggr, RESULTS_DIR / "result_gb_xgboost.json")

results_df = pd.DataFrame(data=results_aggr)
results_df.to_csv(RESULTS_DIR / "result_gb_xgboost.csv", index=False)
results_df.to_latex(RESULTS_DIR / "result_gb_xgboost.tex", index=False)
