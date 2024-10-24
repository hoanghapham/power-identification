
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
# PROJECT_DIR = Path.cwd().parent.parent.resolve()
sys.path.append(str(PROJECT_DIR))

# Import
import pandas as pd

from lib.data_processing import load_data
from lib.evaluation import train_evaluate_xgboost, get_average_metrics
from lib.logger import CustomLogger
from lib.utils import write_ndjson_file


RESULTS_DIR = PROJECT_DIR / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set up logger
logger = CustomLogger("results_gb_XGBoost10", log_to_local=True, log_path=PROJECT_DIR / "logs")

# Params
NFOLDS = 5
PREFERRED_DEVICE = 'cuda'

#%%
## Load data
logger.info(f"Load file: power-gb-train.tsv")
data = load_data(file_path_list=[PROJECT_DIR / "data/train/power/power-gb-train.tsv"],text_head="text_en")
logger.info(f"Data size: {len(data)}, % positive class: {sum(data.labels) / len(data) * 100:.2f}%")



#%%
## Word feature
# logger.info("Fit XGBoost model, word feature")

# # Train, test, evalute
# model_XGBoost_word, results_XGBoost_word = train_evaluate_xgboost(data, NFOLDS, "word", logger, PREFERRED_DEVICE, iterations=10)
# avg_XGBoost_word = get_average_metrics(results_XGBoost_word)
# results_XGBoost_word.append({"fold": "average", **avg_XGBoost_word})

# logger.info([f"{key}: {value:.3f}" for key, value in avg_XGBoost_word.items()])
# write_ndjson_file(results_XGBoost_word, RESULTS_DIR / "results_gb_XGBoost10iter_word.json")

#%%
## char feature

logger.info("Fit XGBoost model, char feature")

# Train, test, evalute
model_XGBoost_char, result_XGBoost_char = train_evaluate_xgboost(data, NFOLDS, "char", logger, PREFERRED_DEVICE, iterations=10)
avg_XGBoost_char = get_average_metrics(result_XGBoost_char)
result_XGBoost_char.append({"fold": "average", **avg_XGBoost_char})

logger.info([f"{key}: {value:.3f}" for key, value in avg_XGBoost_char.items()])
write_ndjson_file(result_XGBoost_char, RESULTS_DIR / "results_gb_XGBoost10iter_char.json")


# Write results
logger.info("Write result")
results_aggr = [
    # {"model": "XGBoost_word", **avg_XGBoost_word},
    {"model": "XGBoost_char", **avg_XGBoost_char}
]

write_ndjson_file(results_aggr, RESULTS_DIR / "results_gb_XGBoost10iter.json")

results_df = pd.DataFrame(data=results_aggr)
results_df.to_csv(RESULTS_DIR / "results_gb_XGBoost10iter.csv", index=False)
results_df.to_latex(RESULTS_DIR / "results_gb_XGBoost10iter.tex", index=False)
