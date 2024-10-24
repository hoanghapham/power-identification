#%%
import sys
from pathlib import Path
from dotenv import dotenv_values

# Make packages in projects directory available for importing
env = dotenv_values(".env")
PROJECT_DIR = Path(env["PROJECT_DIR"])

sys.path.append(str(PROJECT_DIR))

# Import
import torch
import pandas as pd

from lib.data_processing import load_data
from lib.evaluation import get_average_metrics, train_evaluate_rnn
from lib.utils import check_cuda_memory, write_ndjson_file
from lib.logger import CustomLogger


# Set up logger
logger = CustomLogger("experiment_gb_rnnbig", log_to_local=False, log_path=PROJECT_DIR / "logs")


# Setup folders
RESULTS_DIR = PROJECT_DIR / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Setup device
PREFERRED_DEVICE = "cuda"

if torch.cuda.is_available():
    DEVICE = PREFERRED_DEVICE
    logger.info(f"CUDA available. Use: {DEVICE}")
    check_cuda_memory()
else:
    DEVICE = "cpu"
    logger.info(f"CUDA not available. Use: {DEVICE}")


# Kfold
NFOLDS = 5

#%%

# Load data
logger.info(f"Load file: power-gb-train.tsv")
data = load_data(file_path_list=[PROJECT_DIR / "data/train/power/power-gb-train.tsv"],text_head="text_en")
logger.info(f"Data size: {len(data)}, % positive class: {sum(data.labels) / len(data) * 100:.2f}%")


#%%
# Recurrent Neural Network
## Word features
# logger.info("Fit RNNClassifier model, word feature")

# model_RNNClassifier_word, results_RNNClassifier_word = train_evaluate_rnn(data, NFOLDS, "word", logger, device=DEVICE, hidden_dim=128)
# avg_RNNClassifier_word = get_average_metrics(results_RNNClassifier_word)
# results_RNNClassifier_word.append({"fold": "average", **avg_RNNClassifier_word})

# logger.info([f"{key}: {value:.3f}" for key, value in avg_RNNClassifier_word.items()])
# write_ndjson_file(results_RNNClassifier_word, RESULTS_DIR / "results_gb_RNNClassifierbig_word.json")

#%%
## Char feature
logger.info("Fit RNNClassifier model, char feature")

model_RNNClassifier_char, results_RNNClassifier_char = train_evaluate_rnn(data, NFOLDS, "char", logger, device=DEVICE, hidden_dim=128)
avg_RNNClassifier_char = get_average_metrics(results_RNNClassifier_char)
results_RNNClassifier_char.append({"fold": "average", **avg_RNNClassifier_char})

logger.info([f"{key}: {value:.3f}" for key, value in avg_RNNClassifier_char.items()])
write_ndjson_file(results_RNNClassifier_char, RESULTS_DIR / "results_gb_RNNClassifierbig_char.json")



#%%
# Write results
logger.info("Write result")
results_aggr = [
    # {"model": "RNNClassifier_word", **avg_RNNClassifier_word},
    {"model": "RNNClassifier_char", **avg_RNNClassifier_char}
]

write_ndjson_file(results_aggr, RESULTS_DIR / "results_gb_RNNClassifierbig.json")

results_df = pd.DataFrame(data=results_aggr)
results_df.to_csv(RESULTS_DIR / "results_gb_RNNClassifierbig.csv", index=False)
results_df.to_latex(RESULTS_DIR / "results_gb_RNNClassifierbig.tex", index=False)