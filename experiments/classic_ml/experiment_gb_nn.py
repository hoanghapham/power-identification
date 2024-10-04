#%%
import sys
from pathlib import Path
from dotenv import dotenv_values

# Make packages in projects directory available for importing
env = dotenv_values(".env")
PROJECT_DIR = Path(env["PROJECT_DIR"])
# PROJECT_DIR = Path.cwd().parent.parent.resolve()

sys.path.append(str(PROJECT_DIR))

# Import
import torch
import pandas as pd

from lib.data_processing import load_data
from lib.evaluation import get_average_metrics, train_evaluate_nn
from lib.utils import check_cuda_memory, write_ndjson_file
from lib.logger import CustomLogger

# Set up logger
logger = CustomLogger("experiment_gb_nn", log_to_local=False)


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
# Neural Network
## Word features
logger.info("Fit NeuralNetwork model, word feature")

model_NeuralNetwork_word, results_NeuralNetwork_word = train_evaluate_nn(data, NFOLDS, "word", logger, device=DEVICE)
avg_NeuralNetwork_word = get_average_metrics(results_NeuralNetwork_word)
results_NeuralNetwork_word.append({"fold": "average", **avg_NeuralNetwork_word})

logger.info([f"{key}: {value:.3f}" for key, value in avg_NeuralNetwork_word.items()])
write_ndjson_file(results_NeuralNetwork_word, RESULTS_DIR / "results_gb_NeuralNetwork_word.json")


# %%
## Char features
logger.info("Fit NeuralNetwork model, char feature")

model_NeuralNetwork_char, results_NeuralNetwork_char = train_evaluate_nn(data, NFOLDS, "char", logger, device=DEVICE)
avg_NeuralNetwork_char = get_average_metrics(results_NeuralNetwork_char)
results_NeuralNetwork_char.append({"fold": "average", **avg_NeuralNetwork_char})

logger.info([f"{key}: {value:.3f}" for key, value in avg_NeuralNetwork_char.items()])
write_ndjson_file(results_NeuralNetwork_char, RESULTS_DIR / "results_gb_NeuralNetwork_char.json")

# %%


# Write results
logger.info("Write result")
results_aggr = [
    {"model": "NeuralNetwork_word", **avg_NeuralNetwork_word},
    {"model": "NeuralNetwork_char", **avg_NeuralNetwork_char}
]

write_ndjson_file(results_aggr, RESULTS_DIR / "results_gb_NeuralNetwork.json")

results_df = pd.DataFrame(data=results_aggr)
results_df.to_csv(RESULTS_DIR / "results_gb_NeuralNetwork.csv", index=False)
results_df.to_latex(RESULTS_DIR / "results_gb_NeuralNetwork.tex", index=False)