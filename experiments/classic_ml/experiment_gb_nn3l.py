#%%
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))


# Import
import torch
import pandas as pd

from lib.data_processing import load_data
from lib.evaluation import get_average_metrics, train_evaluate_nn
from lib.utils import check_cuda_memory, write_ndjson_file
from lib.logger import CustomLogger

# Set up logger
logger = CustomLogger("experiment_gb_nn3l", log_to_local=True, log_path=PROJECT_DIR / "logs")


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
# logger.info("Fit NeuralNetwork model, word feature")

# model_NeuralNetwork3l_word, results_NeuralNetwork3l_word = train_evaluate_nn(data, NFOLDS, "word", logger, device=DEVICE, n_hidden_layers=3)
# avg_NeuralNetwork3l_word = get_average_metrics(results_NeuralNetwork3l_word)
# results_NeuralNetwork3l_word.append({"fold": "average", **avg_NeuralNetwork3l_word})

# logger.info([f"{key}: {value:.3f}" for key, value in avg_NeuralNetwork3l_word.items()])
# write_ndjson_file(results_NeuralNetwork3l_word, RESULTS_DIR / "results_gb_NeuralNetwork3l_word.json")


# %%
## Char features
logger.info("Fit NeuralNetwork model, char feature")

model_NeuralNetwork3l_char, results_NeuralNetwork3l_char = train_evaluate_nn(data, NFOLDS, "char", logger, device=DEVICE, n_hidden_layers=3)
avg_NeuralNetwork3l_char = get_average_metrics(results_NeuralNetwork3l_char)
results_NeuralNetwork3l_char.append({"fold": "average", **avg_NeuralNetwork3l_char})

logger.info([f"{key}: {value:.3f}" for key, value in avg_NeuralNetwork3l_char.items()])
write_ndjson_file(results_NeuralNetwork3l_char, RESULTS_DIR / "results_gb_NeuralNetwork3l_char.json")

# %%


# Write results
logger.info("Write result")
results_aggr = [
    # {"model": "NeuralNetwork3l_word", **avg_NeuralNetwork3l_word},
    {"model": "NeuralNetwork3l_char", **avg_NeuralNetwork3l_char}
]

write_ndjson_file(results_aggr, RESULTS_DIR / "results_gb_NeuralNetwork3l.json")

results_df = pd.DataFrame(data=results_aggr)
results_df.to_csv(RESULTS_DIR / "results_gb_NeuralNetwork3l.csv", index=False)
results_df.to_latex(RESULTS_DIR / "results_gb_NeuralNetwork3l.tex", index=False)