
# Classifiers from sklearn
# Models tested: 
# - LinearSVC
# - LogisticRegression
# - SGDClassifier
# - DecisionTreeClassifier
# - RandomForestClassifier

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
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

from lib.data_processing import load_data
from lib.utils import write_ndjson_file
from lib.evaluation import get_average_metrics, train_evaluate_sklearn
from lib.logger import CustomLogger
from pathlib import Path


# Set up
RESULTS_DIR = PROJECT_DIR / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = CustomLogger("experiment_hr_sklearn", log_to_local=False)

NFOLDS = 5


#%%

## Load data
logger.info(f"Load file: power-hr-train.tsv")
data = load_data(file_path_list=[PROJECT_DIR / "data/train/power/power-hr-train.tsv"],text_head="text_en")
logger.info(f"Data size: {len(data)}, % positive class: {sum(data.labels) / len(data) * 100:.2f}%")


# %%
## Linear SVC
### Word feature
logger.info("Fit LinearSVC model, word feature")
base_svc = LinearSVC(dual="auto")
model_LinearSVC_word = CalibratedClassifierCV(estimator=base_svc, cv=5)

# Train, test, evalute
model_LinearSVC_word, results_LinearSVC_word = train_evaluate_sklearn(model_LinearSVC_word, data, NFOLDS, "word", logger)
avg_LinearSVC_word = get_average_metrics(results_LinearSVC_word)
results_LinearSVC_word.append({"fold": "average", **avg_LinearSVC_word})

logger.info([f"{key}: {value:.3f}" for key, value in avg_LinearSVC_word.items()])
write_ndjson_file(results_LinearSVC_word, RESULTS_DIR / "results_hr_LinearSVC_word.json")


#%%
### Char features
logger.info("Fit LinearSVC model, char feature")
base_svc = LinearSVC(dual="auto")
model_LinearSVC_char = CalibratedClassifierCV(estimator=base_svc, cv=5)

# Train, test, evalute
model_LinearSVC_char, results_LinearSVC_char = train_evaluate_sklearn(model_LinearSVC_char, data, NFOLDS, "char", logger)
avg_LinearSVC_char = get_average_metrics(results_LinearSVC_char)
results_LinearSVC_char.append({"fold": "average", **avg_LinearSVC_char})

logger.info([f"{key}: {value:.3f}" for key, value in avg_LinearSVC_char.items()])
write_ndjson_file(results_LinearSVC_char, RESULTS_DIR / "results_hr_LinearSVC_char.json")


# %%
## Logistic Regression
### Word features

logger.info("Fit LogisticRegression model, word feature")
model_LogReg_word = LogisticRegression()

# Train, test, evalute
model_LogReg_word, results_LogReg_word = train_evaluate_sklearn(model_LogReg_word, data, NFOLDS, "word", logger)
avg_LogReg_word = get_average_metrics(results_LogReg_word)
results_LogReg_word.append({"fold": "average", **avg_LogReg_word})

logger.info([f"{key}: {value:.3f}" for key, value in avg_LogReg_word.items()])
write_ndjson_file(results_LogReg_word, RESULTS_DIR / "results_hr_LogReg_word.json")

# %%
### Char features

logger.info("Fit LogisticRegression model, char feature")
model_LogReg_char = LogisticRegression()

# Train, test, evalute
model_LogReg_char, results_LogReg_char = train_evaluate_sklearn(model_LogReg_char, data, NFOLDS, "char", logger)
avg_LogReg_char = get_average_metrics(results_LogReg_char)
results_LogReg_char.append({"fold": "average", **avg_LogReg_char})

logger.info([f"{key}: {value:.3f}" for key, value in avg_LogReg_char.items()])
write_ndjson_file(results_LogReg_char, RESULTS_DIR / "results_hr_LogReg_char.json")

#%%
## SGDClassifier
### word feature

logger.info("Fit SGDClassifier model, word feature")
model_SGD_word = SGDClassifier(loss="log_loss")

# Train, test, evalute
model_SGD_word, results_SGD_word = train_evaluate_sklearn(model_SGD_word, data, NFOLDS, "word", logger)
avg_SGD_word = get_average_metrics(results_SGD_word)
results_SGD_word.append({"fold": "average", **avg_SGD_word})

logger.info([f"{key}: {value:.3f}" for key, value in avg_SGD_word.items()])
write_ndjson_file(results_SGD_word, RESULTS_DIR / "results_hr_SGD_word.json")


# %%
### Char features
logger.info("Fit SGDClassifier model, char feature")
model_SGD_char = SGDClassifier(loss="log_loss")

# Train, test, evalute
model_SGD_char, results_SGD_char = train_evaluate_sklearn(model_SGD_char, data, NFOLDS, "char", logger)
avg_SGD_char = get_average_metrics(results_SGD_char)
results_SGD_char.append({"fold": "average", **avg_SGD_char})

logger.info([f"{key}: {value:.3f}" for key, value in avg_SGD_char.items()])
write_ndjson_file(results_SGD_char, RESULTS_DIR / "results_hr_SGD_char.json")

#%%
## Decision Tree Classifier
### word feature
logger.info("Fit DecisionTreeClassifier model, word feature")
model_DecisionTree_word = DecisionTreeClassifier()

# Train, test, evalute
model_DecisionTree_word, results_DecisionTree_word = train_evaluate_sklearn(model_DecisionTree_word, data, NFOLDS, "word", logger)
avg_DecisionTree_word = get_average_metrics(results_DecisionTree_word)
results_DecisionTree_word.append({"fold": "average", **avg_DecisionTree_word})

logger.info([f"{key}: {value:.3f}" for key, value in avg_DecisionTree_word.items()])
write_ndjson_file(results_DecisionTree_word, RESULTS_DIR / "results_hr_DecisionTree_word.json")

#%%
### char feature
logger.info("Fit DecisionTreeClassifier model, char feature")
model_DecisionTree_char = DecisionTreeClassifier()

# Train, test, evalute
model_DecisionTree_char, results_DecisionTree_char = train_evaluate_sklearn(model_DecisionTree_char, data, NFOLDS, "char", logger)
avg_DecisionTree_char = get_average_metrics(results_DecisionTree_char)
results_DecisionTree_char.append({"fold": "average", **avg_DecisionTree_char})

logger.info([f"{key}: {value:.3f}" for key, value in avg_DecisionTree_char.items()])
write_ndjson_file(results_DecisionTree_char, RESULTS_DIR / "results_hr_DecisionTree_char.json")


#%%
## Random Forest Classifier
### word feature

logger.info("Fit RandomForestClassifier model, word feature")
model_RandomForest_word = RandomForestClassifier()

# Train, test, evalute
model_RandomForest_word, results_RandomForest_word = train_evaluate_sklearn(model_RandomForest_word, data, NFOLDS, "word", logger)
avg_RandomForest_word = get_average_metrics(results_RandomForest_word)
results_RandomForest_word.append({"fold": "average", **avg_RandomForest_word})

logger.info([f"{key}: {value:.3f}" for key, value in avg_RandomForest_word.items()])
write_ndjson_file(results_RandomForest_word, RESULTS_DIR / "results_hr_RandomForest_word.json")


#%%
### char feature

logger.info("Fit RandomForestClassifier model, char feature")
model_RandomForest_char = RandomForestClassifier()

# Train, test, evalute
model_RandomForest_char, results_RandomForest_char = train_evaluate_sklearn(model_RandomForest_char, data, NFOLDS, "char", logger)
avg_RandomForest_char = get_average_metrics(results_RandomForest_char)
results_RandomForest_char.append({"fold": "average", **avg_RandomForest_char})

logger.info([f"{key}: {value:.3f}" for key, value in avg_RandomForest_char.items()])
write_ndjson_file(results_RandomForest_char, RESULTS_DIR / "results_hr_RandomForest_char.json")


#%%
# Final results
logger.info("Write final results")

results_aggr = [
    {"model": "LinearSVC_word", **avg_LinearSVC_word},
    {"model": "LinearSVC_char", **avg_LinearSVC_char},
    {"model": "SGD_word", **avg_SGD_word},
    {"model": "SGD_char", **avg_SGD_char},
    {"model": "DecisionTree_word", **avg_DecisionTree_word},
    {"model": "DecisionTree_char", **avg_DecisionTree_char},
    {"model": "RandomForest_word", **avg_RandomForest_word},
    {"model": "RandomForest_char", **avg_RandomForest_char}
]

results_df = pd.DataFrame(data=results_aggr)
results_df.to_csv(RESULTS_DIR / "results_hr_sklearn.csv", index=False)
results_df.to_latex(RESULTS_DIR / "results_hr_sklearn.tex", index=False)
