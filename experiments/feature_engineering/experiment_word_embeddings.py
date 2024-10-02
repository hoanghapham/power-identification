
## Experiment: Different word embedding methods on GB dataset
#
# This file contains the following embedding methods:
# - Binary Bow with max features 1000, 5000, 10000, 20000, 50000
# - TF-IDF with max features 1000, 5000, 10000, 20000, 50000
# - Word2Vec with dimensions 100, 200, 300
# - GloVe with max features 100, 200, 300
# - DistilBERT embeddings with max lengths 64, 128, 256

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
import torch
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from sklearn.model_selection import KFold
from lib.models import TrainConfig, NeuralNetwork
from lib.data_processing import load_data, split_data, get_embeddings, create_dataloader
from lib.evaluation import evaluate
from lib.logger import CustomLogger

RESULTS_DIR = PROJECT_DIR / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set up logger
logger = CustomLogger("experiment_word_embeddings", log_to_local=False)

## Load data
logger.info(f"Load file: power-gb-train.tsv")
data = load_data(file_path_list=[PROJECT_DIR / "data/train/power/power-gb-train.tsv"], text_head="text")
train_raw, test_raw = split_data(data, test_size=0.2, random_state=0)
logger.info(f"Data size: {len(data)}, % positive class: {sum(data.labels) / len(data) * 100:.2f}%")

# Extract text and labels
X_train = train_raw.texts
y_train = train_raw.labels
y_train = np.array(y_train)
X_test = test_raw.texts
y_test = test_raw.labels
y_test = np.array(y_test)

# Set training configurations
train_config = TrainConfig(num_epochs=10,early_stop=False,violation_limit=5)

# Train and evaluate models
def train_and_evaluate(embedding_method, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                       max_features=None, train_config=train_config, n_splits=5):
    # Generate embeddings for the train and test sets
    logger.info(f"Generating {embedding_method} embeddings with max features {max_features}")
    X_train_embedded, X_test_embedded = get_embeddings(embedding_method, X_train, X_test, max_features=max_features)

    # Cross-validate
    logger.info("Starting cross-validation")
    kf = KFold(n_splits=n_splits)
    accuracies = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_embedded)):
        logger.info(f"Training on fold {fold + 1}/{n_splits}")

        # Split data into training and validation sets
        X_train_fold, X_val_fold = X_train_embedded[train_index], X_train_embedded[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Create and train the model
        model_nn = NeuralNetwork(input_size=X_train_embedded.shape[1], hidden_size=128)
        train_dataloader = create_dataloader(X_train_fold, y_train_fold, batch_size=32, shuffle=True)
        
        logger.info(f"Fitting feed-forward neural network model with {embedding_method} embeddings")
        model_nn.fit(train_dataloader, train_config, disable_progress_bar=False)
        
        # Evaluate on the validation fold
        with torch.no_grad():
            # Convert X_val_fold to dense if it is sparse
            if issparse(X_val_fold):
                X_val_fold = X_val_fold.toarray()
            X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(model_nn.device)
            y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).to(model_nn.device)

            # Get predicted labels
            y_val_pred = model_nn.predict(X_val_tensor)

            # Calculate accuracy
            corrects = (y_val_pred == y_val_tensor).sum().item()
            accuracy = corrects / len(y_val_tensor)
            accuracies.append(accuracy)

            logger.info(f"Fold {fold + 1}/{n_splits} Accuracy: {accuracy:.4f}")

    # Average accuracy across all folds
    average_accuracy = np.mean(accuracies)
    logger.info(f"Cross-Validation Average Accuracy: {average_accuracy:.4f}")
    
    # Create DataLoader for training
    train_dataloader = create_dataloader(X_train_embedded, y_train, batch_size=32, shuffle=True)

    # Create and train the final model on the full training set
    logger.info(f"Fitting feed-forward neural network model with {embedding_method} embeddings on the entire training set")
    model_nn = NeuralNetwork(input_size=X_train_embedded.shape[1], hidden_size=128)
    model_nn.fit(train_dataloader, train_config, disable_progress_bar=False)
    
    # Test and evaluate on the test set
    logger.info(f"Testing and evaluating on the test set")
    with torch.no_grad():
        # Convert X_val_fold to dense if it is sparse
        if issparse(X_test_embedded):
            X_test_embedded = X_test_embedded.toarray()
        X_test_tensor = torch.tensor(X_test_embedded, dtype=torch.float32).to(model_nn.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(model_nn.device)

        # Get predicted labels and logits from the model
        y_pred = model_nn.predict(X_test_tensor)  # Get predicted labels
        logits = model_nn.forward(X_test_tensor)  # Get logits

        # Evaluate the results using evaluate function
        result = evaluate(y_test_tensor.cpu().numpy(), y_pred.cpu().numpy(), logits.cpu().numpy())
        return result

# Main script to run the experiments
if __name__ == "__main__":
    results_dict = {}
    for embedding_method, max_features in {'Binary_BoW': [1000, 5000, 10000, 20000, 50000],
                                           'TF-IDF': [1000, 5000, 10000, 20000, 50000],
                                           'Word2Vec': [100, 200, 300],
                                           'GloVe': [100, 200, 300],
                                           'DistilBERT': [64, 128, 256]}.items():
        for feature in max_features:
            result = train_and_evaluate(embedding_method, max_features=feature)
            logger.info("Write result")
            results_dict[f"{embedding_method} max feature {feature}"] = result
            
    results_df = pd.DataFrame.from_dict(results_dict, orient="index") \
        .reset_index().rename(columns={"index": "model"})
        
    # Write result to JSON, CSV, LaTeX
    with open(RESULTS_DIR / "results_embeddings.json", "w") as f:
        json.dump(results_dict, f)
        
    results_df.to_csv(RESULTS_DIR / "results_embeddings.csv", index=False)
    
    results_df.to_latex(RESULTS_DIR / "results_embeddings.tex", index=False)
