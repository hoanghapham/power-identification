
## Experiment: Different word embedding methods on GB dataset
#
# This file contains the following embedding methods based on words:
# - Binary Bow with max features 1000, 5000, 10000, 20000, 50000
# - TF-IDF with max features 1000, 5000, 10000, 20000, 50000
# - Word2Vec with dimensions 100, 200, 300
# - GloVe (dimension 300) with max features 100, 200, 300
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
from collections import defaultdict
from scipy.sparse import issparse

from sklearn.model_selection import KFold
from lib.models import TrainConfig, NeuralNetwork
from lib.data_processing import load_data, get_embeddings, create_dataloader
from lib.evaluation import evaluate
from lib.logger import CustomLogger

device = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = PROJECT_DIR / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set up logger
logger = CustomLogger("experiment_word_embeddings", log_to_local=False)

## Load data
logger.info(f"Load file: power-gb-train.tsv")
data = load_data(file_path_list=[PROJECT_DIR / "data/train/power/power-gb-train.tsv"], text_head="text")
logger.info(f"Data size: {len(data)}, % positive class: {sum(data.labels) / len(data) * 100:.2f}%")

# Extract text and labels (full dataset for KFold)
df = pd.DataFrame({'text': data.texts, 'label': data.labels})

# Set training configurations
train_config = TrainConfig(num_epochs=10,early_stop=False,violation_limit=5)

# Train and evaluate models
def train_and_evaluate(embedding_method, df=df, max_features=None, train_config=train_config, n_splits=5, device=device):
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    # Initialize dictionaries to store fold-wise metrics
    fold_metrics = defaultdict(list)  # Store metrics across all folds
    
    for fold, (train_index, val_index) in enumerate(kf.split(df)):
        logger.info(f"Running fold {fold + 1}/{n_splits}")
        
        # Split data into training and validation sets
        train_data = df.iloc[train_index]  # Get training set
        val_data = df.iloc[val_index]      # Get validation set
        
        # Prepare texts and labels
        X_train_fold = train_data['text'].values
        y_train_fold = train_data['label'].values
        X_val_fold = val_data['text'].values
        y_val_fold = val_data['label'].values
        
        # Generate embeddings for the train and validation sets
        logger.info(f"Generating {embedding_method} embeddings with max features {max_features}")
        X_train_embedded, X_val_embedded = get_embeddings(embedding_method, X_train_fold, X_val_fold, max_features=max_features)

        # Create DataLoader for the fold's training data
        train_dataloader = create_dataloader(X_train_embedded, y_train_fold, batch_size=32, shuffle=True)

        # Initialize the model
        model_nn = NeuralNetwork(input_size=X_train_embedded.shape[1], hidden_size=128, device=device)
        
        # Train the model
        logger.info(f"Fitting feed-forward neural network model with {embedding_method} embeddings")
        model_nn.fit(train_dataloader, train_config, disable_progress_bar=False)
        
        # Evaluate on the validation fold
        with torch.no_grad():
            if issparse(X_val_embedded):
                X_val_embedded = X_val_embedded.toarray()
            X_val_tensor = torch.tensor(X_val_embedded, dtype=torch.float32).to(model_nn.device)
            y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).to(model_nn.device)

            # Get predicted labels and logits from the model
            y_val_pred = model_nn.predict(X_val_tensor)  # Get predicted labels
            logits = model_nn.forward(X_val_tensor)  # Get logits

            # Evaluate the results using evaluate function
            metrics = evaluate(y_val_tensor.cpu().numpy(), y_val_pred.cpu().numpy(), logits.cpu().numpy())
            
            # Store the metrics for this fold
            for key, value in metrics.items():
                fold_metrics[key].append(value)
                
        logger.info(f"Fold {fold + 1} results: {metrics}")
        
    # Calculate average metrics across all folds
    average_metrics = {metric: np.mean(values) for metric, values in fold_metrics.items()}
    
    logger.info(f"Cross-validation complete. Average results: {average_metrics}")
            
    return fold_metrics, average_metrics

# Main script to run the experiments
if __name__ == "__main__":
    results_dict = {}
    for embedding_method, max_features in {'Binary_BoW': [1000, 5000, 10000, 20000, 50000],
                                           'TF-IDF': [1000, 5000, 10000, 20000, 50000],
                                           'Word2Vec': [100, 200, 300],
                                           'GloVe': [100, 200, 300],
                                           'DistilBERT': [64, 128, 256]}.items():
        for feature in max_features:
            logger.info(f"Running {embedding_method} with max features/dimensions/lengths: {feature}")
            # Train and evaluate with cross-validation
            fold_metrics, average_metrics = train_and_evaluate(embedding_method=embedding_method, 
                                                               df=df, 
                                                               max_features=feature,
                                                               train_config=train_config, 
                                                               n_splits=5, 
                                                               device=device)                   
            
            # Log results for this embedding method and configuration
            logger.info(f"Results for {embedding_method} max feature {feature}: {average_metrics}")
            
            # Save results to the dictionary (store both per-fold and average metrics)
            results_dict[f"{embedding_method} max feature {feature}"] = {"fold_metrics": fold_metrics,
                                                                         "average_metrics": average_metrics}
            
    results_df = pd.DataFrame.from_dict(results_dict, orient="index") \
        .reset_index().rename(columns={"index": "model"})
        
    # Write result to JSON, CSV, LaTeX
    with open(RESULTS_DIR / "results_embeddings.json", "w") as f:
        json.dump(results_dict, f)
        
    results_df.to_csv(RESULTS_DIR / "results_embeddings.csv", index=False)
    
    results_df.to_latex(RESULTS_DIR / "results_embeddings.tex", index=False)
