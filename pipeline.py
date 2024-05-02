import torch
import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from data_cv import get_data

class IdeologyPowerDetectionPipeline:
    def __init__(self, data_dir='data', models_dir='models', pred_dir='predictions'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.pred_dir = pred_dir
        self.vec = None
        self.model = None

    def load_data(self, task, pcode, num_splits=None):
        data_files = [os.path.join(self.data_dir, task, f"{task}-{pcode}-train.tsv")]
        if num_splits is None:
            t_trn, y_trn, t_val, y_val = get_data(data_files[0])
            return [(t_trn, y_trn, t_val, y_val)]
        else:
            data_splits = get_data(data_files[0], num_splits=num_splits)
            return data_splits

    def preprocess_data(self, texts):
        if self.vec is None:
            self.vec = TfidfVectorizer(sublinear_tf=True, analyzer="char", ngram_range=(1,3))
            x_trn = self.vec.fit_transform(texts)
        else:
            x_trn = self.vec.transform(texts)
        return x_trn

    def train_model(self, x_trn, y_trn):
        model = LogisticRegression(max_iter=500)
        model.fit(x_trn, y_trn)
        self.model = model

    def hyperparameter_tuning(self, x_trn, y_trn):
        logreg_params = {
            'C': uniform(0.1, 10),  # Uniform distribution between 0.1 and 10
            'max_iter': randint(100, 1000)  # Discrete values between 100 and 1000
        }

        param_distributions = logreg_params

        random_search = RandomizedSearchCV(
            LogisticRegression(), param_distributions, 
            n_iter=10, cv=5, scoring='f1_macro', verbose=2, n_jobs=-1
        )
        random_search.fit(x_trn, y_trn)

        best_params = random_search.best_params_
        best_model = random_search.best_estimator_

        return best_params, best_model

    def evaluate_model(self, x_val, y_val):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        y_pred = self.model.predict(x_val)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_val, y_pred, average='macro'
        )
        return precision, recall, fscore

    def save_model(self, task, pcode):
        os.makedirs(self.models_dir, exist_ok=True)
        model_file = os.path.join(self.models_dir, f"{task}-{pcode}.joblib")
        joblib.dump(self.model, model_file)

    def save_predictions(self, task, pcode, test_pred):
        os.makedirs(self.pred_dir, exist_ok=True)
        pred_file = os.path.join(self.pred_dir, f"{task}-{pcode}-predictions.tsv")
        with open(pred_file, "wt") as fpred:
            for i, p in enumerate(test_pred):
                print(f"{i+1}\t{p}", file=fpred)

pipeline = IdeologyPowerDetectionPipeline()

# Example usage for proof of concept
task = 'power'
pcode = 'at'
num_splits = 5  # Number of folds for K-fold cross-validation

# Initialize lists to store scores for each fold
precisions, recalls, fscores = [], [], []

# Load data
data_splits = pipeline.load_data(task, pcode, num_splits)

# Concatenate all training data from different folds
t_trn_all, y_trn_all = [], []
for t_trn, y_trn, _, _ in data_splits:
    t_trn_all.extend(t_trn)
    y_trn_all.extend(y_trn)
    
# Preprocess entire training data
x_trn_all = pipeline.preprocess_data(t_trn_all)

# Perform hyperparameter tuning once on the entire training dataset
best_params, best_model = pipeline.hyperparameter_tuning(x_trn_all, y_trn_all)

# Load data again
data_splits = pipeline.load_data(task, pcode, num_splits)

# Perform K-fold cross-validation
for fold, (t_trn, y_trn, t_val, y_val) in enumerate(data_splits):
    print(f"Fold {fold + 1}/{num_splits}")
    
    # Check for empty splits
    if len(t_trn) == 0 or len(t_val) == 0:
        print("Empty split. Skipping evaluation for this fold.")
        continue
    
    # Preprocess data
    x_trn = pipeline.preprocess_data(t_trn)
    x_val = pipeline.preprocess_data(t_val)
    
    # Train model using the best hyperparameters
    pipeline.model = best_model
    
    # Evaluate model
    precision, recall, fscore = pipeline.evaluate_model(x_val, y_val)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {fscore:.4f}")

    # Save scores for this fold
    precisions.append(precision)
    recalls.append(recall)
    fscores.append(fscore)

    # Save model (optional)
    # pipeline.save_model(task, f"{pcode}-fold{fold}")

# Calculate mean and standard deviation of scores
if len(precisions) > 0:
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_fscore = np.mean(fscores)
    std_precision = np.std(precisions)
    std_recall = np.std(recalls)
    std_fscore = np.std(fscores)

    # Print summary of evaluation metrics
    print("Evaluation Summary:")
    print(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"Mean F1-score: {mean_fscore:.4f} ± {std_fscore:.4f}")
else:
    print("No valid data points for evaluation.")
    
# Save predictions (optional)
# pipeline.save_predictions(task, pcode, test_pred)
