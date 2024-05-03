import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from utils import get_data  # Assuming this imports the get_data function from some module

from models import NeuralNetwork

# Define a pipeline class for our task
class IdeologyPowerDetectionPipeline:
    def __init__(
            self, 
            vectorizer,
            model,
            data_dir='data', 
            models_dir='models', 
            pred_dir='predictions', 
            num_epochs=20, 
            early_stop_patience=5,
        ):
        # Initialize the pipeline with default directories and parameters
        self.data_dir = data_dir
        self.num_epochs = num_epochs
        self.models_dir = models_dir
        self.pred_dir = pred_dir
        self.vectorizer = vectorizer
        self.model = model
        self.early_stop_patience = early_stop_patience
        self.best_val_loss = float('inf')  # Set initial best validation loss to infinity
        self.patience_counter = 0  # Counter for early stopping
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if CUDA is available

    def load_data(self, task, pcode):
        # Load data from a TSV file using the get_data function
        data_file = os.path.join(self.data_dir, task, f"{task}-{pcode}-train.tsv")
        t_trn, y_trn, t_val, y_val = get_data(data_file)
        return t_trn, y_trn, t_val, y_val

    def preprocess_data(self, texts):
        # Preprocess text data using TF-IDF vectorization
        if self.vectorizer is None:
            self.vec = TfidfVectorizer(sublinear_tf=True, analyzer="char", ngram_range=(1,3))
            x_trn = self.vectorizer.fit_transform(texts)
        else:
            x_trn = self.vectorizer.transform(texts)
        return x_trn
    
    def train_model(self, train_func, *args, **kwargs):
        train_func()

    def train_model(self, x_trn, y_trn, x_val, y_val):
        # Train a neural network model using PyTorch
        input_size = x_trn.shape[1]
        num_classes = len(torch.unique(torch.tensor(y_trn)))
        hidden_size = 64  # Adjust as needed
        
        model = NeuralNetwork(input_size, hidden_size, num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate
        
        # Convert data to PyTorch tensors and move to GPU
        x_trn_tensor = torch.tensor(x_trn.toarray(), dtype=torch.float32).to(self.device)
        y_trn_tensor = torch.tensor(y_trn, dtype=torch.long).to(self.device)
        x_val_tensor = torch.tensor(x_val.toarray(), dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
        
        for epoch in range(self.num_epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(x_trn_tensor)
            loss = criterion(outputs, y_trn_tensor)
            loss.backward()
            optimizer.step()

            # Evaluate on validation set for early stopping
            self.model.eval()
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stop_patience:
                print(f"Validation loss did not improve for {self.early_stop_patience} epochs. Stopping early.")
                break

        self.model = model

    def evaluate_model(self, x_val, y_val):
        # Evaluate the trainde model on validation data
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        with torch.no_grad():
            # Move tensors to the same device as the model
            device = next(self.model.parameters()).device
    
            x_val_tensor = torch.tensor(x_val.toarray(), dtype=torch.float32, device=device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
            outputs = self.model(x_val_tensor)
            _, predicted = torch.max(outputs, 1)
            
            # Convert tensors to numpy arrays
            predicted = predicted.cpu().numpy()
            y_val = y_val_tensor.cpu().numpy()
            
            # Compute precision, recall, and F1-score
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, predicted, average='macro')
        
        return precision, recall, f1

    def save_model(self, task, pcode):
        # Save the trained model to a file
        os.makedirs(self.models_dir, exist_ok=True)
        model_file = os.path.join(self.models_dir, f"{task}-{pcode}.pt")
        torch.save(self.model.state_dict(), model_file)
        
    def save_predictions(self, task, pcode, test_pred):
        # Save predictions to a file
        os.makedirs(self.pred_dir, exist_ok=True)
        pred_file = os.path.join(self.pred_dir, f"{task}-{pcode}-predictions.tsv")
        with open(pred_file, "wt") as fpred:
            for i, p in enumerate(test_pred):
                print(f"{i+1}\t{p}", file=fpred)

pipeline = IdeologyPowerDetectionPipeline()

# Example usage for proof of concept
task = 'orientation'
pcode = 'gb'

# Load data
t_trn, y_trn, t_val, y_val = pipeline.load_data(task, pcode)
    
# Preprocess data
x_trn = pipeline.preprocess_data(t_trn)
x_val = pipeline.preprocess_data(t_val)
    
# Train model
pipeline.train_model(x_trn, y_trn, x_val, y_val)
    
# Evaluate model
precision, recall, fscore = pipeline.evaluate_model(x_val, y_val)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {fscore:.4f}")

# Save model (optional)
# pipeline.save_model(task, pcode)

# Save predictions (optional)
# pipeline.save_predictions(task, pcode, test_pred)