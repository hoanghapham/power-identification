#%%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from utils import CustomDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if CUDA is available


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Define the forward pass of the neural network
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def train_neural_network(
    train_data: CustomDataset, 
    dev_data: CustomDataset,
    num_classes: int = 2, 
    hidden_size: int = 64, 
    batch_size: int = 64,
    num_epochs: int = 20,
    early_stop_patience: int = 5
):
    input_size = len(train_data[0][0])
    best_val_loss = float('inf')
    patience_counter = 0

    model = NeuralNetwork(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as train_batches:
            for X_train, y_train in train_batches:
                model(X_train)
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

        # Evaluate on validation set for early stopping
        model.eval()
        X_dev = torch.stack([dta[0] for dta in dev_data])
        y_dev = torch.as_tensor([dev[1] for dev in dev_data], dtype=torch.long)
        output = model(X_dev)
        val_loss = criterion(output, y_dev).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Validation loss did not improve for {early_stop_patience} epochs. Stopping early.")
            break
    
    return model


def evaluate_model(model, test_dataset: CustomDataset):
    with torch.no_grad():
        X_test = torch.stack([dta[0] for dta in test_dataset])
        y_test = torch.as_tensor([test[1] for test in test_dataset])
        y_out = model(X_test)
        y_pred = y_out.argmax(dim=1)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    return precision, recall, f1