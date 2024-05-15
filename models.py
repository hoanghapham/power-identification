#%%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from utils import ParliamentDataset, RawParliamentData, encode_text


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if CUDA is available

class TrainConfig():
    def __init__(self, batch_size: int, num_epochs: int, early_stop: bool, violation_limit: int) -> None:
        self.batch_size         = batch_size
        self.num_epochs         = num_epochs
        self.early_stops        = early_stop
        self.violation_limit    = violation_limit
        

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size = 64, device = 'cpu'):
        super(NeuralNetwork, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self, x):
        # Define the forward pass of the neural network
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


    def fit(
        self,
        train_dataset: ParliamentDataset,
        dev_dataset: ParliamentDataset,
        train_config: TrainConfig,
    ):
        """Train a neural network model

        Parameters
        ----------
        train_data : ParliamentDataset
        dev_data : ParliamentDataset
        num_classes : int, optional
            Number of classes to be predicted, by default 2
        hidden_size : int, optional
            Number of hidden nodes, by default 64
        batch_size : int, optional
            Number of samples in a data batch, by default 64
        num_epochs : int, optional
            Number of epochs to train, by default 20
        train_config.violation_limit : int, optional
            Number of epochs to wait when the loss cannot be reduced further, by default 5
        device: str, optional
            Can be either 'cpu' or 'cuda'. Only use `cuda` if your machine has a graphic card supporting CUDA.
        Returns
        -------
        NeuralNetwork
        """

        # input_size = len(train_data[0][0])
        best_val_loss = float('inf')
        violation_counter = 0

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adjust learning rate

        train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)

        for epoch in range(train_config.num_epochs):
            self.train()
            
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch") as train_batches:
                for X_train, y_train in train_batches:
                    X_train = X_train.to(self.device)
                    y_train = y_train.to(self.device)
                    optimizer.zero_grad()
                    outputs = self(X_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()

            # Evaluate on validation set for early stopping
            self.eval()
            X_dev = torch.stack([dev[0] for dev in dev_dataset]).to(self.device)
            y_dev = torch.stack([dev[1] for dev in dev_dataset]).to(self.device)
            output = self(X_dev)
            val_loss = criterion(output, y_dev).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                violation_counter = 0
            else:
                violation_counter += 1

            if train_config.early_stops:
                if violation_counter >= train_config.violation_limit:
                    print(f"Validation loss did not improve for {train_config.violation_limit} epochs. Stopping early.")
                    break
            

def evaluate_model(model: NeuralNetwork, test_dataset: ParliamentDataset):
    if next(model.parameters()).device.type == 'cuda':
        model = model.cpu()
    with torch.no_grad():
        # X_test = torch.stack([dta[0] for dta in test_dataset])
        X_test = torch.stack([test[0] for test in test_dataset])
        y_test = torch.stack([test[1] for test in test_dataset])
        y_out = model(X_test).cpu()
        y_pred = y_out.argmax(dim=1)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    return precision, recall, f1
