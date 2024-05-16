#%%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from utils import EncodedDataset, PositionalEncoder

from typing import Optional

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if CUDA is available

class TrainConfig():
    def __init__(
            self, 
            num_epochs: int,
            early_stop: bool,
            violation_limit: int,
            optimizer_params: Optional[dict] = None,
        ) -> None:
        self.optimizer_params   = optimizer_params
        self.num_epochs         = num_epochs
        self.violation_limit    = violation_limit
        self.early_stop        = early_stop
        

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size = 64, device = 'cpu'):
        assert device in ['cpu', 'cuda'], "device must be 'cpu' or 'cuda'"

        super().__init__()

        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        if device == 'cuda':
            if torch.cuda.is_available():
                self.to('cuda')
                self.device = 'cuda'
            else:
                print("CUDA not available. Run model on CPU")
                self.device = 'cpu'
        else:
            self.device = 'cpu'

    def forward(self, x):
        # Define the forward pass of the neural network
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def fit(
        self,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader,
        train_config: TrainConfig,
    ):
        """Train a neural network model

        Parameters
        ----------
        train_data : EncodedDataset
        dev_data : EncodedDataset
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
        best_val_loss = float('inf')
        violation_counter = 0
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adjust learning rate

        for epoch in range(train_config.num_epochs):
            self.train()
            
            with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", unit="batch") as train_batches:
                for X_train, y_train in train_batches:
                    X_train = X_train.to(self.device)
                    y_train = y_train.to(self.device)
                    optimizer.zero_grad()
                    outputs = self(X_train)
                    loss = loss_function(outputs, y_train)
                    loss.backward()
                    optimizer.step()

            # Evaluate on validation set for early stopping
            self.eval()
            X_dev = torch.concat([tup[0] for tup in list(dev_dataloader)]).to(self.device)
            y_dev = torch.concat([tup[1] for tup in list(dev_dataloader)]).to(self.device)
            output = self(X_dev)
            val_loss = loss_function(output, y_dev).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                violation_counter = 0
            else:
                violation_counter += 1

            if train_config.early_stops:
                if violation_counter >= train_config.violation_limit:
                    print(f"Validation loss did not improve for {train_config.violation_limit} epochs. Stopping early.")
                    break
            

def evaluate_nn_model(model: NeuralNetwork, test_dataset: EncodedDataset):
    with torch.no_grad():
        # X_test = torch.stack([dta[0] for dta in test_dataset])
        X_test = torch.stack([test[0] for test in test_dataset]).to(model.device)
        y_test = torch.stack([test[1] for test in test_dataset]).to(model.device)
        y_out = model(X_test)
        y_pred = y_out.argmax(dim=1)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    return precision, recall, f1


class RNNClassifier(nn.Module):
    def __init__(
        self,
        encoder: PositionalEncoder,
        rnn_network: nn.Module = nn.LSTM,
        word_embedding_dim: int = 32,
        class_num: int = 2,
        hidden_dim: int = 64,
        bidirectional: bool = False,
        dropout: float = 0.0,
        device: str = 'cpu'
    ):
        """An RNN based classifier

        Parameters
        ----------
        encoder : PositionalEncoder
        rnn_network : nn.Module, optional
            The network type to be used, can be either nn.LSTM or nn.GRU. By default nn.LSTM
        word_embedding_dim : int, optional
            The dimensionality of the word embedding, by default 32
        hidden_dim : int, optional
            The dimensionality of the hidden state in the RNN, by default 64
        bidirectional : bool, optional
            Specify if the RNN is bi-directional or not, by default False
        dropout : float, optional
            Ratio of random weight drop-out while training, by default 0.0
        device : str, optional
            Device to train the model on, can be either 'cuda' or 'cpu'. By default 'cpu'
        """

        assert rnn_network in [nn.LSTM, nn.GRU], "rnn_network must be nn.LSTM or nn.GRU"
        assert device in ['cpu', 'cuda'], "device must be 'cpu' or 'cuda'"

        super().__init__()
        self.hidden_dim_        = hidden_dim
        self.vocabulary_size_   = len(encoder.vocabulary)
        self.class_num_         = class_num
        self.pad_token_idx_     = encoder.token_to_index('<PAD>')
        self.encoder_           = encoder

        if device == 'cuda':
            if torch.cuda.is_available():
                self.to('cuda')
                self.device = 'cuda'
            else:
                print("CUDA not available. Run model on CPU.")
                self.device = 'cpu'
        else:
            self.device = 'cpu'

        # Initiate the word embedder.
        # It is actually a nn.Linear module with a look up table to return the embedding
        # corresponding to the token's positional index
        self._get_word_embedding = nn.Embedding(
            num_embeddings=self.vocabulary_size_,
            embedding_dim=word_embedding_dim,
            padding_idx=self.pad_token_idx_
        ).to(self.device)

        # Initiate the network
        self._rnn_network = rnn_network(
            input_size=word_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        ).to(self.device)

        # Initiate a linear layer to transform output of _rnn_network to the class space
        # Direction: 1 if uni-directional, 2 if bi-directional
        # This is a binary classification, so only need 1 output unit
        directions = bidirectional + 1
        self._fc = nn.Linear(hidden_dim * directions, 1).to(self.device)

        # Sigmoid to convert output to probability between 0 and 1
        self._sigmoid = nn.Sigmoid()

        # Store loss and accuracy to plot
        self.training_loss_ = list()
        self.training_accuracy_ = list()


    def forward(self, padded_sentences):
        """The forward pass through the network"""
        batch_size, max_sentence_length = padded_sentences.size()
        embedded_sentences = self._get_word_embedding(padded_sentences)

        # Prepare a PackedSequence object, and pass data through the RNN
        # TODO: Why do we need to pack the data this way?
        sentence_lengths = (padded_sentences != self.pad_token_idx_).sum(dim=1)
        sentence_lengths = sentence_lengths.long().cpu()

        packed_input = nn.utils.rnn.pack_padded_sequence(
            input=embedded_sentences,
            lengths=sentence_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        rnn_output, _ = self._rnn_network(packed_input)  # Returned another PackedSequence

        # Unpack the PackedSequence
        unpacked_sequence, _ = nn.utils.rnn.pad_packed_sequence(sequence=rnn_output, batch_first=True)
        unpacked_sequence = unpacked_sequence.contiguous().view(-1, unpacked_sequence.shape[2])

        # Pass data through the fully-connected linear layer
        class_space = self._fc(unpacked_sequence)

        # Reshape data Example size: (64, 2000, 1)
        reshaped = class_space.view(batch_size, max_sentence_length, 1)

        # With RNN, need to collapse the soft prediction (logit) into a one-dimension vector
        # TODO: Ask Fredrik about how to collapse the prediction
        collapsed = torch.stack([reshaped[i, j-1] for i, j in enumerate(sentence_lengths)]).squeeze()

        # sigmoid applied to convert to value between 0 - 1
        # Use sigmoid as output for nn.CrossEntropyLoss()
        scores = self._sigmoid(collapsed)

        return scores.to(self.device)


    def fit(self, train_dataloader: DataLoader, train_config: TrainConfig, no_progress_bar: bool = True) -> None:
        """Training loop for the RNN model. The loop will modify the model itself and returns nothing

        Parameters
        ----------
        train_dataloader : DataLoader
        train_config: TrainConfig
            An object containing various configs for the training loop
        train_encoder : PositionalEncoder
            The encoder providing the vocabulary and tagset for an internal batch_encoder
        Returns
        -------
        """
        # Make sure that the training process do not modify the initial model

        best_lost = float('inf')
        violations = 0
        optimizer = torch.optim.Adam(self.parameters(), **train_config.optimizer_params)
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(train_config.num_epochs):
            with tqdm(
                train_dataloader,
                total   = len(train_dataloader),
                unit    = "batch",
                desc    = f"Epoch {epoch + 1}",
                disable = no_progress_bar,
            ) as batches:

                for ids, speakers, raw_inputs, raw_targets in batches:

                    # Initiate a batch-specific encoder that inherits the vocabulary from the pre-trained encoder
                    # to transform data in the batch.
                    batch_encoder = PositionalEncoder(vocabulary=self.encoder_.vocabulary,)

                    # max_sentence_length_ of each batch are allowed to be varied since it is learned here -> more memory-efficient
                    #
                    train_inputs = batch_encoder.fit_transform(raw_inputs).to(self.device)
                    train_targets = torch.as_tensor(raw_targets, dtype=torch.float).to(self.device)  # nn.CrossEntropyLoss() require target to be float

                    # Reset gradients, then run forward pass
                    self.zero_grad()
                    scores = self(train_inputs)

                    # Calc loss
                    loss = loss_function(scores.view(-1), train_targets.view(-1))

                    # Backward propagation. After each iteration through the batches,
                    # accumulate the gradient for each theta
                    # Run the optimizer to update the parameters
                    loss.backward()
                    optimizer.step()

                    # Evaluate with training batch accuracy
                    pred = scores >= 0.5
                    correct = (pred * 1.0 == train_targets).sum().item()
                    accuracy = correct / len(train_targets)

                    # Save accuracy and loss for plotting
                    self.training_accuracy_.append(accuracy)
                    self.training_loss_.append(loss.item())

                    # Add loss and accuracy info to tqdm's progress bar
                    batches.set_postfix(loss=loss.item(), batch_accuracy=accuracy)

                    # Early stop:
                    if train_config.early_stop:
                        if loss < best_lost:
                            best_lost = loss
                            violations = 0
                        else:
                            violations += 1

                        if violations == train_config.violation_limit:
                            print(f"No improvement for {train_config.violation_limit} epochs. Stop early.")
                            break


