#%%
from typing import Optional
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

import numpy as np

from lib.data_processing import PositionalEncoder


class TrainConfig():

    """Convenience object to store training configurations.
    """
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
    """Simple Network model built using torch's modules
    """
    def __init__(
            self, 
            input_size, 
            hidden_size = 64, 
            n_linear_layers: int = 1,
            output_size = 1,  # binary classification only need 1 output
            positive_pred_threshold: float = 0.5,
            # class_weights: Optional[torch.Tensor] = torch.Tensor([1.0, 1.0]),
            pos_weight: float = 1.0,
            dropout: float = 0,
            device = 'cpu',
        ):
        assert device in ['cpu', 'cuda'], "device must be 'cpu' or 'cuda'"
        super().__init__()

        # Define the layers of the neural network

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # Hidden layers can be dynamically generated
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout), nn.ReLU()) 
            for i in range(n_linear_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_size, output_size)

        if device == 'cuda':
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.cuda()
            else:
                print("CUDA not available. Run model on CPU")
                self.device = 'cpu'
                self.cpu()
        else:
            self.device = 'cpu'
            self.cpu()

        self.sigmoid = nn.Sigmoid()
        self.positive_pred_threshold = positive_pred_threshold
        # self.class_weights = class_weights.to(self.device)
        self.pos_weight = torch.Tensor([pos_weight]).to(self.device)

        self.training_accuracy_: list = []
        self.training_loss_: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network. Return the logits

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        # Define the forward pass of the neural network
        activation = self.input_layer(x.to(self.device))

        for layer in self.hidden_layers:
            activation = layer(activation)
        
        logits = self.output_layer(activation).squeeze()

        return logits

    def predict(self, x: torch.Tensor, positive_pred_threshold: float = None) -> torch.Tensor:
        """Output prediction labels (1, 0)

        Parameters
        ----------
        x : torch.Tensor
        positive_pred_threshold : float, optional
            If the result of sigmoid(logits) is greater than the threshold, then the prediction is 1.

        Returns
        -------
        torch.Tensor
            A tensor of labels (0, 1)
        """
        if not positive_pred_threshold:
            positive_pred_threshold = self.positive_pred_threshold
        logits = self.forward(x.to(self.device))
        pred = (self.sigmoid(logits) >= positive_pred_threshold).squeeze() * 1.0  # Convert to 1-0 labels
        return pred

    def fit(
        self,
        train_dataloader: DataLoader,
        train_config: TrainConfig,
        disable_progress_bar: bool = True
    ) -> None:
        """Train the model

        Parameters
        ----------
        train_dataloader : DataLoader
        train_config : TrainConfig
        disable_progress_bar : bool, optional
            If True, disable the progress bar, by default True
        """
        best_loss = float('inf')
        violations = 0
        loss_function = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        optimizer = optim.Adam(self.parameters(), lr=0.001)  # Adjust learning rate
        print()

        for epoch in range(train_config.num_epochs):
            self.train()
            
            with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", unit="batch", disable=disable_progress_bar) as train_batches:
                for X_train, y_train in train_batches:
                    X_train = X_train.to(self.device)
                    y_train = y_train.float().to(self.device)
                    optimizer.zero_grad()
                    logits = self(X_train)

                    # print(logits.shape, y_train.shape)
                    loss = loss_function(logits, y_train)
                    loss.backward()
                    optimizer.step()

                    # Evaluate on train set 
                    self.eval()
                    pred = self.predict(X_train)
                    corrects = (pred == y_train).sum().item()
                    accuracy = corrects / len(y_train)

                    self.training_accuracy_.append(accuracy)
                    self.training_loss_.append(loss.item())

                    train_batches.set_postfix(batch_accuracy=accuracy, loss=loss.item())

                    torch.cuda.empty_cache()
            
                    if loss < best_loss:
                        best_loss = loss
                        violations = 0
                    else:
                        violations += 1

                    if train_config.early_stop:
                        if violations >= train_config.violation_limit:
                            print(f"Validation loss did not improve for {train_config.violation_limit} iterations. Stopping early.")
                            break


class RNNClassifier(nn.Module):
    def __init__(
        self,
        encoder: PositionalEncoder,
        rnn_network: nn.Module = nn.LSTM,
        word_embedding_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 1,  # binary classification only need 1 output
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
        self.output_dim_        = output_dim
        self.pad_token_idx_     = encoder.token_to_index('<PAD>')
        self.encoder_           = encoder

        if device == 'cuda':
            if torch.cuda.is_available():
                self.to('cuda')
                self.device = 'cuda'
            else:
                print("CUDA not available. Run model on CPU.")
                self.device = 'cpu'
                self.to('cpu')
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


    def forward(self, padded_sentences: torch.Tensor) -> torch.Tensor:
        """The forward pass through the network"""
        batch_size, max_sentence_length = padded_sentences.size()
        embedded_sentences = self._get_word_embedding(padded_sentences)

        # Prepare a PackedSequence object, and pass data through the RNN
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
        # Get the last token in each sentence as the output of RNN
        collapsed_output = torch.stack([reshaped[i, j-1] for i, j in enumerate(sentence_lengths)]).squeeze().to(self.device)

        return collapsed_output
    

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions for the input tensor"""
        logits = self.forward(x.to(self.device))
        pred = (self._sigmoid(logits) >= 0.5).squeeze() * 1.0  # Convert to 1-0
        return pred

    def fit(
        self, 
        train_dataloader: DataLoader,
        train_config: TrainConfig,
        no_progress_bar: bool = True
    ) -> None:
        """Training loop for the RNN model. The loop will modify the model itself and returns nothing

        Parameters
        ----------
        train_dataloader : DataLoader
        train_config: TrainConfig
            An object containing various configs for the training loop
        train_encoder : PositionalEncoder
            The encoder providing the vocabulary and tagset for an internal batch_encoder
        """

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
                    logits = self(train_inputs)

                    # Calc loss
                    loss = loss_function(logits.view(-1), train_targets.view(-1))

                    # Backward propagation. After each iteration through the batches,
                    # accumulate the gradient for each theta
                    # Run the optimizer to update the parameters
                    loss.backward()
                    optimizer.step()

                    # Evaluate with training batch accuracy
                    pred = self.predict(train_inputs)
                    correct = (pred == train_targets).sum().item()
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


def save_model(model: NeuralNetwork | RNNClassifier, model_dir: str | Path, model_name: str) -> None:
    """Save model state to local storage. 

    The function will create three files:
    - model_name.pt
    - model_name_training_accuracy.npy
    - model_name_training_loss.npy

    Parameters
    ----------
    model : NeuralNetwork | RNNClassifier
    model_dir : str | Path
        Path to the directory to save the model
    model_name : str
        Name of the model, without any extension.
    """
    torch.save(model.state_dict(), f"{model_dir}/{model_name}.pt")
    np.save(f"{model_dir}/{model_name}_training_accuracy.npy", model.training_accuracy_)
    np.save(f"{model_dir}/{model_name}_training_loss.npy", model.training_loss_)


def load_model(model: NeuralNetwork | RNNClassifier, model_dir: str | Path, model_name: str) -> NeuralNetwork | RNNClassifier:
    """Pass in an initiated model and load the saved state

    The function will load the following files:
    - model_name.pt
    - model_name_training_accuracy.npy
    - model_name_training_loss.npy

    The model is loaded as an instance of NeuralNetwork or RNNClassifier.
    The training accuracy and training loss values are assigned to the model's attributes

    Parameters
    ----------
    model : NeuralNetwork | RNNClassifier
    model_dir : str | Path
        Path to the directory where the model was saved
    model_name : str
        Name of the model to load, without extension

    Returns
    -------
    NeuralNetwork | RNNClassifier
    """
    model.load_state_dict(torch.load(f"{model_dir}/{model_name}.pt"))
    model.training_accuracy_ = np.load(f"{model_dir}/{model_name}_training_accuracy.npy")
    model.training_loss_ = np.load(f"{model_dir}/{model_name}_training_loss.npy")
    return model


