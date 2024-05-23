#%%
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

import numpy as np
import pandas as pd
import altair as alt

from utils import EncodedDataset, PositionalEncoder

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

class SVMClassifier():
    def __init__(self) -> None:
        pass

class SVClassifier():
    def __init__(self) -> None:
        pass



class NeuralNetwork(nn.Module):
    def __init__(
            self, 
            input_size, 
            hidden_size = 64, 
            output_size = 1,  # binary classification only need 1 output
            positive_pred_threshold: float = 0.5,
            # class_weights: Optional[torch.Tensor] = torch.Tensor([1.0, 1.0]),
            pos_weight: float = 1.0,
            device = 'cpu',
        ):
        assert device in ['cpu', 'cuda'], "device must be 'cpu' or 'cuda'"
        super().__init__()

        # Define the layers of the neural network

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

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
        self.positive_perd_threshold = positive_pred_threshold
        # self.class_weights = class_weights.to(self.device)
        self.pos_weight = torch.Tensor([pos_weight]).to(self.device)

        self.training_accuracy_ = []
        self.training_loss_ = []

    def forward(self, x: torch.Tensor):
        # Define the forward pass of the neural network
        logits = self.network(x.to(self.device)).squeeze()
        return logits

    def predict(self, x: torch.Tensor):
        logits = self.forward(x.to(self.device))
        pred = (self.sigmoid(logits) >= self.positive_perd_threshold).squeeze() * 1.0  # Convert to 1-0
        return pred


    def fit(
        self,
        train_dataloader: DataLoader,
        train_config: TrainConfig,
        disable_progress_bar: bool = True
    ):
        """Train a neural network model

        Parameters
        ----------
        train_data : EncodedDataset
        dev_data : EncodedDataset
        output_size : int, optional
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
        # Get the last token in each sentence as the output of RNN
        collapsed_output = torch.stack([reshaped[i, j-1] for i, j in enumerate(sentence_lengths)]).squeeze().to(self.device)

        # sigmoid applied to convert to value between 0 - 1
        # Use sigmoid as output for nn.CrossEntropyLoss()
        # scores = self._sigmoid(collapsed)

        return collapsed_output
    

    def predict(self, x: torch.Tensor):
        """Make predictions for the input tensor"""
        logits = self.forward(x.to(self.device))
        pred = (self._sigmoid(logits) >= 0.5).squeeze() * 1.0  # Convert to 1-0
        return pred


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


def save_model(model: NeuralNetwork | RNNClassifier, model_name: str):
    """Save model state to disc"""
    torch.save(model.state_dict(), f"models/{model_name}.pt")
    np.save(f"models/{model_name}_training_accuracy_.npy", model.training_accuracy_)
    np.save(f"models/{model_name}_training_loss_.npy", model.training_loss_)


def load_model(model: NeuralNetwork | RNNClassifier, model_name: str):
    """Pass in an initiated model and load the saved state"""
    model.load_state_dict(torch.load(f"models/{model_name}.pt"))
    model.training_accuracy_ = np.load(f"models/{model_name}_training_accuracy_.npy")
    model.training_loss_ = np.load(f"models/{model_name}_training_loss_.npy")
    return model



def plot_results(model: NeuralNetwork | RNNClassifier, train_config: TrainConfig, train_dataloader: DataLoader):
    # Plot training accuracy and loss side-by-side
    epochs_arr = []
    for epoch in range(1, train_config.num_epochs + 1):
        epochs_arr += [epoch] * len(train_dataloader)

    result_df = pd.DataFrame({
        'training_acc': model.training_accuracy_,
        'training_loss': model.training_loss_,
        'iteration': range(1, len(model.training_accuracy_) + 1),
        'epoch': epochs_arr
    })

    nn_training_accuracy_chart = alt.Chart(result_df).mark_line().encode(
        x = alt.X("iteration:N", axis = alt.Axis(title = 'Iteration', values=list(range(0, len(model.training_accuracy_), 100)), labelAngle=-30)),
        y = alt.Y("training_acc:Q", axis = alt.Axis(title = 'Accuracy')),
        color = alt.Color("epoch:N", scale=alt.Scale(scheme='category20'), title='Epoch'),
    ).properties(
        width=600,
        height=400,
        title = 'Training Accuracy'
    )

    nn_training_loss_chart = alt.Chart(result_df).mark_line().encode(
        x = alt.X("iteration:N", axis = alt.Axis(title = 'Iteration', values=list(range(0, len(model.training_accuracy_), 100)), labelAngle=-30)),
        y = alt.Y("training_loss:Q", axis = alt.Axis(title = 'Loss')),
        color = alt.Color("epoch:N", scale=alt.Scale(scheme='category20'), title='Epoch'),
    ).properties(
        width=600,
        height=400,
        title = 'Training Loss'
    )

    (nn_training_accuracy_chart | nn_training_loss_chart).properties(
        title = 'Base Neural Network with Tf-Idf vectors'
    ).show()
    

def evaluate_nn_model(model: NeuralNetwork, test_dataset: EncodedDataset):
    with torch.no_grad():
        # X_test = torch.stack([dta[0] for dta in test_dataset])
        X_test = torch.stack([test[0] for test in test_dataset]).to(model.device)
        y_test = torch.stack([test[1] for test in test_dataset]).to(model.device)
        y_pred = model.predict(X_test)

        true_pos = sum([pred == y == 1 for pred, y in zip(y_pred, y_test)])
        true_neg = sum([pred == y == 0 for pred, y in zip(y_pred, y_test)])
        false_pos = sum([(pred == 1) * (y == 0) for pred, y in zip(y_pred, y_test)])
        false_neg = sum([(pred == 0) * (y == 1) for pred, y in zip(y_pred, y_test)])
        total = len(y_test)
    
    precision = (true_pos + true_neg) / total
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg)
    kappa = cohen_kappa_score(y_pred.cpu(), y_test.cpu(), labels=[0, 1], weights='linear')

    return precision.item(), recall.item(), f1.item(), kappa


def evaluate_rnn_model(
        model: nn.Module | RNNClassifier,
        test_dataloader,
        train_encoder
    ) -> float:
    """Evaluate the model on an inputs-targets set, using accuracy metric.

    Parameters
    ----------
    model : nn.Module
        Should be one of the two custom RNN taggers we defined.
    inputs : torch.Tensor
    targets : torch.Tensor
    pad_tag_idx : int
        Index of the <PAD> tag in the tagset to be ignored when calculating accuracy

    Returns
    -------
    float
        Accuracy metric (ignored the <PAD> tag)
    """
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    total = 0

    for ids, speakers, raw_inputs, raw_targets in tqdm(test_dataloader, unit="batch", desc="Predicting"):

        batch_encoder = PositionalEncoder(vocabulary=train_encoder.vocabulary)
        inputs = batch_encoder.fit_transform(raw_inputs)
        targets = torch.as_tensor(raw_targets, dtype=torch.float).to(model.device)  # nn.CrossEntropyLoss() require target to be float

        # Make prediction
        pred = model.predict(inputs.to(model.device))
        true_pos += sum([pred == y == 1 for pred, y in zip(pred, targets)])
        true_neg += sum([pred == y == 0 for pred, y in zip(pred, targets)])
        false_pos += sum([(pred == 1) * (y == 0) for pred, y in zip(pred, targets)])
        false_neg += sum([(pred == 0) * (y == 1) for pred, y in zip(pred, targets)])
        total += len(targets)
    

    precision = (true_pos + true_neg) / total
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg)

    return precision.item(), recall.item(), f1.item()
# %%
