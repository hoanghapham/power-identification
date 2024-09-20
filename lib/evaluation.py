import altair as alt
import pandas as pd
import numpy as np

import torch
from torch import nn

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .models import RNNClassifier, NeuralNetwork, TrainConfig
from .data_processing import PositionalEncoder

def plot_results(
        title: str, 
        model: NeuralNetwork | RNNClassifier, 
        train_config: TrainConfig, 
        train_dataloader: DataLoader
    ) -> None:
    """Plot training accuracy and loss of Neural Network models side-by-side 

    Parameters
    ----------
    title : str
        Title of the plot
    model : NeuralNetwork | RNNClassifier
    train_config : TrainConfig
    train_dataloader : DataLoader
    """
    epochs_arr = []
    for epoch in range(1, train_config.num_epochs + 1):
        epochs_arr += [epoch] * len(train_dataloader)

    result_df = pd.DataFrame({
        'training_acc': model.training_accuracy_,
        'training_loss': model.training_loss_,
        'iteration': range(1, len(model.training_accuracy_) + 1),
        'epoch': epochs_arr
    })

    training_accuracy_chart = alt.Chart(result_df).mark_line().encode(
        x = alt.X("iteration:N", axis = alt.Axis(title = 'Iteration', values=list(range(0, len(model.training_accuracy_), 100)), labelAngle=-30)),
        y = alt.Y("training_acc:Q", axis = alt.Axis(title = 'Accuracy')),
        color = alt.Color("epoch:N", scale=alt.Scale(scheme='category20'), title='Epoch'),
    ).properties(
        width=600,
        height=400,
        title = 'Training Accuracy'
    )

    training_loss_chart = alt.Chart(result_df).mark_line().encode(
        x = alt.X("iteration:N", axis = alt.Axis(title = 'Iteration', values=list(range(0, len(model.training_accuracy_), 100)), labelAngle=-30)),
        y = alt.Y("training_loss:Q", axis = alt.Axis(title = 'Loss')),
        color = alt.Color("epoch:N", scale=alt.Scale(scheme='category20'), title='Epoch'),
    ).properties(
        width=600,
        height=400,
        title = 'Training Loss'
    )

    (training_accuracy_chart | training_loss_chart).properties(title = title).show()
    

def evaluate(y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Conveninece function to evaluate predction of models.

    The function returns a dictionary of metrics:
    - Accuracy
    - Precision
    - Recall
    - F1
    - AUC

    Parameters
    ----------
    y_test : np.ndarray
        Labels of the test set
    y_pred : np.ndarray
        Prediction produced by the model
    y_prob : np.ndarray
        Probability array produced by the model

    Returns
    -------
    dict
    """

    true_pos = sum([pred == y == 1 for pred, y in zip(y_pred, y_test)])
    true_neg = sum([pred == y == 0 for pred, y in zip(y_pred, y_test)])
    false_pos = sum([(pred == 1) * (y == 0) for pred, y in zip(y_pred, y_test)])
    false_neg = sum([(pred == 0) * (y == 1) for pred, y in zip(y_pred, y_test)])
    total = len(y_test)

    accuracy = (true_pos + true_neg) / total
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg)
    auc = roc_auc_score(y_test, y_prob)

    if isinstance(accuracy, torch.Tensor):
        result = {
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "auc": auc.item(),
        }    
    else:

        result = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    return result


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

    with torch.no_grad():

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
        
        accuracy = (true_pos + true_neg) / total
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg)

    return accuracy.item(), precision.item(), recall.item(), f1.item()
