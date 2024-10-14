from logging import Logger

import altair as alt
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import xgboost as xgb

from lib.data_processing import RawDataset, PositionalEncoder, encode_torch_data
from lib.models import RNNClassifier, NeuralNetwork, TrainConfig



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

    # print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
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



# Helper functions
def get_average_metrics(result_list: list[dict]) -> dict:
    accuracy = np.mean([[result['accuracy'] for result in result_list]])
    precision = np.mean([[result['precision'] for result in result_list]])
    recall = np.mean([[result['recall'] for result in result_list]])
    f1 = np.mean([[result['f1'] for result in result_list]])
    auc = np.mean([[result['auc'] for result in result_list]])
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def train_evaluate_sklearn(model: BaseEstimator, data: RawDataset, nfolds: int, feature_type: str, logger: Logger):
    result_list = []
    kfold = KFold(n_splits=nfolds, shuffle=False, random_state=None)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(data), start=1):
        logger.info(f"CV fold {fold_idx}")

        # Prepare encoders
        if feature_type == "word": 
            encoder = TfidfVectorizer(max_features=50000)
        elif feature_type == "char":
            encoder = TfidfVectorizer(max_features=50000, analyzer="char", ngram_range=(3,5), use_idf=True, sublinear_tf=True)
        else:
            raise ValueError(f"Not supported feature type: {feature_type}")
        
        encoder.fit(data.subset(train_idx).texts)

        X_train = encoder.transform(data.subset(train_idx).texts)
        X_test = encoder.transform(data.subset(test_idx).texts)
        label_train = data.subset(train_idx).labels
        label_test = data.subset(test_idx).labels

        # Fit model
        model.fit(X_train, label_train)

        # Predict & Evaluate
        pred = model.predict(X_test)
        probs = model.predict_proba(X_test)

        result = evaluate(label_test, pred, probs[:, 1])
        result_list.append({"fold": str(fold_idx), **result})

    return model, result_list


def train_evaluate_xgboost(data: RawDataset, nfolds: int, feature_type: str, logger: Logger, device: str = "cpu"):
    result_list = []
    kfold = KFold(n_splits=nfolds, shuffle=False, random_state=None)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(data), start=1):
        logger.info(f"CV fold {fold_idx}")

        # Prepare encoders
        if feature_type == "word": 
            encoder = TfidfVectorizer(max_features=50000)
        elif feature_type == "char":
            encoder = TfidfVectorizer(max_features=50000, analyzer="char", ngram_range=(3,5), use_idf=True, sublinear_tf=True)
        else:
            raise ValueError(f"Not supported feature type: {feature_type}")
        
        encoder.fit(data.subset(train_idx).texts)

        X_train = encoder.transform(data.subset(train_idx).texts)
        X_test = encoder.transform(data.subset(test_idx).texts)
        label_train = data.subset(train_idx).labels
        label_test = data.subset(test_idx).labels

        train_dmat = xgb.DMatrix(X_train, pd.array(label_train).astype("category"))
        test_dmat = xgb.DMatrix(X_test, pd.array(label_test).astype("category"))

        params = {
            "booster": "gbtree",
            "objective": "binary:logistic",  # there is also binary:hinge but hinge does not output probability
            "tree_method": "hist",  # default to hist
            "device": device,

            # Params for tree booster
            "eta": 0.3,
            "gamma": 0.0,  # Min loss achieved to split the tree
            "max_depth": 6,
            "reg_alpha": 0,
            "reg_lambda": 1,
        }

        # Train
        ITERATIONS = 2000

        evals = [(train_dmat, "train")]

        model = xgb.train(
            params = params,
            dtrain = train_dmat,
            num_boost_round = ITERATIONS,
            evals = evals,
            verbose_eval = False
        )

        # Evaluate
        probs = model.predict(test_dmat)
        result = evaluate(label_test, probs > 0.5, probs)
        result_list.append({"fold": str(fold_idx), **result})

    return model, result_list


def train_evaluate_nn(data: RawDataset, nfolds: int, feature_type: str, logger: Logger, device: str):
    result_list = []
    kfold = KFold(n_splits=nfolds, shuffle=False, random_state=None)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(data), start=1):
        logger.info(f"CV fold {fold_idx}")

        # Prepare encoders
        if feature_type == "word": 
            encoder = TfidfVectorizer(max_features=50000)
        elif feature_type == "char":
            encoder = TfidfVectorizer(max_features=50000, analyzer="char", ngram_range=(3,5), use_idf=True, sublinear_tf=True)
        else:
            raise ValueError(f"Not supported feature type: {feature_type}")
        
        encoder.fit(data.subset(train_idx).texts)
        
        # Encode data
        train_data = encode_torch_data(data.subset(train_idx), encoder)
        test_data = encode_torch_data(data.subset(test_idx), encoder)

        # Init model
        train_config = TrainConfig(
            num_epochs      = 10,
            early_stop      = False,
            violation_limit = 5
        )

        dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

        model = NeuralNetwork(
            input_size=len(encoder.vocabulary_),
            hidden_size=128,
            device=device
        )

        # Train
        model.fit(dataloader, train_config, disable_progress_bar=True)

        # Evaluate
        with torch.no_grad():
            X_test_nn = torch.stack([test[0] for test in test_data]).cpu()
            y_test_nn = torch.stack([test[1] for test in test_data]).cpu()
            y_pred_nn = model.predict(X_test_nn)
            logits_nn = model.forward(X_test_nn)

        result = evaluate(y_test_nn.cpu(), y_pred_nn.cpu(), logits_nn.cpu())
        result_list.append({"fold": str(fold_idx), **result})

    return model, result_list


def train_evaluate_rnn(data: RawDataset, nfolds: int, feature_type: str, logger: Logger, device: str):

    result_list = []
    kfold = KFold(n_splits=nfolds, shuffle=False)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(data), start=1):
        logger.info(f"CV fold {fold_idx}")

        if feature_type == "word":
            encoder = PositionalEncoder()
        elif feature_type == "char":
            chars_encoder = TfidfVectorizer(max_features=50000, analyzer="char", ngram_range=(3,5), use_idf=True, sublinear_tf=True)
            encoder = PositionalEncoder(tokenizer=chars_encoder.build_tokenizer())

        encoder.fit(data.subset(train_idx).texts)

        train_dataloader = DataLoader(data.subset(train_idx), batch_size=128, shuffle=True)
        test_dataloader = DataLoader(data.subset(test_idx), batch_size=128, shuffle=False)

        # Prepare baseline config
        train_config = TrainConfig(
            optimizer_params = {'lr': 0.01},
            num_epochs       = 10,
            early_stop       = False,
            violation_limit  = 5
        )

        # Train baseline model
        model = RNNClassifier(
            rnn_network         = nn.LSTM,
            word_embedding_dim  = 32,
            hidden_dim          = 64,
            bidirectional       = False,
            dropout             = 0,
            encoder             = encoder,
            device              = device
        )


        model.fit(train_dataloader, train_config, no_progress_bar=True)

        # Evaluate
        with torch.no_grad():
            model.device = "cpu"
            model.cpu()

            pred_lst = []
            probs_lst = []

            for _, _, raw_inputs, raw_targets in test_dataloader:
                batch_encoder = PositionalEncoder(vocabulary=encoder.vocabulary)
                test_inputs = batch_encoder.fit_transform(raw_inputs).cpu()
                test_targets = torch.as_tensor(raw_targets, dtype=torch.float).cpu()
                
                pred_lst.append(model.predict(test_inputs))
                probs_lst.append(model._sigmoid(model.forward(test_inputs)).squeeze())

        pred = torch.cat(pred_lst).long().numpy()
        probs = torch.concat(probs_lst).numpy()

        result = evaluate(data.subset(test_idx).labels, pred, probs)
        result_list.append({"fold": str(fold_idx), **result})

    return model, result_list