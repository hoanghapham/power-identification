#!/usr/bin/env python3
""" Read and return data for shared task on Ideology and Power Detection.

See shared task web page <https://touche.webis.de/clef24/touche24-web/ideology-and-power-identification-in-parliamentary-debates.html>
for details.
"""
# import sys
import csv
# csv.field_size_limit(sys.maxsize)
from sklearn.model_selection import train_test_split, KFold

def get_data_kfold(fname, encoding='utf-8', num_splits=5, text_head='text', seed=None):
    """Read the data and perform k-fold cross-validation.

    Parameters:
    fname       The filename to read data from.
    encoding    The encoding of the file.
    num_splits  Number of folds for K-fold cross-validation.
    text_head   The header of the text field,
                useful for reading the English translations.
    seed        Random seed for reproducible output.
    """
    texts, speaker_label = [], dict()
    with open(fname, "rt", encoding=encoding) as f:
        csv_r = csv.DictReader(f, delimiter="\t")
        for row in csv_r:
            texts.append((row['id'], row['speaker'], row[text_head]))
            speaker_label[row['speaker']] = int(row.get('label', -1))

    X = [text for _, _, text in texts]
    y = [speaker_label[spk] for _, spk, _ in texts]

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=seed)
    for train_index, val_index in kf.split(X):
        t_trn, t_val = [X[i] for i in train_index], [X[i] for i in val_index]
        y_trn, y_val = [y[i] for i in train_index], [y[i] for i in val_index]

        yield t_trn, y_trn, t_val, y_val