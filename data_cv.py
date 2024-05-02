import csv
from sklearn.model_selection import train_test_split, KFold

def get_data(fname, encoding='utf-8', testset=False, text_head='text', test_size=0.2, seed=None, num_splits=None):
    """Read the test set or training set and split to train/val sets.

    Parameters:
    fname       The filename to read data from.
    testset     If True, return only (ids, texts, labels), no
                training/validation split is done, all labels will
                be -1 if the file does not include labels.
    text_head   The header of the text field,
                useful for reading the English translations.
    test_size   Size or ratio of the test data (see the documentation
                of scikit-learn train_test_split() for details)
    seed        Random seed for reproducible output
    num_splits  Number of folds for K-fold cross-validation
    """
    texts, speaker_label = [], dict()
    with open(fname, "rt", encoding=encoding) as f:
        csv_r = csv.DictReader(f, delimiter="\t")
        for row in csv_r:
            texts.append((row['id'], row['speaker'], row[text_head]))
            speaker_label[row['speaker']] = int(row.get('label', -1))

    if testset: # return only ID and text and label
        return ([x[0] for x in texts],
                [x[2] for x in texts],
                [speaker_label[x[1]] for x in texts])

    if num_splits is None:  # Split data into train and validation sets
        spkset = list(speaker_label.keys())
        labelset = list(speaker_label.values())
        s_trn, s_val, _, _ = train_test_split(spkset, labelset,
                          test_size=test_size, random_state=seed)
        s_val = set(s_val)
        t_trn, t_val, y_trn, y_val = [], [], [], []
        for i, (_, spk, text) in enumerate(texts):
            if spk in s_val:
                t_val.append(text)
                y_val.append(speaker_label[spk])
            else:
                t_trn.append(text)
                y_trn.append(speaker_label[spk])
        return t_trn, y_trn, t_val, y_val
    else:  # Perform K-fold cross-validation
        X = [text for _, _, text in texts]
        y = [speaker_label[spk] for _, spk, _ in texts]
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=seed)
        for train_index, val_index in kf.split(X):
            t_trn, t_val = [X[i] for i in train_index], [X[i] for i in val_index]
            y_trn, y_val = [y[i] for i in train_index], [y[i] for i in val_index]
            yield t_trn, y_trn, t_val, y_val
