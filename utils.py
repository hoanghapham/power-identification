from typing import Optional, Callable
from collections.abc import Iterable
from collections import defaultdict
from pathlib import Path

import csv
import torch
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from scipy.sparse import csr_matrix

def get_data(fname, encoding='utf-8', testset=False, text_head='text', test_size=0.2, seed=None):
    """Read the test set or training set and split to train/val sets.

    Parameters:
    fname       The filename to read data from.
    encoding    The encoding of the file.
    testset     If True, return only (ids, texts, labels), no
                training/validation split is done, all labels will
                be -1 if the file does not include labels.
    text_head   The header of the text field,
                useful for reading the English translations.
    test_size   Size or ratio of the test data (see the documentation
                of scikit-learn train_test_split() for details).
    seed        Random seed for reproducible output.
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

    # First, split the speakers to train/test sets such that
    # there are no overlap of the authors across the split.
    # This is similar to how orientation test set was split.
    spkset = list(speaker_label.keys())
    labelset = list(speaker_label.values())
    s_trn, s_val, _, _ = train_test_split(spkset, labelset,
                      test_size=test_size, random_state=seed)
    s_val = set(s_val)
    # Now split the speeches based on speakers split above
    t_trn, t_val, y_trn, y_val = [], [], [], []
    for i, (_, spk, text) in enumerate(texts):
        if spk in s_val:
            t_val.append(text)
            y_val.append(speaker_label[spk])
        else:
            t_trn.append(text)
            y_trn.append(speaker_label[spk])
    return t_trn, y_trn, t_val, y_val


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


class PositionalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            vocabulary: Optional[Iterable] = None, 
            tokenizer: Optional[Callable] = None
        ) -> None:
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer or self.build_tokenizer()

        self.idx2token_: list = []
        self.token2idx_: dict = {}
        self.max_sentence_length = 0

    def fit(self, X: list[str], y: Optional[list] = None):
        """Learn vocabulary from provided sentences and labels
        May need a sentence splitter?
        """
        vocabulary = set()
        max_sentence_length = 0

        for sentence in tqdm(X, total=len(X), desc="Fit\t\t", unit="sample"):
            tokens = self.tokenizer(sentence)
            if len(tokens) > max_sentence_length:
                max_sentence_length = len(tokens)
            
            for token in tokens:
                vocabulary.add(token)
        
        self.idx2token_ = ['<UNK>'] + list(vocabulary) + ['<PAD>']
        self.token2idx_ = {token: idx for idx, token in enumerate(self.idx2token_)}
        self.max_sentence_length = max_sentence_length

        return self

    def transform(self, X, y: Optional[list] = None):

        crow, col, token_val = [], [], []

        # Iterate through sentences, construct the necessary arrays to create CSR sparse matrix
        for i, text in tqdm(enumerate(X), total=len(X), desc="Transform\t", unit="sample"):
            tokens = self.tokenizer(text)
            crow.append(i * self.max_sentence_length)

            for j, token in enumerate(tokens):
                col.append(j)

                # Append index of tokens and tags to the val arrays
                if token in self.token2idx_:
                    token_val.append(self.token2idx_[token])
                else:
                    token_val.append(self.token2idx_["<UNK>"])

            # Add padding to make all sentences have the same length
            padding_amt = self.max_sentence_length - len(tokens)
            token_val += [self.token2idx_["<PAD>"]] * padding_amt

            # Column index of the paddings runs from (len of tokens) to max_sentence_length
            col += list(range(len(tokens), self.max_sentence_length))

        assert len(token_val) == self.max_sentence_length * len(X), \
            f"Length of token_val is incorrect: {len(token_val)} != {self.max_sentence_length * len(X)}"

        # Construct sparse matrices
        print("Construct matrix")
        mat_size = (len(X), self.max_sentence_length)
        tokens_sparse = torch.sparse_csr_tensor(crow, col, token_val, size=mat_size, dtype=torch.float32)

        return tokens_sparse
    
    def build_tokenizer(self):
        def simple_tokenizer(sentence):
            words: list = sentence.split(' ')
            new_words = []

            for word in words:
                # If there is no punctuation in the word, add to the final list
                # Else, split the words further at the punctuations
                if all([char.isalnum() for char in word]):
                    new_words.append(word)

                else:
                    tmp = ''
                    # Iterate through characters. When encounter a punctuation,
                    # add the previous characters as a word, then add the punctuation
                    for char_idx, char in enumerate(word):
                        if char.isalnum():
                            tmp += char
                            if char_idx == len(word) - 1:
                                new_words.append(tmp)
                        else:
                            if char_idx > 0:
                                new_words.append(tmp)
                            new_words.append(char)
                            tmp = ''
            return new_words
        return simple_tokenizer
    
    def fit_transform(self, X, y: Optional[list] = None):
        self.fit(X, y)
        tokens = self.transform(X, y)
        return tokens


class CustomDataset(Dataset):
    def __init__(self, data, vectorizer: PositionalEncoder | TfidfVectorizer) -> None:
        self.vectorizer = vectorizer
        texts = [tup[2] for tup in data]
        labels = [tup[3] for tup in data]
        enc_texts = self.encode_data(vectorizer, texts)
        self.data_ = list(zip(enc_texts, labels))

    def __len__(self):
        return len(self.data_)

    def __getitem__(self, index):
        return self.data_[index]
    
    def __iter__(self):
        for data in self.data_:
            yield data
    
    def encode_data(self, vectorizer: PositionalEncoder | TfidfVectorizer, texts: list[str]):
        enc_texts_csr = vectorizer.transform(texts)
        
        if isinstance(enc_texts_csr, csr_matrix):
            enc_texts = torch.from_numpy(enc_texts_csr.todense()).float()
        else:
            enc_texts = enc_texts_csr.to_dense()
        
        return enc_texts


class DataProcessor():

    def __init__(self) -> None:
        pass

    def load_data(self, folder_path: str | Path, file_list: list, text_head: str = 'text'):
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)

        data: list[tuple] = []
        
        for fname in file_list:
            print(f"Load {fname}...")
            data += self._load_one(file_path=folder_path / fname, text_head=text_head)
                
        return data

    def _load_one(self, file_path, encoding: str = 'utf-8', text_head: str = 'text'):
        
        data: list[tuple] = []

        with open(file_path, "rt", encoding=encoding) as f:
            csv_r = csv.DictReader(f, delimiter="\t")
            for row in csv_r:
                data.append((
                    row.get('id'), 
                    row.get('speaker'), 
                    row.get(text_head), 
                    int(row.get('label', -1))
                ))
            
        return data

    def split_data(self, data, test_size=0.2, random_state=0):
        """Return train-test sets divided by speakers"""
        speaker_indices = defaultdict(list)

        speakers = [tup[1] for tup in data]

        for idx, speaker in enumerate(speakers):
            speaker_indices[speaker].append(idx)

        train_idx_lst, test_idx_lst = train_test_split(
            list(speaker_indices.values()), 
            test_size=test_size, 
            random_state=random_state
        )

        train_idx = [idx for lst in train_idx_lst for idx in lst]
        test_idx = [idx for lst in test_idx_lst for idx in lst]

        train = [data[i] for i in train_idx]
        test = [data[i] for i in test_idx]

        return train, test


    # def encode_data(
    #         self, 
    #         texts: list[str],
    #         fitted_vectorizer: Optional[TfidfVectorizer | PositionalEncoder] = None, 
    #         init_vec_class: Optional[type] = None,
    #     ):
    #     print("Encoding...")
    #     if fitted_vectorizer is None:
    #         assert init_vec_class is not None, "Must provide an initial vectorizer class"
    #         vectorizer = init_vec_class()
    #         vectorizer.fit(texts)
    #     else:
    #         vectorizer = fitted_vectorizer

    #     enc_texts_csr = vectorizer.transform(texts)

    #     if isinstance(enc_texts_csr, csr_matrix):
    #         enc_texts = torch.from_numpy(enc_texts_csr.todense()).float()
    #     else:
    #         enc_texts = enc_texts_csr.to_dense()
        
    #     return enc_texts, vectorizer

        
        

