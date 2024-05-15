from typing import Optional, Callable
from typing_extensions import Self
from collections.abc import Iterable
from collections import defaultdict
from pathlib import Path

import csv
import torch
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from scipy.sparse import csr_matrix


class PositionalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            vocabulary: Optional[Iterable],
            tokenizer: Optional[Callable]
        ) -> None:
        self.vocabulary_ = vocabulary
        self.tokenizer = tokenizer or self.build_tokenizer()
        self.max_sentence_length_ = 0

    def token_to_index(self, token):
        if token in self.vocabulary_:
            return self.vocabulary_[token]
        else:
            return self.vocabulary_['<UNK>']

    def index_to_token(self, index):
        return list(self.vocabulary_.keys())[index]

    def fit(self, X: list[str], y: Optional[list] = None):
        """Learn vocabulary from provided sentences and labels
        May need a sentence splitter?
        """
        vocabulary: set = set()
        max_sentence_length = 0

        # Iterate through sentences in the input, tokenize, and update vocabulary
        for sentence in tqdm(X, total=len(X), desc="Fit\t\t", unit="sample"):
            tokens = self.tokenizer(sentence)
            if len(tokens) > max_sentence_length:
                max_sentence_length = len(tokens)
            
            new_token_set = set(tokens)
            vocabulary.union(new_token_set)
        
        self.vocabulary_ = ['<UNK>'] + list(vocabulary) + ['<PAD>']
        self.max_sentence_length_ = max_sentence_length

        return self

    def transform(self, X, y: Optional[list] = None):

        crow, col, token_val = [], [], []

        # Iterate through sentences, construct the necessary arrays to create CSR sparse matrix
        for i, text in tqdm(enumerate(X), total=len(X), desc="Transform\t", unit="sample"):
            tokens = self.tokenizer(text)
            crow.append(i * self.max_sentence_length_)

            for j, token in enumerate(tokens):
                col.append(j)

                # Append index of tokens and tags to the val arrays
                if token in self.vocabulary_:
                    token_val.append(self.token_to_index(token))
                else:
                    token_val.append(self.token_to_index("<UNK>"))

            # Add padding to make all sentences have the same length
            padding_amt = self.max_sentence_length_ - len(tokens)
            token_val += [self.token_to_index("<PAD>")] * padding_amt

            # Column index of the paddings runs from (len of tokens) to max_sentence_length
            col += list(range(len(tokens), self.max_sentence_length_))

        assert len(token_val) == self.max_sentence_length_ * len(X), \
            f"Length of token_val is incorrect: {len(token_val)} != {self.max_sentence_length_ * len(X)}"

        # Construct sparse matrices
        mat_size = (len(X), self.max_sentence_length_)
        tokens_sparse = torch.sparse_csr_tensor(crow, col, token_val, size=mat_size, dtype=torch.long)

        return tokens_sparse
    
    #TODO: use spacy tokenizer

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


class RawParliamentData():
    """Class to hold raw data load directly from the tsv files.
    """
    def __init__(self, ids: list[str], speakers: list[str], texts: list[str], labels: list[int]) -> None:
        assert len(ids) == len(speakers) == len(texts) == len(labels), "All arrays must have the same length"
        self.ids_ = ids
        self.speakers_ = speakers
        self.texts_ = texts
        self.labels_ = labels

    def subset(self, index_list: list[int]):

        data = RawParliamentData(
            [self.ids_[idx] for idx in index_list],
            [self.speakers_[idx] for idx in index_list],
            [self.texts_[idx] for idx in index_list],
            [self.labels_[idx] for idx in index_list],
        )
        
        return data

    def __getitem__(self, index: int):
        return (self.ids_[index], self.speakers_[index], self.texts_[index], self.labels_[index])

    def __add__(self, other: Self):
        return RawParliamentData(
            self.ids_ + other.ids_,
            self.speakers_ + other.speakers_,
            self.texts_ + other.texts_,
            self.labels_ + other.labels_
        )

    def __iter__(self):
        for data in zip(self.ids_, self.speakers_, self.texts_, self.labels_):
            yield data

    def __len__(self):
        return len(self.ids_)


class ParliamentDataset(Dataset):
    """Custom Dataset object to hold parliament debate data. Each item in the dataset
    is a tuple of (input tensor, label)
    """
    def __init__(
            self, 
            inputs: torch.Tensor, 
            labels: torch.Tensor, 
        ) -> None:
        super().__init__()
        assert len(inputs) == len(labels), "Inputs and labels have different length"
        self.data_ = list(zip(inputs, labels))

    def __len__(self):
        return len(self.data_)

    def __getitem__(self, index):
        return self.data_[index]
    
    def __iter__(self):
        for data in self.data_:
            yield data


def load_data(folder_path: str | Path, file_list: list, text_head: str = 'text') -> RawParliamentData:
    """Load the Parliament Debate dataset. 

    Parameters
    ----------
    folder_path : str | Path
        Parent folder containing the text files
    file_list : list
        List of files you want to load
    text_head : str, optional
        Name of the text column, either 'text' or 'text_en', by default 'text'

    Returns
    -------
    list[tuple]
        Returns a list of tuples containing: text ID, speaker ID, text, label
    """
    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    data = RawParliamentData([], [], [], [])
    
    for fname in file_list:
        print(f"Load {fname}...")
        tmp_data = _load_one(file_path=folder_path / fname, text_head=text_head)

        data += tmp_data

    return data


def _load_one(file_path, encoding: str = 'utf-8', text_head: str = 'text') -> RawParliamentData:
    """Load one file and return """
    
    ids         : list[str] = []
    speakers    : list[str] = []
    texts       : list[str] = []
    labels      : list[int] = []

    with open(file_path, "rt", encoding=encoding) as f:
        csv_r = csv.DictReader(f, delimiter="\t")
        
        for row in csv_r:
            ids.append(row.get('id'))
            speakers.append(row.get('speaker'))
            texts.append(row.get(text_head))
            labels.append(int(row.get('label', -1)))
        
    return RawParliamentData(ids, speakers, texts, labels)


def split_data(data: RawParliamentData, test_size=0.2, random_state=None) -> tuple[RawParliamentData, RawParliamentData]:
    """Return train-test sets divided by speakers, so that speakers of the train and test set do not overlap"""
    speaker_indices = defaultdict(list)

    # Inverted index: {speaker: [idx1, idx2]}
    for idx, speaker in enumerate(data.speakers_):
        speaker_indices[speaker].append(idx)

    # Split list of (indices list)
    train_indices_lst, test_indices_lst = train_test_split(
        list(speaker_indices.values()), 
        test_size=test_size, 
        random_state=random_state
    )

    train_indices = [idx for lst in train_indices_lst for idx in lst]
    test_indices = [idx for lst in test_indices_lst for idx in lst]
    
    return data.subset(train_indices), data.subset(test_indices)


def encode_text(encoder: PositionalEncoder | TfidfVectorizer, texts: list[str]) -> torch.Tensor:
    """Convenience function to encode text using a supplied encoder

    Parameters
    ----------
    encoder : PositionalEncoder | TfidfVectorizer
        A fitted (trained) encoder
    texts : list[str]

    Returns
    -------
    torch.Tensor
        Tensor containing vector representation of text
    """
    # No fit_, because the encoder should be already fitted with training data
    enc_texts_csr = encoder.transform(texts)  
    
    if isinstance(enc_texts_csr, csr_matrix):
        enc_texts = torch.from_numpy(enc_texts_csr.todense()).float()
    else:
        enc_texts = enc_texts_csr.to_dense()
    
    return enc_texts


def create_dataset(raw_data: RawParliamentData, vectorizer: PositionalEncoder | TfidfVectorizer):
    """Convenience function to create the final dataset"""
    inputs = encode_text(vectorizer, raw_data.texts_)
    labels = torch.tensor(raw_data.labels_)
    return ParliamentDataset(inputs, labels)

