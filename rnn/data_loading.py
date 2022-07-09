import torch
import csv
import numpy as np

from dataclasses import dataclass
from collections.abc import Iterable
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Embedding
from typing import Dict, List, Optional


TEST_SOURCE = "/Users/lukanamacinski/FER-workspace/deep-learning/data/LAB-3/sst_test_raw.csv"
TRAIN_SOURCE = "/Users/lukanamacinski/FER-workspace/deep-learning/data/LAB-3/sst_train_raw.csv"
VALID_SOURCE = "/Users/lukanamacinski/FER-workspace/deep-learning/data/LAB-3/sst_valid_raw.csv"
EMBEDDING_SOURCE = "/Users/lukanamacinski/FER-workspace/deep-learning/data/LAB-3/sst_glove_6b_300d.txt"


class Vocab:
    stoi: Dict
    itos: Dict

    def __init__(self,
                 frequencies: Dict[str, int],
                 max_size: int = -1,
                 min_freq: int = 0,
                 add_special_chars: bool = True):

        if max_size != -1 and max_size <= 0 or min_freq < 0:
            raise ValueError

        self.stoi = dict()
        self.itos = dict()
        self.add_special_chars = add_special_chars

        sorted_frequencies = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}

        if max_size != -1:
            sorted_frequencies = dict(list(sorted_frequencies.items())[:max_size])

        if self.add_special_chars:
            self.stoi["<PAD>"] = 0
            self.itos[0] = "<PAD>"
            self.stoi["<UNK>"] = 1
            self.itos[1] = "<UNK>"

        for index, (key, _) in enumerate(sorted_frequencies.items()):
            self.stoi[key] = index + 2 if self.add_special_chars else index
            self.itos[index + 2 if self.add_special_chars else index] = key

    def encode(self, token_list: List[str] or str):
        encoded_tokens = []
        if isinstance(token_list, List):
            for token in token_list:
                encoded_tokens.append(self.stoi[token] if token in self.stoi.keys() else self.stoi["<UNK>"])

        if isinstance(token_list, str):
            encoded_tokens = self.stoi[token_list] if token_list in self.stoi.keys() else self.stoi["<UNK>"]

        return torch.tensor(encoded_tokens, dtype=torch.long)


@dataclass
class Instance(Iterable):
    text: str
    label: str

    def __iter__(self):
        return iter((self.text, self.label))


class NLPDataset(Dataset):
    instances: List
    text_vocab: Vocab
    label_vocab: Vocab

    def __init__(self,
                 csv_source_string: str,
                 vocab_max_size: int = -1,
                 vocab_min_freq: int = 0,
                 text_vocab: Vocab = None,
                 label_vocab: Vocab = None):
        with open(csv_source_string) as source:
            self.instances = []
            csv_reader = csv.reader(source)
            for row in list(csv_reader):
                self.instances.append(Instance(row[0].split(), row[1].strip()))
        if text_vocab is None and label_vocab is None:
            self.__calculate_frequencies__()
            self.text_vocab = self.__create_text_vocab(max_size=vocab_max_size, min_freq=vocab_min_freq)
            self.label_vocab = self.__create_label_vocab()
        else:
            self.text_vocab = text_vocab
            self.label_vocab = label_vocab

    def __getitem__(self, index):
        return self.text_vocab.encode(self.instances[index].text), self.label_vocab.encode(self.instances[index].label)

    def __len__(self):
        return len(self.instances)

    def __calculate_frequencies__(self):
        self.__text_dict = dict()
        self.__label_dict = dict()

        for instance in self.instances:
            instance_text, instance_label = instance
            for token in instance_text:
                if token in self.__text_dict:
                    self.__text_dict[token] += 1
                else:
                    self.__text_dict[token] = 1
            if instance_label in self.__label_dict:
                self.__label_dict[instance_label] += 1
            else:
                self.__label_dict[instance_label] = 1

    def __create_text_vocab(self, max_size: int = -1, min_freq: int = 0):
        return Vocab(self.__text_dict, max_size=max_size, min_freq=min_freq)

    def __create_label_vocab(self):
        return Vocab(self.__label_dict, add_special_chars=False)


def create_embedding_matrix(vocabulary: Vocab, source_path_string: str = None, freeze: bool = False) -> Embedding:
    string_to_vector = parse_vector_representation_file(source_path_string=source_path_string)
    embedding_matrix = np.zeros(shape=(len(vocabulary.stoi.keys()), 300))

    for index, word in enumerate(vocabulary.stoi.keys()):
        if index == 0:
            embedding_matrix[index] = np.zeros(300)
        else:
            if string_to_vector is None:
                embedding_matrix[index] = np.random.rand(300)
            else:
                if word in string_to_vector.keys():
                    embedding_matrix[index] = string_to_vector[word]
                else:
                    embedding_matrix[index] = np.random.rand(300)

    embedding_matrix = torch.tensor(embedding_matrix)

    if string_to_vector is None:
        return Embedding.from_pretrained(embeddings=embedding_matrix,
                                         freeze=freeze,
                                         padding_idx=0)
    else:
        return Embedding.from_pretrained(embeddings=embedding_matrix,
                                         freeze=freeze,
                                         padding_idx=0)


def parse_vector_representation_file(source_path_string: str) -> Optional[Dict[str, np.ndarray]]:
    if source_path_string is None:
        return None

    source = open(file=source_path_string, mode='r')
    lines = source.readlines()

    string_to_vector = dict()

    for line in lines:
        line = line.split()
        string_to_vector[line[0]] = np.array(line[1:], dtype=float)

    return string_to_vector


def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts = pad_sequence(texts, True, 0)
    labels = torch.flatten(torch.tensor(labels))

    return texts, labels, lengths


if __name__ == '__main__':
    train_dataset = NLPDataset(csv_source_string=TRAIN_SOURCE)
    test_dataset = NLPDataset(csv_source_string=TRAIN_SOURCE)
    valid_dataset = NLPDataset(csv_source_string=TRAIN_SOURCE)

    instance_text, instance_label = train_dataset.instances[3]
    print(f"Text: {instance_text}")
    print(f"Label: {instance_label}")

    numericalized_text, numericalized_label = train_dataset[3]
    print(f"Numericalized text: {numericalized_text}")
    print(f"Numericalized label: {numericalized_label}")

    batch_size = 2  # Only for demonstrative purposes
    shuffle = False  # Only for demonstrative purposes
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, collate_fn=collate_fn)
    texts, labels, lengths = next(iter(train_dataloader))

    print(f"Texts: {texts}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")

    embedding_matrix = create_embedding_matrix(train_dataset.text_vocab,
                                               source_path_string=EMBEDDING_SOURCE)

    print(embedding_matrix(train_dataset.text_vocab.encode("the")))

    print(test_dataset.label_vocab.stoi)
    print(train_dataset.label_vocab.stoi)
    print(valid_dataset.label_vocab.stoi)
