import torch
import numpy as np
import torch.nn as nn

import data_loading
import metrics

from torch.utils.data import DataLoader


class RecurrentNetwork(nn.Module):
    def __init__(self,
                 dataset: data_loading.NLPDataset,
                 freeze: bool,
                 rnn_cell,
                 hidden_size: int = 150,
                 num_layers: int = 2,
                 bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.recurrent_layers = nn.ModuleList([rnn_cell(input_size=300,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers,
                                                        bidirectional=bidirectional),
                                               rnn_cell(input_size=150,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers,
                                                        bidirectional=bidirectional)])
        self.layers = nn.ModuleList([nn.Linear(in_features=hidden_size, out_features=hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(in_features=hidden_size, out_features=1)])
        self.embedding_matrix = data_loading.create_embedding_matrix(dataset.text_vocab,
                                                                     source_path_string=data_loading.EMBEDDING_SOURCE,
                                                                     freeze=freeze)

        self.double()

    def forward(self, input):
        y = self.embedding_matrix(input)
        h = None
        y = torch.transpose(y, 0, 1)
        for layer in self.recurrent_layers:
            y, h = layer(y, h)

        y = y[-1]

        for layer in self.layers:
            y = layer(y)
        return y

    def parameters(self, recurse: bool = True):
        parameters = list()
        for layer in self.recurrent_layers:
            parameters += list(layer.parameters())
        for layer in self.layers:
            parameters += list(layer.parameters())
        parameters += list(self.embedding_matrix.parameters())
        return parameters

    def predict(self, input):
        with torch.no_grad():
            y = torch.sigmoid(self.forward(input))
            y = torch.round(y).int()
        return y


def train(model: nn.Module, data: DataLoader,
          optimizer: torch.optim.Optimizer, criterion, clip_value):
    model.train()
    for batch_num, (text, label, _) in enumerate(data):
        model.zero_grad()
        logits = model(text)
        loss = criterion(logits, label.double().reshape(-1, 1))
        loss.backward()

        if batch_num % 10 == 0:
            print(f'\tBatch: {batch_num}, loss: {loss}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()


def evaluate(model: nn.Module, data: DataLoader, criterion):
    model.eval()
    with torch.no_grad():
        y_hat = None
        y = None
        for batch_num, (text, label, _) in enumerate(data):
            logits = model.forward(text)
            loss = criterion(logits, label.double().reshape(-1, 1))

            if batch_num % 10 == 0:
                print(f'Evaluation batch: {batch_num}, loss: {loss}')

            if y_hat is None:
                y_hat = model.predict(text).detach().numpy().reshape(1, -1)
            else:
                y_hat = np.append(y_hat, model.predict(text).detach().numpy().reshape(1, -1))
            if y is None:
                y = label.detach().numpy().reshape(1, -1)
            else:
                y = np.append(y, label.detach().numpy().reshape(1, -1))
        confusion_matrix = metrics.get_confusion_matrix(y_hat, y)
        print('Confusion matrix:')
        print(confusion_matrix)
        print(f'Accuracy: {metrics.get_accuracy(confusion_matrix)}')
        print(f'F1: {metrics.get_f1(confusion_matrix)}')


if __name__ == '__main__':
    seed = 7052020
    epochs = 5
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = data_loading.NLPDataset(csv_source_string=data_loading.TRAIN_SOURCE)
    valid_dataset = data_loading.NLPDataset(csv_source_string=data_loading.VALID_SOURCE,
                                            text_vocab=train_dataset.text_vocab,
                                            label_vocab=train_dataset.label_vocab)

    test_dataset = data_loading.NLPDataset(csv_source_string=data_loading.TEST_SOURCE,
                                           text_vocab=train_dataset.text_vocab,
                                           label_vocab=train_dataset.label_vocab)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=10,
                                  shuffle=True, collate_fn=data_loading.collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=32,
                                  shuffle=False, collate_fn=data_loading.collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32,
                                 shuffle=False, collate_fn=data_loading.collate_fn)

    model = RecurrentNetwork(train_dataset, freeze=True, rnn_cell=nn.RNN)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}')
        train(model, train_dataloader, optimizer, criterion, clip_value=1)
        evaluate(model, valid_dataloader, criterion)

    evaluate(model, test_dataloader, criterion)
