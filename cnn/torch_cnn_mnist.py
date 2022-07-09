import torch
import nn_utils
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.01
NO_OF_EPOCHS = 8
BATCH_SIZE = 50

DATA_DIR = '/Users/lukanamacinski/FER-workspace/deep-learning/data'
SAVE_DIR = '/Users/lukanamacinski/FER-workspace/deep-learning/out'

TRANSFORM = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)))

DS_TRAIN, DS_TEST = MNIST(DATA_DIR, train=True, download=False, transform=TRANSFORM), MNIST(DATA_DIR, train=False,
                                                                                            transform=TRANSFORM)

TRAIN, VALIDATION = random_split(DS_TRAIN, [55000, 5000])

TRAIN_LOADER = DataLoader(TRAIN, batch_size=BATCH_SIZE, shuffle=True)
VALIDATION_LOADER = DataLoader(VALIDATION, batch_size=BATCH_SIZE, shuffle=True)
TEST_LOADER = DataLoader(DS_TEST, batch_size=BATCH_SIZE, shuffle=False)


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), padding=(2, 2)),
                       nn.MaxPool2d(kernel_size=(2, 2)),
                       nn.ReLU(),
                       nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
                       nn.MaxPool2d(kernel_size=(2, 2)),
                       nn.ReLU(),
                       nn.Flatten(),
                       nn.Linear(in_features=1568, out_features=512),
                       nn.ReLU(),
                       nn.Linear(in_features=512, out_features=10)])

    def forward(self, x):
        h = torch.Tensor(x)
        for layer in self.layers:
            h = layer.forward(h)
        return h


def train_model(model, num_examples, train_loader, validation_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    losses = []
    assert num_examples % BATCH_SIZE == 0
    num_batches = num_examples // BATCH_SIZE

    for epoch in range(1, NO_OF_EPOCHS + 1):
        epoch_loss = 0
        cnt_correct = 0
        for i, (x, y_) in enumerate(train_loader):
            y_pred = model.forward(x)
            loss = criterion(y_pred, y_)
            loss.backward()
            epoch_loss += loss.detach().numpy()

            optimizer.step()
            optimizer.zero_grad()

            predicted_class = torch.argmax(y_pred, dim=1)
            cnt_correct += (predicted_class == y_).sum().numpy()

            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i * BATCH_SIZE, num_examples, loss))
            if i % 100 == 0:
                nn_utils.draw_conv_filters_torch(epoch, i * BATCH_SIZE, model.layers[0].weight, SAVE_DIR)
            if i > 0 and i % 50 == 0:
                print("Train accuracy = %.2f" % (cnt_correct / ((i + 1) * BATCH_SIZE) * 100))
        print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
        evaluate("Validation", validation_loader, model)
        losses.append(epoch_loss / num_batches)

    plt.title("Gubitci kroz epohe")
    plt.plot(range(1, NO_OF_EPOCHS+1), losses)
    plt.show()
    return model


def evaluate(name, validation_loader, model):
    with torch.no_grad():
        print("\nRunning evaluation: ", name)
        batch_size = BATCH_SIZE
        num_examples = len(validation_loader.dataset)
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size
        cnt_correct = 0
        loss_avg = 0
        criterion = nn.CrossEntropyLoss()
        for i, (x, y_) in enumerate(validation_loader):
            y_pred = model.forward(x)
            loss = criterion(y_pred, y_)
            loss_avg += loss.detach().numpy()
            predicted_class = torch.argmax(y_pred, dim=1)
            cnt_correct += (predicted_class == y_).sum().numpy()
        valid_acc = cnt_correct / num_examples * 100
        loss_avg /= num_batches
        print(name + " accuracy = %.2f" % valid_acc)
        print(name + " avg loss = %.2f\n" % loss_avg)


cnn = ConvolutionalNetwork()
train_model(model=cnn, num_examples=len(TRAIN_LOADER.dataset), train_loader=TRAIN_LOADER,
            validation_loader=VALIDATION_LOADER)
