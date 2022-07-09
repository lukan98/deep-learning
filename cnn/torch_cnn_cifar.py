import os

import numpy
import torch
import numpy as np
from matplotlib import pyplot as plt

import nn_utils
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import skimage as ski
import skimage.io

# Hyperparameters
LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.9
WEIGHT_DECAY = 0.01
NO_OF_EPOCHS = 1
BATCH_SIZE = 50
CLASS_COUNT = 10

DATA_DIR = '/Users/lukanamacinski/FER-workspace/deep-learning/data'
SAVE_DIR = '/Users/lukanamacinski/FER-workspace/deep-learning/out'

TRANSFORM = transforms.Compose((transforms.ToTensor(),
                                transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))))

DS_TRAIN, DS_TEST = CIFAR10(DATA_DIR, train=True, download=False, transform=TRANSFORM), CIFAR10(DATA_DIR, train=False,
                                                                                                transform=TRANSFORM)

TRAIN, VALIDATION = random_split(DS_TRAIN, [45000, 5000])

TRAIN_LOADER = DataLoader(TRAIN, batch_size=BATCH_SIZE, shuffle=True)
VALIDATION_LOADER = DataLoader(VALIDATION, batch_size=BATCH_SIZE, shuffle=True)
TEST_LOADER = DataLoader(DS_TEST)


# UTILITY FUNCTIONS
def draw_image(img, mean, std):
    # img = img.transpose(1, 2, 0)
    # print(img.shape)
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)


class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=(5, 5), padding=(2, 2)),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=(2, 2)),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
             nn.Flatten(),
             nn.Linear(in_features=1568, out_features=256),
             nn.ReLU(),
             nn.Linear(in_features=256, out_features=128),
             nn.ReLU(),
             nn.Linear(in_features=128, out_features=10)])

    def forward(self, x):
        h = torch.Tensor(x)
        for layer in self.layers:
            h = layer.forward(h)
        return h


def train_model(model, num_examples, train_loader, validation_loader, evaluate_flag=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=LEARNING_RATE_DECAY)

    assert num_examples % BATCH_SIZE == 0

    plot_data = {'train_loss': [],
                 'valid_loss': [],
                 'train_acc': [],
                 'valid_acc': [],
                 'lr': []}

    for epoch in range(1, NO_OF_EPOCHS + 1):
        for i, (x, y_) in enumerate(train_loader):
            y_pred = model.forward(x)
            loss = criterion(y_pred, y_)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 5 == 0:
                print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i * BATCH_SIZE, num_examples, loss))
            if i % 100 == 0:
                nn_utils.draw_conv_filters_torch_multichannel(epoch, i * BATCH_SIZE, model.layers[0].weight, SAVE_DIR)

        if evaluate_flag:
            train_loss, train_acc = evaluate(name="Test", model=model, data_loader=train_loader, criterion=criterion)
            val_loss, val_acc = evaluate(name="Validation", model=model, data_loader=validation_loader, criterion=criterion)

            plot_data['train_loss'] += [train_loss]
            plot_data['valid_loss'] += [val_loss]
            plot_data['train_acc'] += [train_acc]
            plot_data['valid_acc'] += [val_acc]
            plot_data['lr'] += [scheduler.get_last_lr()]
            scheduler.step()

    if evaluate_flag:
        plot_training_progress(save_dir=SAVE_DIR, data=plot_data)
    return model


def evaluate(name, model, data_loader, criterion):
    confusion_matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=numpy.int64)
    losses = []
    with torch.no_grad():
        for _, (x, y_) in enumerate(data_loader):
            y_pred = model.forward(x)
            losses.append(criterion(y_pred, y_))
            y_pred = torch.argmax(y_pred, dim=1).detach().numpy()
            y_true = y_.detach().numpy()
            for i in range(len(x)):
                confusion_matrix[y_true[i]][y_pred[i]] += 1
        accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
        print(f'\n{name} confusion matrix: ')
        print(confusion_matrix)
        print(f'\n{name} accuracy = %.2f' % accuracy)
        for class_index in range(CLASS_COUNT):
            print(f'{class_index + 1} performance:')
            print(f'Precision = %.2f'
                  % (confusion_matrix[class_index][class_index] / np.sum(confusion_matrix[class_index, :]))
                  )
            print(f'Recall = %.2f'
                  % (confusion_matrix[class_index][class_index] / np.sum(confusion_matrix[:, class_index]))
                  )
    return np.average(losses), accuracy


cnn = ConvolutionalNetwork(3)
cnn = train_model(model=cnn, num_examples=len(TRAIN_LOADER.dataset),
                  train_loader=TRAIN_LOADER, validation_loader=VALIDATION_LOADER)

images_by_loss = []

with torch.no_grad():
    criterion = nn.CrossEntropyLoss()
    for i, (x, y_) in enumerate(TEST_LOADER):
        predicted_classes = cnn.forward(x)
        loss = criterion(predicted_classes, y_)
        images_by_loss.append((loss.detach().numpy(), i, predicted_classes.detach().numpy()[0]))
    images_by_loss.sort(reverse=True)
    images_by_loss = images_by_loss[:20]
    for (loss, index, predicted_classes) in images_by_loss:
        best_indices = np.argpartition(predicted_classes, -3)[-3:]
        print(f'Actual class: {DS_TEST.targets[index]}, Predicted class: {np.argmax(predicted_classes)}')
        print(f'Other guesses: {best_indices}, with probability of: {predicted_classes[best_indices]}, respectively')
        draw_image(img=DS_TEST.data[index], mean=0, std=1)


