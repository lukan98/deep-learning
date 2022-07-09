import torch
import torchvision
import pt_deep
import data
import numpy as np
import math
import matplotlib.pyplot as plt

torch.manual_seed(100)
np.random.seed(100)

dataset_root = '/Users/lukanamacinski/FER-workspace/deep-learning/data'  # change this to your preference
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=False)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=False)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()

activation_function = lambda x: x

def train_early_stop(model, X, Y_, param_niter, param_delta, param_lambda=0,
                     verbose=False, print_step=1, early_stop_trigger=5):
    if (print_step <= 0):
        raise ValueError('The print step must be at least 1')
    if (not torch.is_tensor(X) or not torch.is_tensor(Y_)):
        raise ValueError("X and Yoh_ both must be torch.Tensor data types!")
    if (early_stop_trigger <= 0):
        raise ValueError('The early stop trigger must be a positive integer!')

    X_train, Y_train, X_validation, Y_validation = data.sample_validation_set(X, Y_, int(len(X) / 5))
    best_val_loss = math.inf

    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

    for i in range(int(param_niter)):
        norm_sum = 0
        if param_lambda > 0:
            for weight_index in range(len(model.weights)):
                norm_sum += torch.linalg.norm(model.weights[weight_index])
        loss = model.get_loss(X_train, Y_train) + param_lambda * norm_sum
        loss.backward()
        optimizer.step()

        validation_loss = model.get_loss(X_validation, Y_validation) + param_lambda * norm_sum

        if verbose and i % print_step == 0:
            print(f'step: {i}, loss:{loss}')
            print(f'Validation loss: {validation_loss}')

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_model = model
            counter = 0
        else:
            counter += 1
            if counter == early_stop_trigger:
                model.weights = best_model.weights
                model.biases = best_model.biases
                print(f'Training stopped early at iteration {i-early_stop_trigger}')
                break

        optimizer.zero_grad()

def train_mb(model, X, Y_, param_niter, param_delta, param_lambda=0,
             verbose=False, print_step=1, batch_size=None):
    if batch_size == None:
        batch_size = X.shape[0]
    if (print_step <= 0):
        raise ValueError('The print step must be at least 1')
    if (not torch.is_tensor(X) or not torch.is_tensor(Y_)):
        raise ValueError("X and Yoh_ both must be torch.Tensor data types!")

    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

    for i in range(int(param_niter)):
        permuted_indices = torch.randperm(len(X))
        X_batches = torch.split(X[permuted_indices], batch_size)
        Y_batches = torch.split(Y_[permuted_indices], batch_size)

        losses = []
        for batch_index in range(len(X_batches)):
            norm_sum = 0
            if param_lambda > 0:
                for weight_index in range(len(model.weights)):
                    norm_sum += torch.linalg.norm(model.weights[weight_index])
            loss = model.get_loss(X_batches[batch_index], Y_batches[batch_index]) + param_lambda * norm_sum
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if verbose and i % print_step == 0:
            print(f'step: {i}, average loss per batch:{sum(losses)/len(losses)}')

    return

model = pt_deep.PTDeep([D, C], activation_function)

# train_early_stop(model, X=torch.Tensor(x_train.reshape(-1, D)), Y_=torch.Tensor(data.class_to_onehot(y_train)),
#                  param_niter=1000, param_delta=0.1, param_lambda=0.01,
#                  verbose=True, print_step=10, early_stop_trigger=5)

# train_mb(model, X=torch.Tensor(x_train.reshape(-1, D)), Y_=torch.Tensor(data.class_to_onehot(y_train)),
#          param_niter=1000, param_delta=0.1, param_lambda=0.01,
#          verbose=True, print_step=10, batch_size=1000)

pt_deep.train(model, X=torch.Tensor(x_train.reshape(-1, D)), Yoh_=torch.Tensor(data.class_to_onehot(y_train)),
              optimizer=torch.optim.SGD, param_niter=10000, param_delta=0.2, param_lambda=0, schedule=False,
              verbose=True, print_step=10)

Y_hat = np.argmax(pt_deep.eval(model, x_test.reshape(-1, D)), axis=1)

accuracy, precision, confusion_matrix = data.eval_perf_multi(Y_hat, y_test)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Confusion matrix:\n', confusion_matrix)

weights = model.weights[0].detach().numpy().T.reshape((-1, 28, 28))

fig, ax = plt.subplots(2, 5)

for i, number in enumerate(weights):
    ax[int(i / 5), i % 5].imshow(number)

plt.show()
