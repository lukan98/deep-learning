# Deep Learning Labs

A collection of deep-learning exercises written in Python and PyTorch for the **Deep Learning** course at FER, University of Zagreb.

## Contents

| Directory | Topic |
|---|---|
| `binary-logistic-regression/` | Binary logistic regression implemented from scratch with NumPy |
| `logistic-regression/` | Multiclass logistic regression, a two-layer neural network, PyTorch models, an SVM wrapper and an MNIST model comparison |
| `cnn/` | Convolutional network layers implemented with NumPy/Cython, plus PyTorch CNN training on MNIST and CIFAR-10 |
| `rnn/` | RNN and baseline models for SST-2 sentiment classification using GloVe embeddings |

## Requirements

- Python 3.8 or newer
- Python packages:
  - `numpy`
  - `matplotlib`
  - `torch`
  - `torchvision`
  - `scikit-learn`
  - `scipy`
  - `scikit-image`
  - `Cython` (only needed to build the Cython extension in `cnn/`)

Install them with:

```bash
pip install numpy matplotlib torch torchvision scikit-learn scipy scikit-image Cython
```

## Data

The repository already contains the datasets used by the labs:

- `cnn/data/MNIST/`
- `cnn/data/cifar-10-batches-py/`
- `rnn/data/` (SST train/validation/test splits and GloVe embeddings)

Several scripts contain absolute paths from the original author's machine, for example:

- `cnn/train.py`, `cnn/train_l2reg.py`, `cnn/torch_cnn_mnist.py`, `cnn/torch_cnn_cifar.py`: `DATA_DIR` and `SAVE_DIR`
- `logistic-regression/mnist_shootout.py`: `dataset_root`
- `rnn/data_loading.py`: `TRAIN_SOURCE`, `VALID_SOURCE`, `TEST_SOURCE`, `EMBEDDING_SOURCE`

Update those paths to match your checkout before running the scripts.

## Cython extension

The NumPy CNN implementation depends on the `im2col_cython` extension. Build it once from the `cnn/` directory:

```bash
cd cnn
python3 setup_cython.py build_ext --inplace
```

## Running

Most modules are executable and contain a `__main__` block. Run them from their own directory so that local imports resolve correctly, for example:

```bash
cd logistic-regression
python3 fcann2.py
```

Other examples:

```bash
cd binary-logistic-regression
python3 binlogreg.py

cd cnn
python3 torch_cnn_mnist.py

cd rnn
python3 rnn.py
```

## Notes

- Training the CNN and RNN models can take a significant amount of time.
- The `cnn/data` directory is large because it contains the full MNIST and CIFAR-10 datasets.
