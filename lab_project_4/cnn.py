import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip
import random
import itertools
from numpy.lib.stride_tricks import as_strided

# MARK: Configuration
CONFIG = {
   'epochs': 10,
   'learning_rate': 1e-2,
   'batch_size': 77,
}

# MARK: Data Preparation

# Check whether MNIST data exists.

def get_mnist_data():
    paths = ['archive/train-labels.idx1-ubyte', 
             'archive/train-images.idx3-ubyte', 
             'archive/t10k-images.idx3-ubyte', 
             'archive/t10k-labels.idx1-ubyte']
    for path in paths:
        if not os.path.exists(path):
            file = path.removeprefix('archive/').replace('.','-') + '.gz'
            file = urllib.request.urlretrieve('https://github.com/fgnt/mnist/raw/refs/heads/master/' + file)
            with open(file[0], 'rb') as gz, open(path, 'wb') as out:
                out.write(gzip.decompress(gz.read()))


if not os.path.exists('archive'):
    os.mkdir('archive')

get_mnist_data()

# Load in MNIST data and preprocess it for neural network training.

def read_idx_images(filename):
    """Read MNIST image data from IDX file format.
    
    IDX format specification:
        - Magic number (4 bytes): 2051 for image files
        - Number of images (4 bytes)
        - Number of rows (4 bytes): 28 for MNIST
        - Number of columns (4 bytes): 28 for MNIST
        - Image data: unsigned bytes in row-major order
    
    Args:
        filename: Path to IDX3-ubyte file
        
    Returns:
        ndarray of shape (num_images, 28, 28) with dtype uint8
        
    Raises:
        ValueError: If magic number is not 2051
    """

    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2051:
            raise ValueError(f"Invalid magic number for IDX3 file: expected 2051, got {magic}")
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)

    return images

def read_idx_labels(filename):
    """Read MNIST label data from IDX file format.

    IDX1 format specification:
        - Magic number (4 bytes): 2049 for label files
        - Number of labels (4 bytes)
        - Label data: unsigned bytes (0-9 for MNIST)

    Args:
        filename: Path to IDX1-ubyte file

    Returns:
        ndarray of shape (num_labels,) with dtype uint8

    Raises:
        ValueError: If magic number is not 2049
    """
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2049:
            raise ValueError(f"Invalid magic number for IDX1 file: expected 2049, got {magic}")
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels

# Load raw MNIST data from IDX files
train_images = read_idx_images('archive/train-images.idx3-ubyte')
train_labels = read_idx_labels('archive/train-labels.idx1-ubyte')
test_images = read_idx_images('archive/t10k-images.idx3-ubyte')
test_labels = read_idx_labels('archive/t10k-labels.idx1-ubyte')

# Normalize pixel values to [0,1] range for better neural network training
train_images = train_images.astype(np.float32) / 255
test_images = test_images.astype(np.float32) / 255

# Create class mappings for converting between labels and integer indices
classes = list(set(train_labels))  # [0, 1, 2, ..., 9]
n_classes = len(classes)  # 10 classes for digits 0-9
class_int_label = {classes[i]: i for i in range(n_classes)}  # {0:0, 1:1, ..., 9:9}

# Extract hyperparameters from configuration
epochs = CONFIG['epochs']
lr = CONFIG['learning_rate']
batch_size = CONFIG['batch_size']

# Get dataset dimensions
n_train, *input_dim = train_images.shape  # input_dim = (28,28)
n_test, *_ = test_images.shape

def maxpool(array, spec):
    dims = array.shape
    rank = len(dims)
    if rank != len(spec):
        raise ValueError()
    dims = [dims[i] // spec[i] for i in range(rank)]
    pooled_dims = tuple(zip(dims, spec))
    pooled_dims = tuple(n for t in pooled_dims for n in t)
    pooled_array = array.reshape(pooled_dims)
    pooled_axes = tuple(range(1, 2*rank, 2))
    return np.max(pooled_array, axis=pooled_axes)

def maxpoolD(array, spec):
    pooled = maxpool(array, spec)
    shift = np.max(np.abs(pooled)) / np.sqrt(2)
    aug = np.kron(pooled, np.ones(spec)) + shift
    pre = (array + shift) / aug - 1
    return (abs(pre) < 1e-10).astype(float)

dims = ((32, 3, 3), (64, 3, 3), (64, 3, 3), (2, 2), 7744, 128, 10)

W = {}
b = {}
kernel_dim = dims[0][-1] * dims[0][-2]
scale = np.sqrt(2 / kernel_dim)
W[1] = np.random.randn(kernel_dim, dims[0][0]) * scale
kernel_dim = dims[0][0] * dims[1][-1] * dims[1][-2]
scale = np.sqrt(2 / kernel_dim)
W[2] = np.random.randn(kernel_dim, dims[1][0]) * scale
kernel_dim = dims[1][0] * dims[2][-1] * dims[2][-2]
scale = np.sqrt(2 / kernel_dim)
W[3] = np.random.randn(kernel_dim, dims[2][0])
input_dim = 64 * 11 * 11
scale = np.sqrt(2 / input_dim)
W[5] = np.random.randn(dims[4], 7744) * scale
b[5] = np.zeros(dims[4])
scale = np.sqrt(2 / dims[4])
W[6] = np.random.randn(dims[5], dims[4]) * scale
b[6] = np.zeros(dims[5])
scale = np.sqrt(2 / dims[5])
W[7] = np.random.randn(dims[6], dims[5]) * scale
b[7] = np.zeros(dims[6])
