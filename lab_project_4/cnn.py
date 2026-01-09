import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip
import random
import itertools
from numpy.lib.stride_tricks import sliding_window_view

# MARK: Configuration
CONFIG = {
   'epochs': 50,
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
W[3] = np.random.randn(kernel_dim, dims[2][0]) * scale
input_dim = dims[2][0] * 11 * 11
scale = np.sqrt(2 / input_dim)
W[5] = np.random.randn(dims[4], input_dim) * scale
b[5] = np.zeros(dims[4])
scale = np.sqrt(2 / dims[4])
W[6] = np.random.randn(dims[5], dims[4]) * scale
b[6] = np.zeros(dims[5])
scale = np.sqrt(2 / dims[5])
W[7] = np.random.randn(dims[6], dims[5]) * scale
b[7] = np.zeros(dims[6])

def dist(img):
    A = sliding_window_view(img, dims[0][1:]).reshape(26, 26, 9) @ W[1]
    Z = np.maximum(A, 0)
    A = sliding_window_view(Z, dims[1][1:], axis=(0,1)).reshape(24, 24, dims[0][0] * 9) @ W[2]
    Z = np.maximum(A, 0)
    A = sliding_window_view(Z, dims[2][1:], axis=(0,1)).reshape(22, 22, dims[1][0] * 9) @ W[3]
    Z = np.maximum(A, 0)
    A = maxpool(Z, (2,2,1))
    Z = A.flatten()
    A = W[5] @ Z + b[5]
    Z = np.maximum(A, 0)
    A = W[6] @ Z + b[6]
    Z = np.maximum(A, 0)
    A = W[7] @ Z + b[7]
    y = np.exp(A)
    y = y / np.sum(y)
    return y

def pred(img):
    return (classes[dist(img).argmax()])

def batch_dist(imgs, batch_size=512):
    n_batch = len(imgs)
    if n_batch > batch_size:
        batches = itertools.batched(imgs, batch_size)
        return np.concatenate([batch_dist(batch, batch_size) for batch in batches])
    A = sliding_window_view(imgs, dims[0][1:], axis=(1,2)).reshape(n_batch, 26, 26, 9) @ W[1]
    Z = np.maximum(A, 0)
    A = sliding_window_view(Z, dims[1][1:], axis=(1,2)).reshape(n_batch, 24, 24, dims[0][0] * 9) @ W[2]
    Z = np.maximum(A, 0)
    A = sliding_window_view(Z, dims[2][1:], axis=(1,2)).reshape(n_batch, 22, 22, dims[1][0] * 9) @ W[3]
    Z = np.maximum(A, 0)
    A = maxpool(Z, (1,2,2,1))
    Z = np.array([a.flatten() for a in A])
    A = Z @ W[5].T + b[5]
    Z = np.maximum(A, 0)
    A = Z @ W[6].T + b[6]
    Z = np.maximum(A, 0)
    A = Z @ W[7].T + b[7]
    A = np.array([a - a.max() for a in A])
    y = np.exp(A)
    y = y / np.sum(y, axis=1, keepdims=True)
    return y

def batch_pred(imgs):
    dists = batch_dist(imgs)
    return ([classes[dist.argmax()] for dist in dists])

train_preds = batch_pred(train_images)
matches = (train_preds == train_labels)
acc = [(0, np.sum(matches) / n_train)]
test_preds = batch_pred(test_images)
test_matches = (test_preds == test_labels)
test_acc = [(0, np.sum(test_matches) / n_test)]

for epoch in range(epochs):
    if epoch == 10:
        lr = lr / 10
    if epoch == 25:
        lr = lr / 10
    shuffle = random.sample(range(n_train), n_train)
    batches = itertools.batched(shuffle, batch_size)
    for batch in batches:
        batch = list(batch)
        n_batch = len(batch)
        data_batch = train_images[batch]
        batch_labels = train_labels[batch]
        batch_labels = [class_int_label[i] for i in batch_labels]
        constructed_W = {}
        constructed_W[3] = np.transpose(W[3].reshape(dims[1][0], 9, dims[2][0])[:,::-1,:].reshape(dims[1][0], 9 * dims[2][0]))
        constructed_W[2] = np.transpose(W[2].reshape(dims[0][0], 9, dims[1][0])[:,::-1,:].reshape(dims[0][0], 9 * dims[1][0]))
        Z = {}
        A = {}
        e = {}
        gradW = {}
        gradb = {}
        g = {}
        Z[0] = data_batch
        A[1] = sliding_window_view(Z[0], (3,3), axis=(1,2)).reshape(n_batch, 26, 26, 3*3) @ W[1]
        Z[1] = np.maximum(A[1], 0)
        A[2] = sliding_window_view(Z[1], (3,3), axis=(1,2)).reshape(n_batch, 24, 24, dims[0][0]*3*3) @ W[2]
        Z[2] = np.maximum(A[2], 0)
        A[3] = sliding_window_view(Z[2], (3,3), axis=(1,2)).reshape(n_batch, 22, 22, dims[1][0]*3*3) @ W[3]
        Z[3] = np.maximum(A[3], 0)
        A[4] = maxpool(Z[3], (1,2,2,1))
        Z[4] = A[4].reshape(n_batch, 11 * 11 * dims[2][0])
        A[5] = Z[4] @ W[5].T + b[5]
        Z[5] = np.maximum(A[5], 0)
        A[6] = Z[5] @ W[6].T + b[6]
        Z[6] = np.maximum(A[6], 0)
        A[7] = Z[6] @ W[7].T + b[7]
        A[7] = np.array([a - a.max() for a in A[7]])
        y = np.exp(A[7])
        y = y / np.sum(y, axis=1, keepdims=True)
        label_mat = np.zeros((n_batch, n_classes))
        label_indices = [class_int_label[label] for label in batch_labels]
        label_mat[np.arange(n_batch), label_indices] = 1
        e[7] = y - label_mat
        e[6] = (e[7] @ W[7]) * np.heaviside(Z[6], 0)
        e[5] = (e[6] @ W[6]) * np.heaviside(Z[5], 0)
        e[4] = e[5] @ W[5]
        e[3] = maxpoolD(Z[3], (1,2,2,1)) * np.kron(np.ones((2,2,1)), e[4].reshape(n_batch, 11, 11, dims[2][0])) * np.heaviside(Z[3], 0)
        e[2] = np.transpose(sliding_window_view(np.pad(e[3], ((0,0), (2,2), (2,2), (0,0))), (3,3), axis=(1,2)), (0,1,2,4,5,3)).reshape(n_batch, 24, 24, 9 * dims[2][0]) @ constructed_W[3]
        e[2] = e[2] * np.heaviside(Z[2], 0)
        e[1] = np.transpose(sliding_window_view(np.pad(e[2], ((0,0), (2,2), (2,2), (0,0))), (3,3), axis=(1,2)), (0,1,2,4,5,3)).reshape(n_batch, 26, 26, 9 * dims[1][0]) @ constructed_W[2]
        e[1] = e[1] * np.heaviside(Z[1], 0)
        gradW[7] = e[7].T @ Z[6]
        gradb[7] = np.sum(e[7], axis=0)
        gradW[6] = e[6].T @ Z[5]
        gradb[6] = np.sum(e[6], axis=0)
        gradW[5] = e[5].T @ Z[4]
        gradb[5] = np.sum(e[5], axis=0)
        g[3] = np.tensordot(sliding_window_view(Z[2], (22, 22), axis=(1,2)), e[3], ([0,4,5], [0,1,2]))
        g[2] = np.tensordot(sliding_window_view(Z[1], (24, 24), axis=(1,2)), e[2], ([0,4,5], [0,1,2]))
        g[1] = np.tensordot(sliding_window_view(Z[0], (26, 26), axis=(1,2)), e[1], ([0,3,4], [0,1,2]))
        gradW[3] = np.transpose(g[3], (2,0,1,3)).reshape(dims[1][0] * 9, dims[2][0])
        gradW[2] = np.transpose(g[2], (2,0,1,3)).reshape(dims[0][0] * 9, dims[1][0])
        gradW[1] = g[1].reshape(9, dims[0][0])
        for k in W:
            W[k] -= lr * gradW[k] / n_batch
        for k in b:
            b[k] -= lr * gradb[k] / n_batch

    train_preds = batch_pred(train_images)
    matches = (train_preds == train_labels)
    acc.append((epoch + 1, np.sum(matches) / n_train))
    test_preds = batch_pred(test_images)
    test_matches = (test_preds == test_labels)
    test_acc.append((epoch + 1, np.sum(test_matches) / n_test))
    print(f'Training accuracy after epoch {epoch + 1}: {acc[-1]}')
    print(f'Test accuracy after epoch {epoch + 1}: {test_acc[-1]}')

iters, accuracies = zip(*acc[1:])
_, test_accuracies = zip(*test_acc[1:])

plt.plot(iters, accuracies, label='Training Accuracy')
plt.plot(iters, test_accuracies, label='Test Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy History')
plt.legend()
plt.grid(True)
plt.show()

# Print final training and test accuracies
print(f'Final training accuracy: {acc[-1]}')
print(f'Final test accuracy: {test_acc[-1]}')