import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip
import random
import itertools

# MARK: Configuration
CONFIG = {
   'epochs': 100,
   'learning_rate': 1e-1,
   'hidden_dims': [256, 256],
   'update_frequency': 1,
   'batch_size': 128,
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

# Load in MNIST data.

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
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic != 2049:
            raise ValueError(f"Invalid magic number for IDX1 file: expected 2049, got {magic}")
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels

train_images = read_idx_images('archive/train-images.idx3-ubyte')
train_labels = read_idx_labels('archive/train-labels.idx1-ubyte')
test_images = read_idx_images('archive/t10k-images.idx3-ubyte')
test_labels = read_idx_labels('archive/t10k-labels.idx1-ubyte')

train_images = train_images.astype(np.float32) / 255
test_images = test_images.astype(np.float32) / 255

train_data = np.array([img.flatten() for img in train_images])
test_data = np.array([img.flatten() for img in test_images])
classes = list(set(train_labels))
n_classes = len(classes)
class_int_label = {classes[i]: i for i in range(n_classes)}

epochs = CONFIG['epochs']
lr = CONFIG['learning_rate']
hidden_dims = CONFIG['hidden_dims']
update_frequency = CONFIG['update_frequency']
batch_size = CONFIG['batch_size']

n_train, d0 = train_data.shape
n_test, _ = test_data.shape
L = len(hidden_dims)
all_dims = [d0] + hidden_dims + [n_classes]
W = {}
b = {}
for l in range(1, L+2):
    scale = np.sqrt(2.0 / all_dims[l-1])
    W[l] = np.random.randn(all_dims[l], all_dims[l-1]) * scale
    b[l] = np.zeros(all_dims[l])

def dist(vec):
    z = vec
    for j in range(1, L+2):
        a = W[j] @ z + b[j]
        z = np.maximum(a, 0)
    y = np.exp(a)
    return y / np.sum(y)

def pred(vec):
    y = dist(vec)
    return classes[np.argmax(y)]

def batch_dist(vecs):
    Z = vecs.T
    for j in range(1, L+2):
        A = W[j] @ Z + b[j][:, np.newaxis]
        Z = np.maximum(A, 0)
    Y = np.exp(A)
    Y = Y / Y.sum(axis=0)
    return Y.T

def batch_pred(vecs):
    Y = batch_dist(vecs)
    preds = np.argmax(Y, axis=1)
    return np.array([classes[pred] for pred in preds])

train_preds = batch_pred(train_data)
matches = (train_preds == train_labels)
acc = [(0, np.sum(matches) / n_train)]
test_preds = batch_pred(test_data)
test_matches = (test_preds == test_labels)
test_acc = [(0, np.sum(test_matches) / n_test)]
loss_history = []

for m in range(1,epochs+1):
    A = {}
    Z = {}
    e = {}
    gradQW = {}
    gradQb = {}
    shuffle = random.sample(range(n_train), n_train)
    batches = itertools.batched(shuffle, batch_size)
    for batch in batches:
        batch_data = train_data[list(batch)]
        batch_labels = [train_labels[i] for i in batch]
        n_batch, _ = batch_data.shape
        Z[0] = batch_data.T
        for j in range(1, L+1):
            A[j] = W[j] @ Z[j-1] + b[j][:, np.newaxis]
            Z[j] = np.maximum(A[j], 0)
        A[L+1] = W[L+1] @ Z[L] + b[L+1][:, np.newaxis]
        Y = np.exp(A[L+1].T)
        Y = np.array([row / np.sum(row) for row in Y])
        label_mat = np.zeros((n_batch, n_classes))
        label_indices = [class_int_label[label] for label in batch_labels]
        label_mat[np.arange(n_batch), label_indices] = 1
        e[L+1] = Y - label_mat
        for j in range(L,0,-1):
            e[j] = (e[j+1] @ W[j+1]) * np.heaviside(Z[j], 0).T
        for j in range(1, L+2):
            gradQW[j] = (Z[j-1] @ e[j]).T
            gradQb[j] = np.sum(e[j], axis=0)
        for j in range(1, L+2):
            W[j] -= lr * gradQW[j] / n_batch
            b[j] -= lr * gradQb[j] / n_batch
    if m % update_frequency == 0:
        batch = train_data
        n_batch, _ = train_data.shape
        Z[0] = batch.T
        for j in range(1, L+1):
            A[j] = W[j] @ Z[j-1] + b[j][:, np.newaxis]
            Z[j] = np.maximum(A[j], 0)
        A[L+1] = W[L+1] @ Z[L] + b[L+1][:, np.newaxis]
        Y = np.exp(A[L+1].T)
        Y = np.array([row / np.sum(row) for row in Y])
        label_mat = np.zeros((n_batch, n_classes))
        label_indices = [class_int_label[label] for label in train_labels]
        label_mat[np.arange(n_batch), label_indices] = 1
        loss_history.append((m-1, -np.sum(np.log(Y) * label_mat)))
        print(loss_history[-1])
        train_preds = batch_pred(train_data)
        matches = (train_preds == train_labels)
        acc.append((m, np.sum(matches) / n_train))
        test_preds = batch_pred(test_data)
        test_matches = (test_preds == test_labels)
        test_acc.append((m, np.sum(test_matches) / n_test))

iters, losses = zip(*loss_history)

plt.plot(iters, losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss History")
plt.yscale('log')
plt.grid(True)
plt.show()

iters, accuracies = zip(*acc)
_, test_accuracies = zip(*test_acc)

plt.plot(iters, accuracies, iters, test_accuracies)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy History")
plt.grid(True)
plt.show()

print(acc[-1], test_acc[-1])