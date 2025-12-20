import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip
import random
import itertools

# MARK: Configuration
CONFIG = {
# Digit pairs for binary classification
    'digits': [5, 8],
    'inverse_shift': 1e-2,

    # Learning rate for gradient descent
    # Higher values converge faster but may overshoot
    # Typical range: 1e-6 to 1e-3 depending on loss scaling
    'learning_rate': 1e-6,

    # Maximum training iterations
    # Early stopping occurs if convergence_tolerance is met
    'max_iterations': 100000,

    # Convergence criterion: |L(t-1)/L(t) - 1| < tol
    # Stops when relative change in loss is below threshold
    'convergence_tolerance': 1e-6,

    # Mini-batch size for SGD
    # Larger batches give more stable gradients but slower updates
    # Must divide evenly into dataset size for clean epochs
    'batch_size': 64,
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

def get_train_images_for_digit(n):
    'Get images for given labels.'
    return train_images[train_labels == n]

def train_size(n):
    'Number of training images for given label.'
    return np.sum(train_labels == n)

def get_test_images_for_digit(n):
    'Get images for given labels.'
    return test_images[test_labels == n]

def test_size(n):
    'Number of test images for given label.'
    return np.sum(test_labels == n)

digits = CONFIG['digits']
digits = sorted(list(set(digits) & set(range(10))))
if not digits:
    raise ValueError('no valid digits in config')
train_data = {d: np.array([img.flatten() for img in get_train_images_for_digit(d)]) for d in digits}
test_data = {d: np.array([img.flatten() for img in get_test_images_for_digit(d)]) for d in digits}

# MARK: Computation

data = np.vstack([train_data[n] for n in digits])
lr_labels = np.concat([np.ones(train_size(digits[0])),
                       -np.ones(train_size(digits[1]))])
n = len(lr_labels)

# Exact solution.

shift = CONFIG['inverse_shift']
w_exact = np.linalg.inv(np.transpose(data) @ data + shift * np.eye(784)) @ np.transpose(data) @ lr_labels
def pred_exact(x):
    return np.sign(w_exact.dot(x))

# Full gradient descent method.

def error(w):
    vec = data @ w - lr_labels
    return vec @ vec

yX_vec = 2 * lr_labels @ data
XTX = data.T @ data
def grad(w):
    return 2 * XTX @ w - yX_vec

lr = CONFIG['learning_rate']
max_iterations = CONFIG['max_iterations']
w = np.zeros(784)
prev_loss = error(w)
loss_history = [(0, prev_loss)]
tol = CONFIG['convergence_tolerance']

for i in range(max_iterations):
    w -= lr * grad(w)
    current_loss = error(w)
    loss_history.append((i+1, current_loss))
    if abs(prev_loss/current_loss - 1) < tol:
        print(f'converged at iteration {i+1}')
        break
    prev_loss = current_loss
    if i % 100 == 99:
        print(i+1, 'iterations completed, error =', current_loss)

# Stochastic gradient descent method.

def batchGrad(w, vecs, labels):
    """Compute gradient for mini-batch in linear regression.

    Implements: ∇L = 2 * Σᵢ (xᵢᵀw - yᵢ)xᵢ
    where L = Σᵢ (xᵢᵀw - yᵢ)² is the squared loss.

    Args:
        w: Weight vector
        vecs: Mini-batch feature matrix (batch_size, 784)
        labels: Mini-batch labels ∈ {-1, +1}

    Returns:
        Gradient vector of shape (784,)

    Note:
        Implementation uses NumPy broadcasting: vecs @ w @ vecs - labels @ vecs
        produces correct gradient due to dimension coercion. This formulation
        deliberately avoids computing vecs.T to reduce transpose overhead.
        Gradient is computed for mini-batch only. For equivalent behavior to
        full-batch GD, learning rate should be scaled by (batch_size / total_samples)
        or regularization adjusted.
    """
    return 2 * (vecs @ w @ vecs - labels @ vecs)

def batchSGD(w, batchSize, lr):
    shuffle = random.sample(range(n), n)
    batches = itertools.batched(shuffle, batchSize)
    for batch in batches:
        w -= lr * batchGrad(w, data[list(batch)], lr_labels[list(batch)])

lr = CONFIG['learning_rate']
max_iterations = CONFIG['max_iterations']
batchSize = CONFIG['batch_size']
w_SGD = np.zeros(784)
prev_loss = error(w_SGD)
loss_history_SGD = [(0, prev_loss)]
tol = CONFIG['convergence_tolerance']

for i in range(max_iterations):
    batchSGD(w_SGD, batchSize, lr)
    current_loss = error(w_SGD)
    loss_history_SGD.append((i+1, current_loss))
    if abs(prev_loss/current_loss - 1) < tol:
        print(f'converged at iteration {i+1}')
        break
    prev_loss = current_loss
    if i % 100 == 99:
        print(i+1, 'iterations completed, error =', current_loss)

# MARK: Results

def pred_data(data, w):
    return np.sign(data @ w)

def pred_accuracy(results, label):
    return (results == label).mean()

training_error = error(w)
training_error_SGD = error(w_SGD)
training_error_exact = error(w_exact)
print([training_error, training_error_SGD, training_error_exact])

train_preds_0 = pred_data(train_data[digits[0]], w_exact)
train_preds_1 = pred_data(train_data[digits[1]], w_exact)
test_preds_0 = pred_data(test_data[digits[0]], w_exact)
test_preds_1 = pred_data(test_data[digits[1]], w_exact)

train_preds_0_acc = pred_accuracy(train_preds_0, 1)
train_preds_1_acc = pred_accuracy(train_preds_1, -1)
test_preds_0_acc = pred_accuracy(test_preds_0, 1)
test_preds_1_acc = pred_accuracy(test_preds_1, -1)

print([train_preds_0_acc, train_preds_1_acc, test_preds_0_acc, test_preds_1_acc])

train_preds_0 = pred_data(train_data[digits[0]], w_SGD)
train_preds_1 = pred_data(train_data[digits[1]], w_SGD)
test_preds_0 = pred_data(test_data[digits[0]], w_SGD)
test_preds_1 = pred_data(test_data[digits[1]], w_SGD)

train_preds_0_acc = pred_accuracy(train_preds_0, 1)
train_preds_1_acc = pred_accuracy(train_preds_1, -1)
test_preds_0_acc = pred_accuracy(test_preds_0, 1)
test_preds_1_acc = pred_accuracy(test_preds_1, -1)

print([train_preds_0_acc, train_preds_1_acc, test_preds_0_acc, test_preds_1_acc])

train_preds_0 = pred_data(train_data[digits[0]], w)
train_preds_1 = pred_data(train_data[digits[1]], w)
test_preds_0 = pred_data(test_data[digits[0]], w)
test_preds_1 = pred_data(test_data[digits[1]], w)

train_preds_0_acc = pred_accuracy(train_preds_0, 1)
train_preds_1_acc = pred_accuracy(train_preds_1, -1)
test_preds_0_acc = pred_accuracy(test_preds_0, 1)
test_preds_1_acc = pred_accuracy(test_preds_1, -1)

print([train_preds_0_acc, train_preds_1_acc, test_preds_0_acc, test_preds_1_acc])

iters, losses = zip(*loss_history)

plt.plot(iters, losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss History")
plt.yscale('log')
plt.grid(True)
plt.show()

iters, losses = zip(*loss_history_SGD)

plt.plot(iters, losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss History")
plt.yscale('log')
plt.grid(True)
plt.show()