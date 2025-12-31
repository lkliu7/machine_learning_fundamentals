"""
Fully Connected Neural Network for MNIST Classification

This script implements a feedforward neural network with configurable hidden layers
to classify handwritten digits from the MNIST dataset. The network uses:
- ReLU activation for hidden layers
- Softmax activation for output layer
- Cross-entropy loss function
- Mini-batch stochastic gradient descent for optimization
- He weight initialization for better convergence
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip
import random
import itertools

# MARK: Configuration
CONFIG = {
   'epochs': 10,
   'learning_rate': 1e-1,
   'hidden_dims': [256, 256], # e.g., [256, 256] for 2 hidden layers with 256 neurons each
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

# Flatten 28x28 images to 784-dimensional vectors for fully connected network
train_data = np.array([img.flatten() for img in train_images])
test_data = np.array([img.flatten() for img in test_images])

# Create class mappings for converting between labels and integer indices
classes = list(set(train_labels))  # [0, 1, 2, ..., 9]
n_classes = len(classes)  # 10 classes for digits 0-9
class_int_label = {classes[i]: i for i in range(n_classes)}  # {0:0, 1:1, ..., 9:9}

# Extract hyperparameters from configuration
epochs = CONFIG['epochs']
lr = CONFIG['learning_rate']
hidden_dims = CONFIG['hidden_dims']
update_frequency = CONFIG['update_frequency']
batch_size = CONFIG['batch_size']

# Get dataset dimensions
n_train, input_dim = train_data.shape  # input_dim = 784 (28*28 flattened pixels)
n_test, _ = test_data.shape

# Network architecture setup
L = len(hidden_dims)  # Number of hidden layers
all_dims = [input_dim] + hidden_dims + [n_classes]  # [784, 256, 256, 10] for 2 hidden layers

# Initialize weights and biases for each layer using He initialization
# He initialization: scale weights by sqrt(2/fan_in) for ReLU activations
W = {}  # Weight matrices: W[l] has shape (layer_l_size, layer_(l-1)_size)
b = {}  # Bias vectors: b[l] has shape (layer_l_size,)
for l in range(1, L+2):  # Layers 1 to L+1 (L hidden + 1 output)
    scale = np.sqrt(2.0 / all_dims[l-1])  # He initialization scale factor
    W[l] = np.random.randn(all_dims[l], all_dims[l-1]) * scale
    b[l] = np.zeros(all_dims[l])

def dist(vec):
    """Forward propagation for a single input vector to compute class probabilities.

    Args:
        vec: Input vector of shape (784,) representing flattened image

    Returns:
        Probability distribution over classes of shape (10,)
    """
    z = vec  # Current layer activations
    for j in range(1, L+2):  # For each layer (hidden + output)
        a = W[j] @ z + b[j]  # Linear transformation: Wz + b
        z = np.maximum(a, 0)  # ReLU activation: max(0, a)
    # Note: 'a' from the last iteration is the output layer pre-activation
    y = np.exp(a)  # Softmax numerator: exp(output_logits)
    return y / np.sum(y)  # Softmax: normalize to probability distribution

def pred(vec):
    """Predict class for a single input vector.

    Args:
        vec: Input vector of shape (784,)

    Returns:
        Predicted class label (0-9)
    """
    y = dist(vec)  # Get probability distribution
    return classes[np.argmax(y)]  # Return class with highest probability

def batch_dist(vecs):
    """Forward propagation for a batch of input vectors (vectorized version).

    Args:
        vecs: Input batch of shape (batch_size, 784)

    Returns:
        Probability distributions of shape (batch_size, 10)
    """
    Z = vecs.T  # Transpose to (784, batch_size) for matrix multiplication
    for j in range(1, L+2):  # For each layer
        A = W[j] @ Z + b[j][:, np.newaxis]  # Linear: (layer_size, batch_size)
        Z = np.maximum(A, 0)  # ReLU activation
    # A from last iteration contains output layer pre-activations
    Y = np.exp(A)  # Softmax numerator: (10, batch_size)
    Y = Y / Y.sum(axis=0)  # Softmax normalization along class dimension
    return Y.T  # Transpose back to (batch_size, 10)

def batch_pred(vecs):
    """Predict classes for a batch of input vectors.

    Args:
        vecs: Input batch of shape (batch_size, 784)

    Returns:
        Array of predicted class labels of shape (batch_size,)
    """
    Y = batch_dist(vecs)  # Get probability distributions
    preds = np.argmax(Y, axis=1)  # Get class indices with highest probabilities
    return np.array([classes[pred] for pred in preds])  # Convert indices to labels

# Evaluate initial performance before training
train_preds = batch_pred(train_data)
matches = (train_preds == train_labels)
acc = [(0, np.sum(matches) / n_train)]  # List of (epoch, accuracy) tuples
test_preds = batch_pred(test_data)
test_matches = (test_preds == test_labels)
test_acc = [(0, np.sum(test_matches) / n_test)]  # Initial test accuracy
loss_history = []  # List of (epoch, loss) tuples

# Main training loop with mini-batch stochastic gradient descent
for m in range(1,epochs+1):

    # Initialize storage for forward and backward pass computations
    A = {}  # Pre-activations for each layer
    Z = {}  # Activations for each layer
    e = {}  # Error/gradient signals for backpropagation
    gradQW = {}  # Weight gradients
    gradQb = {}  # Bias gradients

    # Create random mini-batches for this epoch
    shuffle = random.sample(range(n_train), n_train)  # Shuffle training indices
    batches = itertools.batched(shuffle, batch_size)  # Split into mini-batches

    # Process each mini-batch
    for batch in batches:
        batch_data = train_data[list(batch)]
        batch_labels = train_labels[list(batch)]
        n_batch, _ = batch_data.shape

        # Forward propagation
        Z[0] = batch_data.T  # Input layer: (784, batch_size)
        # Hidden layers with ReLU activation
        for j in range(1, L+1):
            A[j] = W[j] @ Z[j-1] + b[j][:, np.newaxis]  # Linear transformation
            Z[j] = np.maximum(A[j], 0)  # ReLU activation
        # Output layer (no activation yet)
        A[L+1] = W[L+1] @ Z[L] + b[L+1][:, np.newaxis]

        # Softmax activation for output layer
        Y = np.exp(A[L+1].T)  # Shape: (batch_size, 10)
        Y = np.array([row / np.sum(row) for row in Y])  # Normalize to probabilities

        # Convert labels to one-hot encoding for cross-entropy loss
        label_mat = np.zeros((n_batch, n_classes))
        label_indices = [class_int_label[label] for label in batch_labels]
        label_mat[np.arange(n_batch), label_indices] = 1  # One-hot encoding

        # Backpropagation: compute error signals
        e[L+1] = Y - label_mat  # Output layer error: prediction - true_label
        # Propagate error backwards through hidden layers
        for j in range(L,0,-1):
            # Error = (next_layer_error @ next_layer_weights) * ReLU_derivative
            e[j] = (e[j+1] @ W[j+1]) * np.heaviside(Z[j], 0).T

        # Compute gradients
        for j in range(1, L+2):
            gradQW[j] = (Z[j-1] @ e[j]).T  # Weight gradient: input @ error
            gradQb[j] = np.sum(e[j], axis=0)  # Bias gradient: sum of errors

        # Update weights and biases using gradient descent
        for j in range(1, L+2):
            W[j] -= lr * gradQW[j] / n_batch  # Scale by batch size
            b[j] -= lr * gradQb[j] / n_batch

    # Periodically evaluate performance and compute loss
    if m % update_frequency == 0:
        # Forward pass on entire training set to compute loss
        batch = train_data
        n_batch, _ = train_data.shape
        Z[0] = batch.T
        for j in range(1, L+1):
            A[j] = W[j] @ Z[j-1] + b[j][:, np.newaxis]
            Z[j] = np.maximum(A[j], 0)
        A[L+1] = W[L+1] @ Z[L] + b[L+1][:, np.newaxis]
        Y = np.exp(A[L+1].T)
        Y = np.array([row / np.sum(row) for row in Y])

        # Compute cross-entropy loss
        label_mat = np.zeros((n_batch, n_classes))
        label_indices = [class_int_label[label] for label in train_labels]
        label_mat[np.arange(n_batch), label_indices] = 1
        # Cross-entropy: -sum(true_labels * log(predictions))
        loss_history.append((m-1, -np.sum(np.log(Y) * label_mat)))
        print(loss_history[-1])  # Print current loss

        # Evaluate accuracy on both training and test sets
        train_preds = batch_pred(train_data)
        matches = (train_preds == train_labels)
        acc.append((m, np.sum(matches) / n_train))
        test_preds = batch_pred(test_data)
        test_matches = (test_preds == test_labels)
        test_acc.append((m, np.sum(test_matches) / n_test))

# Plot training loss over time
iters, losses = zip(*loss_history)

plt.plot(iters, losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss History")
plt.yscale('log')
plt.grid(True)
plt.show()

# Plot training and test accuracy over time
iters, accuracies = zip(*acc)
_, test_accuracies = zip(*test_acc)

plt.plot(iters, accuracies, label='Training Accuracy')
plt.plot(iters, test_accuracies, label='Test Accuracy')
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy History")
plt.legend()
plt.grid(True)
plt.show()

# Print final training and test accuracies
print(f"Final training accuracy: {acc[-1]}")
print(f"Final test accuracy: {test_acc[-1]}")