import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip
import random
import itertools
from joblib import Parallel, delayed

# MARK: Configuration

CONFIG = {
    'digits': [5, 8],
    'learning_rate': 1e-4,
    'max_iterations': 100000,
    'convergence_tolerance': 1e-6,
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

digits = np.unique(test_labels)
digits = sorted(list(set(digits) & set(range(10))))
if not digits:
    raise ValueError('data error')
train_data = {d: np.array([img.flatten() for img in get_train_images_for_digit(d)]) for d in digits}
test_data = {d: np.array([img.flatten() for img in get_test_images_for_digit(d)]) for d in digits}

# MARK: Primary formulation

digits = CONFIG['digits']
data = np.vstack([train_data[n] for n in digits])
lr_labels = np.concat([np.ones(train_size(digits[0])),
                       -np.ones(train_size(digits[1]))])
n = len(lr_labels)

def h(x):
    return np.maximum(0, 1-x)

def th(x):
    return np.heaviside(x, 1)

def error(w, b, c, data, lr_labels):
    return np.dot(w, w) / 2 + c * np.sum(h(lr_labels * (data @ w + b)))

def gradw(w, b, c, data, lr_labels):
    p = 1 - lr_labels * (data @ w + b)
    return w - c * np.sum(data * lr_labels[:, None] * th(p)[:, None], axis=0)

def gradb(w, b, c, data, lr_labels):
    p = 1 - lr_labels * (data @ w + b)
    return -c * np.sum(lr_labels[:, None] * th(p)[:, None], axis=0)

lr = CONFIG['learning_rate']
max_iterations = CONFIG['max_iterations']
w = np.zeros(784)
b = 0
c = 0.25
prev_loss = error(w,b,c,data,lr_labels)
loss_history = [(0,prev_loss)]
tol = CONFIG['convergence_tolerance']

for i in range(max_iterations):
    w -= lr * gradw(w,b,c,data,lr_labels)
    b -= lr * gradb(w,b,c,data,lr_labels)
    current_loss = error(w,b,c,data,lr_labels)
    loss_history.append((i+1, current_loss))
    if abs(prev_loss/current_loss - 1) < tol:
        print(f'converged at iteration {i+1}')
        break
    prev_loss = current_loss
    if i % 100 == 99:
        print(i+1, 'iterations completed, error =', current_loss)

# MARK: Multi-class SVM

digits = np.unique(test_labels)
digits = sorted(list(set(digits) & set(range(10))))

def h(x):
    return np.maximum(0, 1-x)

def th(x):
    return np.heaviside(x, 1)

def error(w, b, c, data, lr_labels):
    return np.dot(w, w) / 2 + c * np.sum(h(lr_labels * (data @ w + b)))

def gradw(w, b, c, data, lr_labels):
    p = 1 - lr_labels * (data @ w + b)
    return w - c * np.sum(data * lr_labels[:, None] * th(p)[:, None], axis=0)

def gradb(w, b, c, data, lr_labels):
    p = 1 - lr_labels * (data @ w + b)
    return -c * np.sum(lr_labels[:, None] * th(p)[:, None], axis=0)

pairs = [(i,j) for i in range(10) for j in range(10) if j > i]
lr = CONFIG['learning_rate']
max_iterations = CONFIG['max_iterations']
tol = CONFIG['convergence_tolerance']
c = 0.25

def train_pair(pair):
    """Train binary classifier for one pair of digits"""
    data = np.vstack([train_data[n] for n in pair])
    lr_labels = np.concat([np.ones(train_size(pair[0])),
                           -np.ones(train_size(pair[1]))])
    
    w = np.zeros(784)
    b = 0
    c = 0.25
    lr = CONFIG['learning_rate']
    max_iterations = CONFIG['max_iterations']
    tol = CONFIG['convergence_tolerance']
    
    prev_loss = error(w, b, c, data, lr_labels)
    
    for i in range(max_iterations):
        w -= lr * gradw(w, b, c, data, lr_labels)
        b -= lr * gradb(w, b, c, data, lr_labels)
        current_loss = error(w, b, c, data, lr_labels)
        
        if abs(prev_loss/current_loss - 1) < tol:
            break
        prev_loss = current_loss
    
    return (pair, w, b)

results = [train_pair(pair) for pair in pairs]

# Parallelized version.

# results = Parallel(n_jobs=-1)(delayed(train_pair)(pair) for pair in pairs)
    
w = {result[0]: result[1] for result in results}
b = {result[0]: result[2] for result in results}

def pred_pair(pair, img):
    vec = img.flatten()
    index = -np.sign(w[pair] @ vec + b[pair])/2 + 1/2
    index = int(index[0])
    return pair[index]

def pred(img):
    votes = [pred_pair(pair, img) for pair in pairs]
    values, votes = np.unique(votes, return_counts=True)
    max_vote = votes.max()
    most_frequent = values[votes == max_vote]
    return int(random.choice(most_frequent))

classifications = {}
confusion_matrix = {}
for label in digits:
    labels, counts = np.unique([pred(img) for img in train_data[label]], return_counts=True)
    confusion_matrix[label] = dict(zip(labels,counts))
    classifications[label] = dict(zip(labels,counts/len(train_data[label])))

print(confusion_matrix)
print(classifications)

# MARK: Results

def pred(x, w, b=0):
    return np.sign(w.dot(x) + b)

def pred_data(data, w, b=0):
    return np.sign(data @ w + b)

def pred_accuracy(results, label):
    return (results == label).mean()

training_error = error(w,b,c,data,lr_labels)
print([training_error])

train_preds_0 = pred_data(train_data[digits[0]], w, b)
train_preds_1 = pred_data(train_data[digits[1]], w, b)
test_preds_0 = pred_data(test_data[digits[0]], w, b)
test_preds_1 = pred_data(test_data[digits[1]], w, b)

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