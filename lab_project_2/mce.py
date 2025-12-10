import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip

# MARK: Configuration
CONFIG = {
    'digits': [5, 8],
    'learning_rate': 1e-3,
    'max_iterations': 10000,
    'convergence_tolerance': 1e-6
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

def get_test_images_for_digit(n):
    'Get images for given labels.'
    return test_images[test_labels == n]

digits = CONFIG['digits']
digits = sorted(list(set(digits) & set(range(10))))
if not digits:
    raise ValueError('no valid digits in config')
data = np.vstack([get_train_images_for_digit(n) for n in digits])

# MARK: Computation

data = np.array([img.flatten() for img in data])
lr_labels = np.concat([np.ones(len(get_train_images_for_digit(digits[0]))),
                       -np.ones(len(get_train_images_for_digit(digits[1])))])

def l(x):
    return 1 / (1 + np.exp(-x))

n = len(lr_labels)

def error(w):
    return np.sum(l(lr_labels * (data @ w)))

def grad(w):
    p = l(lr_labels * (data @ w))
    return (lr_labels * p * (1 - p)) @ data

lr = CONFIG['learning_rate']
max_iterations = CONFIG['max_iterations']
w = np.zeros(784)
prev_loss = float('inf')
loss_history = [(0,error(w))]
tol = CONFIG['convergence_tolerance']

for i in range(max_iterations):
    w -= lr * grad(w)
    current_loss = error(w)
    loss_history.append((i+1, current_loss))
    if abs(prev_loss/current_loss - 1) < tol:
        print(f'converged at iteration {i+1}')
        break
    prev_loss = current_loss

def pred(x):
    return np.sign(-w.dot(x))

# MARK: Results

train_preds_0 = [pred(img.flatten()) for img in get_train_images_for_digit(digits[0])]
train_preds_1 = [pred(img.flatten()) for img in get_train_images_for_digit(digits[1])]
test_preds_0 = [pred(img.flatten()) for img in get_test_images_for_digit(digits[0])]
test_preds_1 = [pred(img.flatten()) for img in get_test_images_for_digit(digits[1])]

train_preds_0_acc = train_preds_0.count(1) / len(train_preds_0)
train_preds_1_acc = train_preds_1.count(-1) / len(train_preds_1)
test_preds_0_acc = test_preds_0.count(1) / len(test_preds_0)
test_preds_1_acc = test_preds_1.count(-1) / len(test_preds_1)

print([train_preds_0_acc, train_preds_1_acc, test_preds_0_acc, test_preds_1_acc])

iters, losses = zip(*loss_history)

plt.plot(iters, losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss History")
plt.yscale('log')
plt.grid(True)
plt.show() 