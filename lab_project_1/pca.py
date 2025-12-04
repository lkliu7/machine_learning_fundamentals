import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import gzip

# MARK: Configuration
CONFIG = {
    'digits': range(10),
    'variance_threshold': 0.98,
    'lda_shift': 1e-4,
    'pca_dims': [2, 10, 50, 100, 200, 300],
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

def get_images_for_digit(n):
    'Get images for given labels.'
    return train_images[train_labels == n]

# Choose digits to consider.

digits = CONFIG['digits']
digits = list(set(digits) & set(range(10)))
if not digits:
    raise ValueError('no valid digits in config')
data = np.vstack([get_images_for_digit(n) for n in digits])

# MARK: Computation

def mean_vec(images):
    'Calculate mean vector of a set of images (given as 2D arrays).'
    total = np.zeros(784, dtype=np.float32)
    for img in images:
        total += img.flatten()
    return total / len(images)

def class_mean(n):
    'Mean vector for images with a given label.'
    return mean_vec(get_images_for_digit(n))

# Calculation mean vector for all images considered.

data_mean = mean_vec(data)

def covariance_matrix(images):
    'Calculate covariance matrix of a set of images (given as 2D arrays).'
    mean = mean_vec(images)
    total = np.zeros([784,784], dtype=np.float32)
    for img in images:
        vec = img.flatten() - mean
        total += np.outer(vec, vec)
    return total / len(images)

def class_cov_matrix(n):
    'Covariance matrix for images with a given label.'
    return covariance_matrix(get_images_for_digit(n))

# Calculate covariance matrix for all images considered.

S = covariance_matrix(data)

# Diagonalize covariance matrix.

S_vals, S_vecs = np.linalg.eigh(S)
idx = np.argsort(S_vals.real)[::-1]
S_vals = S_vals[idx]
S_vecs = S_vecs[:,idx]

def A(n):
    'Get matrix of eigenvectors corresponding to largest eigenvalues.'
    return S_vecs[:,:n].transpose()

# Calculate total distortion error. 
# Benign warning messages expected on first call; 
# known bug with np.matmul and np.matvec on Apple Silicon. Results not affected.

def total_distortion_error(m):
    mat = A(m)
    double = np.matmul(mat.transpose(), mat)
    bias = data_mean - np.matvec(double, data_mean)
    total = 0
    for img in data:
        vec = img.flatten()
        total += np.linalg.vector_norm(vec - np.matvec(double, vec) - bias) ** 2
    return total

def lda_matrix(dim, shift):
    
    between_class_scatter_matrix = np.zeros([784,784], dtype=np.float32)
    for n in digits:
        card = len(get_images_for_digit(n))
        vec = class_mean(n) - data_mean
        between_class_scatter_matrix += card * np.outer(vec, vec)

    total_scatter_matrix = sum([class_cov_matrix(n) for n in digits])
    mat = np.matmul(np.linalg.inv(total_scatter_matrix + shift * np.eye(784)), between_class_scatter_matrix)
    vals, vecs = np.linalg.eig(mat)
    idx = np.argsort(vals.real)[::-1]
    vals = vals[idx]
    vecs = vecs[:,idx]
    return vecs[:,:dim].transpose()

def pca_projected_images(n, dim):
    images = get_images_for_digit(n)
    pca = A(dim)
    return [np.matvec(pca, img.flatten()) for img in images]

def lda_projected_images(n, dim):
    images = get_images_for_digit(n)
    lda = lda_matrix(dim, CONFIG['lda_shift'])
    return [np.matvec(lda, img.flatten()) for img in images]

# MARK: Visualization
# Plot covariance matrix eigenvalues in descending order.

plt.scatter(range(len(S_vals)), S_vals)
plt.xlabel('index')
plt.ylabel('eigenvalue')
plt.title('Eigenvalues of total covariance matrix')
plt.show()

# Find the number of dimensions needed to keep 98% of total variance in the data.

variances = S_vals.cumsum()
variance_threshold = CONFIG['variance_threshold']
eig_threshold = np.argmax(variances >= variance_threshold * variances[-1])
print(f'{variance_threshold} of variance captured with {eig_threshold + 1} eigenvalues')

# Plot total distortion error with number of principal components kept.

components_list = CONFIG['pca_dims']
plt.scatter(components_list, [total_distortion_error(m) for m in components_list])
plt.xlabel('PCA components')
plt.ylabel('total distortion error')
plt.title('Total distortion error for number of PCA vectors')
plt.show()

# Plot PCA projections in 2D.

for n in digits:
    points = np.transpose(pca_projected_images(n, 2))
    plt.scatter(points[0], points[1], s=2)
plt.xlabel('component along PCA vector 1')
plt.ylabel('component along PCA vector 2')
plt.title('PCA projections of digits.')
plt.show()

# Plot LDA projections in 2D.

for n in digits:
    points = np.transpose(lda_projected_images(n, 2))
    plt.scatter(points[0].real, points[1].real, s=2)
plt.xlabel('component along LDA vector 1')
plt.ylabel('component along LDA vector 2')
plt.title('LDA projections of digits.')
plt.show()