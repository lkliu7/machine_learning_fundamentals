import re
from collections import Counter
from unidecode import unidecode
from scipy.sparse import csr_matrix
import numpy as np
import csv
import scipy.sparse.linalg
import time
import tracemalloc
import matplotlib.pyplot as plt
import scipy.stats


CONFIG = {
    # Dimensions to truncate SVD.
    'svd_dims': [20, 50, 100],
    
    # Alternating algorithm iterations.
    'iters': 10,

    # Learning rate for SGD.
    'learning_rate': 1e-2,

    # SGD steps.
    'SGD_iters': 10 ** 6,
}

with open('enwik8', 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(r'<[^>]+>', '', text)
text = re.sub(r'&[^;]+;', '', text)
text = re.sub(r'#[^;]+;', '', text)
text = re.sub(r'http[^\s]+', '', text)

text = unidecode(text)
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
text = text.lower()

words = text.split()
word_counts = Counter(words)
most_common_words_counts = word_counts.most_common(10000)
most_common_words = [t[0] for t in most_common_words_counts]

with open('wordsim353crowd.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    wordsim_pairs = [(w1.lower(), w2.lower(), float(score)) for w1, w2, score in reader]

wordsim_words = {w for pair in wordsim_pairs for w in pair[:2]}

most_common_set = set(most_common_words)
new_words = wordsim_words - most_common_set
most_common_words.extend(new_words)

word_index = {word: idx for idx, word in enumerate(most_common_words)}

row_indices = []
col_indices = []
values = []

row = 0
for line in text.split('\n'):
    words = line.split()
    if len(words) <= 3:
        continue

    word_ids = [word_index[w] for w in words if w in word_index]
    word_counts = Counter(word_ids)
    for idx, count in word_counts.items():
        row_indices.append(row)
        col_indices.append(idx)
        values.append(count)
    row += 1

document_word_frequency_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(row, len(most_common_words)))
U_SVD = {}
S_SVD = {}
V_SVD = {}
svd_dims = CONFIG['svd_dims']
for dim in svd_dims:
    tracemalloc.start()
    start_time = time.perf_counter()
    u, S, v = scipy.sparse.linalg.svds(document_word_frequency_matrix, dim)
    time_taken = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    U_SVD[dim] = u
    S_SVD[dim] = S
    V_SVD[dim] = v.T
    print(f'SVD for dimension {dim} finished in {time_taken:.3f} seconds, using {peak} bytes of memory')

def matrix_factorization(mat, k, l1, l2, steps):
    n, m = mat.shape
    V = np.random.random((m, k))
    for _ in range(steps):
        A = l1 * np.eye(k) + V.T @ V
        A = np.linalg.inv(A)
        U = V @ A.T
        U = mat @ U
        B = l2 * np.eye(k) + U.T @ U
        B = np.linalg.inv(B)
        V = B @ U.T
        V = V @ mat
        V = V.T
    return U, V

U_alt = {}
V_alt = {}
steps = CONFIG['iters']
for dim in svd_dims:
    tracemalloc.start()
    start_time = time.perf_counter()
    U, V = matrix_factorization(document_word_frequency_matrix, dim, 1, 1, steps)
    time_taken = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    U_alt[dim] = U
    V_alt[dim] = V
    print(f'Alternating algorithm for dimension {dim} finished in {time_taken:.3f} seconds, using {peak} bytes of memory')

def matrix_factorization_SGD(mat, k, l1, l2, lr, steps):
    n, m = mat.shape
    V = np.random.random((m, k))
    U = np.random.random((n, k))
    for _ in range(steps):
        i, j = np.random.randint((n, m))
        u = U[i]
        v = V[j]
        t = -2 * (mat[i,j] - u.dot(v))
        du = t * v + 2 * l1 * u
        dv = t * u + 2 * l2 * v
        u -= lr * du
        v -= lr * dv
        U[i] = u
        V[j] = v
    return U, V

U_SGD = {}
V_SGD = {}
steps = CONFIG['SGD_iters']
lr = CONFIG['learning_rate']
for dim in svd_dims:
    tracemalloc.start()
    start_time = time.perf_counter()
    U, V = matrix_factorization_SGD(document_word_frequency_matrix, dim, 1, 1, lr, steps)
    time_taken = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    U_SGD[dim] = U
    V_SGD[dim] = V
    print(f'SGD matrix factorization for dimension {dim} finished in {time_taken:.3f} seconds, using {peak} bytes of memory')

'''
N.B.: The following reconstruction error computation is highly memory-intensive.
Uncomment it if you are sure you want to run it.
'''

'''
error_alt = {}
error_SGD = {}
error_SVD = {}
# Efficient Frobenius norm computation
def sparse_reconstruction_error(sparse_mat, U, S, V):
    """Compute ||sparse_mat - USV||_F without materializing dense difference"""
    # Reconstruction
    reconstructed = U @ np.diag(S) @ V.T
    
    # ||A||_F^2 for sparse matrix
    norm_A_sq = sparse_mat.data @ sparse_mat.data
    
    # 2*trace(A^T * reconstructed) - computed on sparse entries only
    trace_term = 2 * np.sum(sparse_mat.multiply(reconstructed).data)
    
    # ||reconstructed||_F^2
    norm_B_sq = np.sum(reconstructed ** 2)
    
    return np.sqrt(norm_A_sq - trace_term + norm_B_sq)

for dim in svd_dims:
    error_SVD[dim] = sparse_reconstruction_error(document_word_frequency_matrix, U_SVD[dim], S_SVD[dim], V_SVD[dim])
    error_alt[dim] = sparse_reconstruction_error(document_word_frequency_matrix, U_alt[dim], np.ones(dim), V_alt[dim])
    error_SGD[dim] = sparse_reconstruction_error(document_word_frequency_matrix, U_SGD[dim], np.ones(dim), V_SGD[dim])
    print(f'SVD error: {error_SVD[dim]}, alternating algorithm error: {error_alt[dim]}, SGD algorithm error: {error_SGD[dim]}')

'''

def cosine_similarity(u, v):
    return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))

cosine_vs_human = {}
corr = {}
methods = {
    "alt": V_alt,
    "sgd": V_SGD,
    "svd": V_SVD,
}

for method_name, V_dict in methods.items():
    cosine_vs_human[method_name] = {}
    corr[method_name] = {}

    for dim in svd_dims:
        V = V_dict[dim]

        pairs = [
            (
                cosine_similarity(
                    V[word_index[w1]],
                    V[word_index[w2]]
                ),
                score
            )
            for w1, w2, score in wordsim_pairs
            if w1 in word_index and w2 in word_index
        ]

        cosine_vals, human_scores = zip(*pairs)
        r, _ = scipy.stats.pearsonr(cosine_vals, human_scores)

        cosine_vs_human[method_name][dim] = pairs
        corr[method_name][dim] = r

fig, axes = plt.subplots(
    nrows=len(methods),
    ncols=len(svd_dims),
    figsize=(4 * len(svd_dims), 4 * len(methods)),
    sharex=True,
    sharey=True
)

for row, method in enumerate(methods):
    for col, dim in enumerate(svd_dims):
        ax = axes[row, col]

        cosine_vals, human_scores = zip(*cosine_vs_human[method][dim])
        r = corr[method][dim]

        ax.scatter(cosine_vals, human_scores, alpha=0.4)
        ax.set_title(f"{method}, k={dim}\nr={r:.3f}")

        if row == len(methods) - 1:
            ax.set_xlabel("Cosine similarity")
        if col == 0:
            ax.set_ylabel("Human score")

plt.tight_layout()
plt.show()