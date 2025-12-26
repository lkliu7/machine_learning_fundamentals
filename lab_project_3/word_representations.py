import re
from collections import Counter
from unidecode import unidecode
from scipy.sparse import csr_matrix
import numpy as np
import csv
import scipy.sparse.linalg
import time
import tracemalloc


CONFIG = {
    # Dimensions to truncate SVD.
    'svd_dims': [20, 50, 100],
    
    # Alternating algorithm iterations.
    'iters': 10,
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
    wordsim_words = set()
    for row in reader:
        wordsim_words.add(row[0].lower())
        wordsim_words.add(row[1].lower())

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
    V_SVD[dim] = v
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

error_alt = {}
error_SVD = {}
# Efficient Frobenius norm computation
def sparse_reconstruction_error(sparse_mat, U, S, V):
    """Compute ||sparse_mat - USV||_F without materializing dense difference"""
    # Reconstruction
    reconstructed = U @ np.diag(S) @ V
    
    # ||A||_F^2 for sparse matrix
    norm_A_sq = sparse_mat.data @ sparse_mat.data
    
    # 2*trace(A^T * reconstructed) - computed on sparse entries only
    trace_term = 2 * np.sum(sparse_mat.multiply(reconstructed).data)
    
    # ||reconstructed||_F^2
    norm_B_sq = np.sum(reconstructed ** 2)
    
    return np.sqrt(norm_A_sq - trace_term + norm_B_sq)

for dim in svd_dims:
    error_SVD[dim] = sparse_reconstruction_error(document_word_frequency_matrix, U_SVD[dim], S_SVD[dim], V_SVD[dim])
    error_alt[dim] = sparse_reconstruction_error(document_word_frequency_matrix, U_alt[dim], np.ones(dim), V_alt[dim].T)
    print(f'SVD error: {error_SVD[dim]}, alternating algorithm error: {error_alt[dim]}')

