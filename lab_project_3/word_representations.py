import re
from collections import Counter
import unicodedata
from unidecode import unidecode
from scipy.sparse import csr_matrix
import numpy as np
import csv

with open('enwik8', 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(r'<[^>]+>', '', text)
text = re.sub(r'&[^;]+;', '', text)
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