import re
from collections import Counter
import unicodedata
from unidecode import unidecode

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

word_index = {word: idx for idx, word in enumerate(most_common_words)}