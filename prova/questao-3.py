import math
from collections import Counter, defaultdict
import re

corpus = {
    "d1": "o gato dorme no sofá",
    "d2": "o cachorro late para o carteiro",
    "d3": "o pássaro canta na árvore",
    "d4": "o gato e o cachorro comem juntos",
    "d5": "o cachorro corre atrás do gato no quintal"
}

docs_tokens = {}
for doc_id, text in corpus.items():
    text_proc = re.sub(r'[^\w\sáéíóúãõâêîôûàçÁÉÍÓÚÃÕÂÊÎÔÛÀÇ]', '', text.lower())
    tokens = text_proc.split()
    docs_tokens[doc_id] = tokens

vocab = sorted({t for tokens in docs_tokens.values() for t in tokens})
term_counts = {doc: Counter(tokens) for doc, tokens in docs_tokens.items()}
doc_lengths = {doc: sum(cnt.values()) for doc, cnt in term_counts.items()}

df = {term: sum(1 for tokens in docs_tokens.values() if term in tokens) for term in vocab}
N = len(corpus)

idf = {term: math.log(N / df[term]) for term in vocab}

tf = defaultdict(dict)
tfidf = defaultdict(dict)
for doc in corpus:
    n = doc_lengths[doc]
    for term in vocab:
        f_td = term_counts[doc].get(term, 0)
        tf_val = f_td / n if n > 0 else 0.0
        tf[doc][term] = tf_val
        tfidf[doc][term] = tf_val * idf[term]

for doc in sorted(corpus.keys()):
    print(f"\nDocument {doc} (length={doc_lengths[doc]}):")
    for term in ['gato', 'cachorro', 'o', 'no', 'sofá', 'dorme']:
        if term in vocab:
            print(f"  {term:8} TF={tf[doc][term]:.6f}  IDF={idf[term]:.6f}  TF-IDF={tfidf[doc][term]:.6f}")
