import math
from pprint import pprint

# corpus
docs = {
    "d1": "o gato está no telhado",
    "d2": "o cachorro está no quintal",
    "d3": "o gato e o cachorro brincam juntos"
}

# Tokenização simples
tokenized_docs = {k: v.split() for k, v in docs.items()}

# calcula do TF 
def compute_tf(doc):
    tf = {}
    total_terms = len(doc)
    for term in doc:
        tf[term] = tf.get(term, 0) + 1
    for term in tf:
        tf[term] /= total_terms
    return tf

tf_values = {k: compute_tf(v) for k, v in tokenized_docs.items()}

# calcula IDF
N = len(tokenized_docs)
all_terms = set([term for doc in tokenized_docs.values() for term in doc])
idf = {}
for term in all_terms:
    df = sum(1 for doc in tokenized_docs.values() if term in doc)
    idf[term] = math.log10(N / df)

# calcula TF-IDF
tfidf = {}
for doc_id, tf in tf_values.items():
    tfidf[doc_id] = {term: tf[term] * idf[term] for term in tf}

print("\n===== TF (Term Frequency) =====")
pprint(tf_values)

print("\n===== IDF (Inverse Document Frequency) =====")
pprint(idf)

print("\n===== TF-IDF =====")
pprint(tfidf)
