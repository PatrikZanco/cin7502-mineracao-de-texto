from pprint import pprint
from math import sqrt
from functools import reduce
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# funcao utils
def build_bow_vector(terms, vocab):
    return [terms.count(term) for term in vocab]

def dot(v1, v2):
    return sum(map(lambda x: x[0]*x[1], zip(v1, v2)))

def norm(v):
    return sqrt(sum(map(lambda x: x**2, v)))

def cosine_similarity(v1, v2):
    return dot(v1, v2) / (norm(v1)*norm(v2)) if norm(v1) and norm(v2) else 0

# -------------------------------------------------------------
# === Corpus e consulta ===
# -------------------------------------------------------------

docs = {
    "d1": "o gato está na casa",
    "d2": "o cachorro está no quintal",
    "d3": "o gato e o cachorro brincam"
}

dc = "o cachorro brinca no quintal"

#  VERSÃO 1: Sem NLTK (mesma da questão 4) 

tokenized_docs_v1 = dict(map(lambda kv: (kv[0], kv[1].split()), docs.items()))
query_terms_v1 = dc.split()

vocab_v1 = sorted(reduce(lambda a, b: a.union(b),
                         map(lambda t: set(t), tokenized_docs_v1.values())).union(set(query_terms_v1)))

bow_docs_v1 = dict(map(lambda kv: (kv[0], build_bow_vector(kv[1], vocab_v1)), tokenized_docs_v1.items()))
bow_query_v1 = build_bow_vector(query_terms_v1, vocab_v1)

similarities_v1 = dict(map(lambda kv: (kv[0], cosine_similarity(bow_query_v1, kv[1])), bow_docs_v1.items()))
ranking_v1 = sorted(similarities_v1.items(), key=lambda x: x[1], reverse=True)


# === VERSÃO 2: Com NLTK (tokenização, stopwords, stemming, lematização) 

# Configurações de processamento
stop_words = set(stopwords.words("portuguese"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [t for t in tokens if t.isalpha() and t not in stop_words]
    stemmed = [stemmer.stem(t) for t in filtered]
    lemmatized = [lemmatizer.lemmatize(t) for t in stemmed]
    return lemmatized

tokenized_docs_v2 = dict(map(lambda kv: (kv[0], preprocess(kv[1])), docs.items()))
query_terms_v2 = preprocess(dc)

vocab_v2 = sorted(reduce(lambda a, b: a.union(b),
                         map(lambda t: set(t), tokenized_docs_v2.values())).union(set(query_terms_v2)))

bow_docs_v2 = dict(map(lambda kv: (kv[0], build_bow_vector(kv[1], vocab_v2)), tokenized_docs_v2.items()))
bow_query_v2 = build_bow_vector(query_terms_v2, vocab_v2)

similarities_v2 = dict(map(lambda kv: (kv[0], cosine_similarity(bow_query_v2, kv[1])), bow_docs_v2.items()))
ranking_v2 = sorted(similarities_v2.items(), key=lambda x: x[1], reverse=True)


print("\n===== VERSÃO 1 (Sem NLTK) =====")
print("Vocabulário:")
pprint(vocab_v1)
print("\nRanking de Similaridade:")
for d, s in ranking_v1:
    print(f"{d}: {s:.4f}")

print("\n===== VERSÃO 2 (Com NLTK: stopwords + stemming + lemmatização) =====")
print("Vocabulário:")
pprint(vocab_v2)
print("\nRanking de Similaridade:")
for d, s in ranking_v2:
    print(f"{d}: {s:.4f}")
