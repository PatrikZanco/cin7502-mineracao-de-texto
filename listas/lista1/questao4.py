from pprint import pprint
from math import sqrt
from functools import reduce

docs = {
    "d1": "o gato está na casa",
    "d2": "o cachorro está no quintal",
    "d3": "o gato e o cachorro brincam"
}

dc = "o cachorro brinca no quintal"

tokenized_docs = dict(map(lambda kv: (kv[0], kv[1].split()), docs.items()))
query_terms = dc.split()

#usando set e reduce
vocab = sorted(reduce(lambda a, b: a.union(b),
                      map(lambda terms: set(terms), tokenized_docs.values())).union(set(query_terms)))

#  criar vetor
build_bow_vector = lambda terms: list(map(lambda term: terms.count(term), vocab))

bow_docs = dict(map(lambda kv: (kv[0], build_bow_vector(kv[1])), tokenized_docs.items()))
bow_query = build_bow_vector(query_terms)

dot = lambda v1, v2: sum(map(lambda x: x[0]*x[1], zip(v1, v2)))
norm = lambda v: sqrt(sum(map(lambda x: x**2, v)))
cosine_similarity = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2)) if norm(v1) and norm(v2) else 0

#Calcular similaridade entre consulta e documentos
similarities = dict(map(lambda kv: (kv[0], cosine_similarity(bow_query, kv[1])), bow_docs.items()))

ranking = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

print("\n===== Vocabulário =====")
pprint(vocab)

print("\n===== Vetores Bag of Words =====")
pprint(bow_docs)

print("\n===== Vetor do Documento de Consulta =====")
pprint(bow_query)

print("\n===== Similaridade do Cosseno =====")
pprint(similarities)

print("\n===== Ranking de Similaridade =====")
for doc, score in ranking:
    print(f"{doc}: {score:.4f}")