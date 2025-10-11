from pprint import pprint
import math

docs = {
    "d1": "o gato está na casa",
    "d2": "o cachorro está no quintal",
    "d3": "o gato e o cachorro brincam"
}

# Documento de consulta
dc = "o cachorro brinca no quintal"

# Tokenização 
tokenized_docs = {k: v.split() for k, v in docs.items()}
query_terms = dc.split()

vocab = sorted(set([term for doc in tokenized_docs.values() for term in doc] + query_terms))

def build_bow_vector(terms, vocab):
    """Retorna o vetor de frequências (Bag of Words) para um documento"""
    return [terms.count(term) for term in vocab]

bow_docs = {doc_id: build_bow_vector(terms, vocab) for doc_id, terms in tokenized_docs.items()}
bow_query = build_bow_vector(query_terms, vocab)

def cosine_similarity(v1, v2):
    #calcular similaridade do cosseno
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

#Calcular similaridade
similarities = {doc_id: cosine_similarity(bow_query, vec) for doc_id, vec in bow_docs.items()}

#ordem decrescente 
ranking = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

print("\n===== Vocabulário =====")
pprint(vocab)

print("\n===== Vetores Bag of Words =====")
print("Documentos:")
pprint(bow_docs)
print("\nDocumento de consulta (dc):")
pprint(bow_query)

print("\n===== Similaridade do Cosseno =====")
pprint(similarities)

print("\n===== Ranking de Similaridade =====")
for doc_id, score in ranking:
    print(f"{doc_id}: {score:.4f}")
