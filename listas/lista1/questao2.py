from pprint import pprint

docs = {
    "d1": "o gato está na casa",
    "d2": "o cachorro está no quintal",
    "d3": "o gato e o cachorro brincam"
}

# transforma cada frase em uma lista de palavras
tokenized_docs = {k: v.split() for k, v in docs.items()}

# cria vocabulário
vocab = sorted(set([term for doc in tokenized_docs.values() for term in doc]))

# matriz de frequências
bow = {}
for doc_id, terms in tokenized_docs.items():
    bow[doc_id] = [terms.count(term) for term in vocab]

print("===== Vocabulário =====")
print(vocab)

print("\n===== Vetores Bag of Words =====")
pprint(bow)

# Mostrar em formato de tabela simples
print("\n===== Tabela de Frequência =====")
header = ["Termo"] + list(docs.keys())
print("{:<12} {:<5} {:<5} {:<5}".format(*header))
for term in vocab:
    counts = [tokenized_docs[d].count(term) for d in docs]
    print("{:<12} {:<5} {:<5} {:<5}".format(term, *counts))
