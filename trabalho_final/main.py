from src.preprocess import load_samples
from src.tfidf_model import run_tfidf
from src.word2vec_model import run_word2vec
from src.bert_model import run_bert
from src.utils import plot_similarity
import time

texts = load_samples(file_path="trabalho_final/data/sample.txt") # adicionar path do txt com frases

tfidf_sim = run_tfidf(texts)
word2vec_sim = run_word2vec(texts)
bert_sim = run_bert(texts)

# 1. Gerar gráficos assincronamente
plot_similarity(tfidf_sim, "TF-IDF Similaridade", "tfidf_similarity.png")
plot_similarity(word2vec_sim, "Word2Vec Similaridade", "word2vec_similarity.png")
plot_similarity(bert_sim, "BERT Similaridade", "bert_similarity.png")

# 2. Calcular média de similaridades
results = {
    "TF-IDF": tfidf_sim.values.mean(),
    "Word2Vec": word2vec_sim.values.mean(),
    "BERT": bert_sim.values.mean()
}

print("\nMédia de similaridades entre frases:")
for k, v in results.items():
    print(f"{k}: {v:.3f}")

# 3. Esperar brevemente para garantir que threads terminem
time.sleep(3)
print("\n✅ Gráficos salvos em 'results/plots/'")

