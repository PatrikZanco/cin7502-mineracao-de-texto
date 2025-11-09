import os
import time
from src.preprocess import load_samples
from src.tfidf_model import run_tfidf
from src.word2vec_model import run_word2vec
from src.bert_model import run_bert
from src.utils import plot_similarity
from src import style_classification


def process_similarity(texts, filename_prefix=""):
    """
    Gera matrizes e gráficos de similaridade usando TF-IDF, Word2Vec e BERT.
    """
    print(f"\n Iniciando análise de similaridade para: {filename_prefix}.txt\n")

    tfidf_sim = run_tfidf(texts)
    word2vec_sim = run_word2vec(texts)
    bert_sim = run_bert(texts)

    plots_dir = os.path.join("results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_similarity(tfidf_sim, f"TF-IDF Similaridade ({filename_prefix})",
                    f"{filename_prefix}_tfidf_similarity.png")
    plot_similarity(word2vec_sim, f"Word2Vec Similaridade ({filename_prefix})",
                    f"{filename_prefix}_word2vec_similarity.png")
    plot_similarity(bert_sim, f"BERT Similaridade ({filename_prefix})",
                    f"{filename_prefix}_bert_similarity.png")

    results = {
        "TF-IDF": tfidf_sim.values.mean(),
        "Word2Vec": word2vec_sim.values.mean(),
        "BERT": bert_sim.values.mean()
    }

    print("\n Média de similaridades entre frases:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")

    print(f"\n Gráficos de similaridade salvos em 'results/plots/' para {filename_prefix}\n")
    return results


def main():
    data_dir = "data"
    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    if not txt_files:
        print("⚠️ Nenhum arquivo .txt encontrado em /data/")
        return

    for file in txt_files:
        filepath = os.path.join(data_dir, file)
        filename_prefix = os.path.splitext(file)[0]
        print(f"\n===============================")
        print(f"PROCESSANDO ARQUIVO: {file}")
        print(f"===============================\n")

        texts = load_samples(filepath)

        process_similarity(texts, filename_prefix)

        print(f"\n Executando classificação de estilo para {file}...\n")
        style_classification.process_file(filepath)

        time.sleep(1)

    print("\n Pipeline completo finalizado com sucesso para todos os arquivos!\n")


if __name__ == "__main__":
    main()
