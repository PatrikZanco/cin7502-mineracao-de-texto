import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from src.utils import validate_model



def plot_bars(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    labels = list(results.keys())
    accs = [results[m]["acc"] for m in labels]
    f1s = [results[m]["f1"] for m in labels]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, accs, width, label="Accuracy")
    ax.bar(x + width/2, f1s, width, label="F1-score")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Comparação — Classificação de estilo (culto vs coloquial)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f" Gráfico salvo em {save_path}")



def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"{name:<8} | Accuracy: {acc:.3f} | F1: {f1:.3f}")
    return {"acc": acc, "f1": f1}


def load_style_data(filepath):
    """Extrai frases e rótulos (culto/coloquial) de um arquivo .txt."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    data = []
    for line in lines:
        if line.startswith("Culto:"):
            data.append({"label": 0, "text": line.replace("Culto:", "").strip()})
        elif line.startswith("Coloquial:"):
            data.append({"label": 1, "text": line.replace("Coloquial:", "").strip()})
    return pd.DataFrame(data)



def process_file(filepath):
    """Executa o pipeline completo de classificação de estilo para um arquivo .txt."""
    filename = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n Processando arquivo: {filename}.txt")

    df = load_style_data(filepath)
    if df.empty:
        print(f" Arquivo {filename}.txt não contém dados válidos (culto/coloquial).")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
    )
    print(f"Treino: {len(X_train)}, Teste: {len(X_test)}")

    results = {}

    print("\nTreinando TF-IDF...")
    t0 = time.time()
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    clf_tfidf = LogisticRegression(max_iter=1000).fit(X_train_tfidf, y_train)
    pred_tfidf = clf_tfidf.predict(X_test_tfidf)
    results["TF-IDF"] = print_metrics("TF-IDF", y_test, pred_tfidf)
    results["TF-IDF"]["validation"] = validate_model(clf_tfidf, X_train_tfidf, y_train, name="TF-IDF")
    print(f" Tempo: {time.time()-t0:.2f}s")

    print("\nTreinando Word2Vec...")
    t0 = time.time()
    sentences = [t.split() for t in X_train]
    w2v = Word2Vec(sentences, vector_size=100, min_count=1, window=3)

    def sent_vec(s):
        words = [w for w in s.split() if w in w2v.wv]
        if not words:
            return np.zeros(w2v.vector_size)
        return np.mean(w2v.wv[words], axis=0)

    X_train_w2v = np.vstack([sent_vec(s) for s in X_train])
    X_test_w2v = np.vstack([sent_vec(s) for s in X_test])
    clf_w2v = LogisticRegression(max_iter=1000).fit(X_train_w2v, y_train)
    pred_w2v = clf_w2v.predict(X_test_w2v)
    results["Word2Vec"] = print_metrics("Word2Vec", y_test, pred_w2v)
    results["Word2Vec"]["validation"] = validate_model(clf_w2v, X_train_w2v, y_train, name="Word2Vec")
    print(f" Tempo: {time.time()-t0:.2f}s")

    print("\nTreinando BERT embeddings...")
    t0 = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X_train_bert = model.encode(X_train.tolist(), show_progress_bar=False)
    X_test_bert = model.encode(X_test.tolist(), show_progress_bar=False)
    clf_bert = LogisticRegression(max_iter=1000).fit(X_train_bert, y_train)
    pred_bert = clf_bert.predict(X_test_bert)
    results["BERT"] = print_metrics("BERT", y_test, pred_bert)
    results["BERT"]["validation"] = validate_model(clf_bert, X_train_bert, y_train, name="BERT")
    print(f" Tempo: {time.time()-t0:.2f}s")

    results_dir = os.path.join("results", filename)
    os.makedirs(results_dir, exist_ok=True)
    plot_bars(results, os.path.join(results_dir, f"{filename}_style_comparison.png"))

    summary = pd.DataFrame({
        name: {
            "Accuracy": res["acc"],
            "F1": res["f1"],
            "CV_mean": res["validation"]["cv_mean"],
            "CV_std": res["validation"]["cv_std"],
            "Shuffle_mean": res["validation"]["sh_mean"],
            "Shuffle_std": res["validation"]["sh_std"]
        }
        for name, res in results.items()
    }).T

    summary.to_csv(os.path.join(results_dir, f"{filename}_style_summary.csv"), index=True)
    print(f" Resultados salvos em {results_dir}/")



