import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")  # evita abrir janelas no macOS
import matplotlib.pyplot as plt
import os
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import f1_score
import time


def cosine_sim_matrix(vectors, texts):
    sim = cosine_similarity(vectors)
    return pd.DataFrame(sim, index=texts, columns=texts)

def _plot_thread(sim_df, title, save_path):
    """
    thread-safe: usa objetos Figure independentes
    """
    fig = Figure(figsize=(15, 12))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(sim_df, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(sim_df.columns)))
    ax.set_xticklabels(sim_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(sim_df.index)))
    ax.set_yticklabels(sim_df.index, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    fig.clear()


def plot_similarity(sim_df, title, filename):
    save_path = os.path.join("results", "plots", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    thread = threading.Thread(target=_plot_thread, args=(sim_df, title, save_path))
    thread.start()
    print(f"Gerando gráfico '{filename}' em background...")
    return thread



def validate_model(model, X, y, name="Modelo", cv_splits=5):
    """
    Realiza validação cruzada (5-fold) e múltiplos splits aleatórios.
    Retorna um dicionário contendo médias e desvios.
    """
    print(f"\n Validando {name}...")

    if hasattr(X, "toarray"):
        X_array = X.toarray()
    else:
        X_array = np.array(X)
    y_array = np.array(y)

    scores = cross_val_score(model, X_array, y_array, cv=cv_splits, scoring="f1_weighted")
    cv_mean = np.mean(scores)
    cv_std = np.std(scores)
    print(f"  (5-fold) F1 médio: {cv_mean:.3f} ± {cv_std:.3f}")

    rs = ShuffleSplit(n_splits=cv_splits, test_size=0.3, random_state=42)
    shuffle_scores = []

    for train_idx, test_idx in rs.split(X_array):
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        clf = model.__class__(**model.get_params()).fit(X_train, y_train)
        pred = clf.predict(X_test)
        shuffle_scores.append(f1_score(y_test, pred, average="weighted"))

    sh_mean = np.mean(shuffle_scores)
    sh_std = np.std(shuffle_scores)
    print(f"  (ShuffleSplit) F1 médio: {sh_mean:.3f} ± {sh_std:.3f}")

    return {
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "sh_mean": sh_mean,
        "sh_std": sh_std
    }
