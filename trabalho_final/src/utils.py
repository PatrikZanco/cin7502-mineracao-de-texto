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