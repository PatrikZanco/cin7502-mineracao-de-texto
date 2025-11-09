from gensim.models import Word2Vec
import numpy as np
from src.utils import cosine_sim_matrix

def train_word2vec(sentences, vector_size=50, window=3):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=1)
    return model

def sentence_vector(model, sentence):
    words = [w for w in sentence if w in model.wv]
    if not words:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[words], axis=0)

def run_word2vec(texts):
    sentences = [t.split() for t in texts]
    model = train_word2vec(sentences)
    vectors = np.array([sentence_vector(model, s) for s in sentences])
    return cosine_sim_matrix(vectors, texts)
