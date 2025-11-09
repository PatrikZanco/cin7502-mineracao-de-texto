from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import cosine_sim_matrix

def tfidf_vectors(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return X.toarray()

def run_tfidf(texts):
    vectors = tfidf_vectors(texts)
    return cosine_sim_matrix(vectors, texts)
