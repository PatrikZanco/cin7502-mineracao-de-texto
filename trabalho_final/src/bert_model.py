from sentence_transformers import SentenceTransformer
from src.utils import cosine_sim_matrix

def run_bert(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    return cosine_sim_matrix(embeddings, texts)
