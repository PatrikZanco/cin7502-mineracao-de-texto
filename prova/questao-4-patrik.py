from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer

# Downloads necessários
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')


def preprocess_text(text):
    # a. Converte para minúsculas
    text = text.lower()

    # b. Remove pontuações
    text = re.sub(r'[^\w\s]', '', text)

    # c. Tokeniza
    tokens = word_tokenize(text, language='portuguese')

    # d. Remove stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens = [w for w in tokens if w not in stop_words]

    # e. Aplica RSLPStemmer
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(w) for w in tokens]

    # Retorna string formatada e tokens
    return tokens, " ".join(tokens)


def build_embedding_matrix(processed_texts):
    """
    f. Cria matriz de embeddings usando TF-IDF.
    processed_texts: lista com textos já pré-processados (strings)
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    return X.toarray(), vectorizer.get_feature_names_out()


if __name__ == "__main__":
    text = "Entrega rápida e produto de excelente qualidade!"

    # Pré-processamento
    tokens, processed_string = preprocess_text(text)
    print("Tokens após pré-processamento:", tokens)
    print("Texto processado:", processed_string)

    # Matriz de embeddings
    matrix, vocab = build_embedding_matrix([processed_string])
    print("\nVocabulário:", vocab)
    print("Matriz de Embeddings (TF-IDF):")
    print(matrix)
