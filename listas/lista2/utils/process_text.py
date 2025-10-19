from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import nltk
import re

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

nltk.download('vader_lexicon')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    tokens_formated =  ' '.join(tokens)
    return tokens_formated