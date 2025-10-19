import pandas as pd
from pprint import pprint
from nltk.sentiment import SentimentIntensityAnalyzer

from utils.process_text import preprocess_text

df = pd.read_csv(
    '/data/Lista_de_frases.csv',  
    names=['texto', 'sentimento'],
    skiprows=1
)
df['texto_limpo'] = df['texto'].apply(preprocess_text)

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positivo'
    elif score['compound'] <= -0.05:
        return 'negativo'
    else:
        return 'neutro'

df['sentimento_previsto'] = df['texto_limpo'].apply(get_sentiment)

pprint(df[['texto', 'sentimento_previsto']])
