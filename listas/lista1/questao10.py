import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
nltk.download('rslp')

data = {
    "Documento": [f"d{i}" for i in range(1, 11)],
    "Texto": [
        "O filme foi incrível, adorei cada cena.",
        "O produto chegou quebrado e me decepcionou.",
        "Excelente atendimento, fiquei muito satisfeito.",
        "A comida estava fria e sem sabor.",
        "Gostei bastante da qualidade e do design.",
        "O serviço é péssimo, nunca mais volto.",
        "Uma experiência maravilhosa, recomendo a todos.",
        "O aplicativo trava o tempo todo, é horrível.",
        "Achei o hotel confortável e bem localizado.",
        "O atendimento foi lento e desorganizado."
    ],
    "Sentimento": [
        "Positivo", "Negativo", "Positivo", "Negativo", "Positivo",
        "Negativo", "Positivo", "Negativo", "Positivo", "Negativo"
    ]
}

df = pd.DataFrame(data)

stop_words = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()

def preprocess(text):
    text = text.lower()
    # tokenização simples
    tokens = nltk.word_tokenize(text)
    # remove stopwords e pontuações
    filtered = [t for t in tokens if t.isalpha() and t not in stop_words]
    # stemming
    stemmed = [stemmer.stem(t) for t in filtered]
    return " ".join(stemmed)

df["Texto_limpo"] = df["Texto"].apply(preprocess)

#Vetorização (TF-IDF)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Texto_limpo"])
y = df["Sentimento"].map({"Positivo": 1, "Negativo": 0})

# divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo de Rede Neural (MLP)
mlp = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)

# Avaliação
y_pred = mlp.predict(X_test)
print("\n===== AVALIAÇÃO =====")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"]))

# Teste com novos textos
novos_textos = [
    "O atendimento foi excelente, muito rápido.",
    "O celular travou e quebrou logo depois de chegar.",
    "A comida estava deliciosa e bem temperada."
]

novos_textos_proc = [preprocess(t) for t in novos_textos]
X_novos = vectorizer.transform(novos_textos_proc)
preds = mlp.predict(X_novos)

print("\n===== CLASSIFICAÇÃO DE NOVOS TEXTOS =====")
for texto, pred in zip(novos_textos, preds):
    print(f"'{texto}' -> {'Positivo' if pred == 1 else 'Negativo'}")
