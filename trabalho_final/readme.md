# NLP Comparison — TF-IDF, Word2Vec e BERT

Este projeto compara três técnicas clássicas de **representação de texto em vetores**:  
**TF-IDF**, **Word2Vec** e **BERT**, avaliando suas similaridades entre frases e desempenho semântico.

---

## Objetivo

Demonstrar, de forma prática e reprodutível, como diferentes abordagens de *embeddings* transformam textos em vetores numéricos e como elas captam o significado das frases de maneiras distintas.

---


## Estrutura de Diretórios

````
│
├── data/
│ └── samples.txt               # Frases de exemplo
│
├── src/
│ ├── preprocess.py             # Funções de limpeza e carregamento de texto
│ ├── utils.py                  # Funções de similaridade e visualização
│ ├── tfidf_model.py            # Implementação e métricas TF-IDF
│ ├── word2vec_model.py         # Implementação e métricas Word2Vec
│ ├── bert_model.py             # Implementação e métricas BERT
│
├── results/
│ └── plots/                    # Gráficos comparativos
│
└── main.py                     # Script principal de execução
```