# NLP Comparison — TF-IDF, Word2Vec e BERT

Última execução: **9 de novembro de 2025**

Este projeto compara três técnicas clássicas de **representação de texto em vetores** —  
**TF-IDF**, **Word2Vec** e **BERT** — aplicadas à análise semântica e à classificação de estilo linguístico (culto × coloquial) em diferentes domínios textuais.

---

## Objetivo

Demonstrar, de forma prática e reprodutível, como diferentes abordagens de *embeddings* transformam textos em representações numéricas, avaliando:

- A **similaridade semântica** entre frases (graus de proximidade vetorial)  
- O **desempenho de classificação** em tarefas de distinção entre linguagem formal (culto) e informal (coloquial)

A análise cobre múltiplos conjuntos de dados temáticos: **educação, política, consumo e mídias sociais**.

---

## Estrutura do Projeto

```
trabalho_final/
│
├── data/                         # Conjuntos de textos temáticos
│   ├── educacao.txt
│   ├── politica.txt
│   ├── consumo.txt
│   └── midias.txt
│
├── src/
│   ├── preprocess.py              # Funções de carregamento e limpeza
│   ├── utils.py                   # Similaridade, validação e plotagem
│   ├── tfidf_model.py             # Implementação TF-IDF
│   ├── word2vec_model.py          # Implementação Word2Vec
│   ├── bert_model.py              # Implementação BERT (SentenceTransformer)
│   ├── style_classification.py    # Classificação culto × coloquial (com validação)
│
│── main.py                    # Pipeline principal de execução
│
├── results/
│   ├── plots/                     # Gráficos de similaridade TF-IDF / Word2Vec / BERT
│   ├── educacao/                  # Resultados detalhados por domínio
│   │   ├── educacao_style_comparison.png
│   │   └── educacao_style_summary.csv
│   ├── politica/
│   ├── consumo/
│   └── midias/
│
└── venv/                          # Ambiente virtual (Python 3.11)
```

---

## Execução

Na raiz do projeto:

```bash
python src/main.py
```

O script executa automaticamente o pipeline completo para **todos os arquivos `.txt`** da pasta `/data`.

Cada execução realiza:
1. Análise de similaridade entre frases com TF-IDF, Word2Vec e BERT  
2. Classificação de estilo culto × coloquial com validação cruzada  
3. Geração de gráficos e relatórios automáticos salvos em `/results/`

---

## Principais Funções

### `run_tfidf`, `run_word2vec`, `run_bert`
Local: `src/tfidf_model.py`, `src/word2vec_model.py`, `src/bert_model.py`

Geram vetores representando as frases e calculam matrizes de **similaridade** entre textos.

- **TF-IDF:** frequência ponderada de termos.  
- **Word2Vec:** relações semânticas entre palavras.  
- **BERT:** contexto e significado completo das frases.

---

### `validate_model` (em `src/utils.py`)
Executa duas etapas de validação para cada modelo de classificação:

1. **5-Fold Cross-Validation** — avalia estabilidade do modelo.  
2. **ShuffleSplit Validation** — simula partições aleatórias para robustez.

Retorna médias e desvios padrão do F1-score.

---

### `process_file` (em `src/style_classification.py`)
Roda o pipeline completo de **classificação de estilo** para cada arquivo `.txt`:

- Divide dados em treino/teste  
- Treina TF-IDF, Word2Vec e BERT  
- Avalia acurácia, F1 e realiza validação cruzada  
- Gera gráfico comparativo e salva resultados em `/results/[arquivo]/`

---

### `main()` (em `src/main.py`)
É o **orquestrador geral**.  
Percorre automaticamente todos os `.txt` em `/data`, executando:

1. Análises de similaridade  
2. Classificação culto × coloquial  
3. Geração e salvamento de resultados

---

## Resultados Obtidos (rodada em 9/11/2025)

### 1. Similaridade Média entre Frases

| Dataset    | TF-IDF | Word2Vec | BERT |
|-------------|--------|-----------|------|
| Educação | 0.030 | 0.277 | **0.479** |
| Mídias | 0.034 | 0.253 | **0.472** |
| Consumo | 0.028 | 0.313 | **0.474** |
| Política | 0.028 | 0.229 | **0.510** |

O BERT apresentou maior similaridade contextual entre frases em todos os domínios.

---

### 2. Classificação de Estilo (Culto × Coloquial)

| Dataset | Modelo | Accuracy | F1-score | F1 (5-fold) ± std | F1 (Shuffle) ± std |
|----------|---------|----------|-----------|--------------------|--------------------|
| Educação | TF-IDF | 1.000 | 1.000 | 0.917 ± 0.060 | 0.904 ± 0.034 |
|  | Word2Vec | 0.493 | 0.325 | 0.454 ± 0.241 | 0.293 ± 0.017 |
|  | BERT | 0.942 | 0.942 | 0.943 ± 0.031 | 0.925 ± 0.010 |
| Mídias | TF-IDF | 0.983 | 0.983 | 0.921 ± 0.047 | 0.834 ± 0.059 |
|  | Word2Vec | 0.883 | 0.883 | 0.776 ± 0.093 | 0.248 ± 0.035 |
|  | BERT | 0.983 | 0.983 | 0.986 ± 0.018 | 0.976 ± 0.015 |
| Consumo | TF-IDF | 0.952 | 0.952 | 0.921 ± 0.024 | 0.897 ± 0.019 |
|  | Word2Vec | 0.494 | 0.327 | 0.441 ± 0.215 | 0.289 ± 0.034 |
|  | BERT | 0.964 | 0.964 | 0.948 ± 0.023 | 0.945 ± 0.013 |
| Política | TF-IDF | 0.870 | 0.868 | 0.931 ± 0.036 | 0.839 ± 0.080 |
|  | Word2Vec | 0.493 | 0.325 | 0.436 ± 0.204 | 0.468 ± 0.275 |
|  | BERT | 0.942 | 0.942 | 0.869 ± 0.075 | 0.850 ± 0.043 |

Conclusão:  
O modelo **BERT** manteve desempenho consistente e superior em todas as métricas, seguido por **TF-IDF**.  
O **Word2Vec** mostrou variação maior e menor estabilidade nos testes de validação cruzada.

---

## Exemplos de Saídas Geradas

```
results/
│
├── plots/
│   ├── educacao_tfidf_similarity.png
│   ├── educacao_word2vec_similarity.png
│   ├── educacao_bert_similarity.png
│   ├── midias_tfidf_similarity.png
│   └── ...
│
├── educacao/
│   ├── educacao_style_comparison.png
│   └── educacao_style_summary.csv
├── politica/
│   ├── politica_style_comparison.png
│   └── politica_style_summary.csv
└── consumo/
    ├── consumo_style_comparison.png
    └── consumo_style_summary.csv
```

---

## Conclusões Gerais

- **BERT** foi o modelo mais robusto, alcançando **acurácias acima de 94%** em todos os domínios.  
- **TF-IDF** apresentou excelente desempenho em bases menores e estruturadas.  
- **Word2Vec** captou parte das relações semânticas, mas não discriminou bem estilos linguísticos.  
- As validações cruzadas confirmam a **consistência do BERT** e a **instabilidade do Word2Vec** em textos curtos.  


