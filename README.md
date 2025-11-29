# cin7502-mineracao-de-texto


Este repositório contém os códigos desenvolvidos nas questões de **Análise de Texto**, **Análise de sentimento** e **Redes Neurais**.


## 🚀 Configuração do Ambiente

### 1️⃣ Criar e ativar o ambiente virtual

#### **Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\activate # windowns
source venv/bin/activate # mac os e linux


pip install -r requirements.txt # baixar todas as dependencias do projeto

```

## Tutorial de execução da questão 4 - Prova

- Após realizar o setup, rode na raiz do projeto:
```
python prova/questao-4-patrik.py
```
Resultado esperado:
```
Tokens após pré-processamento: ['entreg', 'rápid', 'produt', 'excel', 'qual']
Texto processado: entreg rápid produt excel qual

Vocabulário: ['entreg' 'excel' 'produt' 'qual' 'rápid']
Matriz de Embeddings (TF-IDF):
[[0.4472136 0.4472136 0.4472136 0.4472136 0.4472136]]

```
