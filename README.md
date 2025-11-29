# cin7502-mineracao-de-texto


Este repositório contém os códigos desenvolvidos nas questões de **Análise de Texto**, **Análise de sentimento** e **Redes Neurais**.


## Configuração do Ambiente

### Criar e ativar o ambiente virtual

#### **Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\activate # windowns
source venv/bin/activate # mac os e linux


pip install -r requirements.txt # baixar todas as dependencias do projeto
 
```
## Tutorial de execução da questão 3 b - Prova
- Após realizar o setup, rode na raiz do projeto:
```
python prova/questao-3.py
```
Resultado esperado:
```
Document d1 (length=5):
  gato     TF=0.200000  IDF=0.510826  TF-IDF=0.102165
  cachorro TF=0.000000  IDF=0.510826  TF-IDF=0.000000
  o        TF=0.200000  IDF=0.000000  TF-IDF=0.000000
  no       TF=0.200000  IDF=0.916291  TF-IDF=0.183258
  sofá     TF=0.200000  IDF=1.609438  TF-IDF=0.321888
  dorme    TF=0.200000  IDF=1.609438  TF-IDF=0.321888

Document d2 (length=6):
  gato     TF=0.000000  IDF=0.510826  TF-IDF=0.000000
  cachorro TF=0.166667  IDF=0.510826  TF-IDF=0.085138
  o        TF=0.333333  IDF=0.000000  TF-IDF=0.000000
  no       TF=0.000000  IDF=0.916291  TF-IDF=0.000000
  sofá     TF=0.000000  IDF=1.609438  TF-IDF=0.000000
  dorme    TF=0.000000  IDF=1.609438  TF-IDF=0.000000

Document d3 (length=5):
  gato     TF=0.000000  IDF=0.510826  TF-IDF=0.000000
  cachorro TF=0.000000  IDF=0.510826  TF-IDF=0.000000
  o        TF=0.200000  IDF=0.000000  TF-IDF=0.000000
  no       TF=0.000000  IDF=0.916291  TF-IDF=0.000000
  sofá     TF=0.000000  IDF=1.609438  TF-IDF=0.000000
  dorme    TF=0.000000  IDF=1.609438  TF-IDF=0.000000

Document d4 (length=7):
  gato     TF=0.142857  IDF=0.510826  TF-IDF=0.072975
  cachorro TF=0.142857  IDF=0.510826  TF-IDF=0.072975
  o        TF=0.285714  IDF=0.000000  TF-IDF=0.000000
  no       TF=0.000000  IDF=0.916291  TF-IDF=0.000000
  sofá     TF=0.000000  IDF=1.609438  TF-IDF=0.000000
  dorme    TF=0.000000  IDF=1.609438  TF-IDF=0.000000

Document d5 (length=8):
  gato     TF=0.125000  IDF=0.510826  TF-IDF=0.063853
  cachorro TF=0.125000  IDF=0.510826  TF-IDF=0.063853
  o        TF=0.125000  IDF=0.000000  TF-IDF=0.000000
  no       TF=0.125000  IDF=0.916291  TF-IDF=0.114536
  sofá     TF=0.000000  IDF=1.609438  TF-IDF=0.000000
  dorme    TF=0.000000  IDF=1.609438  TF-IDF=0.000000
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
