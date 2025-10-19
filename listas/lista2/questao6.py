import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import matplotlib.pyplot as plt

from textblob import TextBlob
from utils.process_text import preprocess_text


headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
}

url_home = "https://br.cointelegraph.com/category/analysis"


def get_links_pages(url: str, headers: dict):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html_data = response.text
        return parser_get_urls_tag(html_data, url)
    except Exception as e:
        print(f"Erro ao acessar {url}: {e}")
        return []


def parser_get_urls_tag(html_data: str, base_url: str):
    soup = BeautifulSoup(html_data, 'html.parser')
    urls_custom = []
    seletor = soup.select('article.post-card-inline a')
    for link in seletor:
        href = link.get('href')
        if href:
            urls_custom.append(urljoin(base_url, href))
    return urls_custom


def get_page_html(url: str):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Erro ao acessar {url}: {e}")
        return None


def extract_post_content(html_data: str):
    soup = BeautifulSoup(html_data, 'html.parser')
    div_content = soup.select_one('div.post-content.relative.post-content_margin')
    if div_content:
        return div_content.get_text(separator=" ", strip=True)
    return None


def sentiment_analysis(text):
    """Classifica sentimento como positivo, negativo ou neutro"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return "positivo"
    elif polarity < -0.1:
        return "negativo"
    else:
        return "neutro"


def main():
    print("Coletando links...")
    urls = get_links_pages(url = url_home, headers = headers)
    print(f"{len(urls)} links encontrados.\n")

    dados = []
    for i, url in enumerate(urls[:10], 1):
        print(f"Extraindo artigo {i}: {url}")
        html = get_page_html(url)
        if html:
            texto = extract_post_content(html)
            if texto:
                texto_limpo = preprocess_text(texto)
                sentimento = sentiment_analysis(texto_limpo)
                dados.append({
                    "url": url,
                    "texto_original": texto,
                    "texto_preprocessado": texto_limpo,
                    "sentimento": sentimento
                })

    df = pd.DataFrame(dados)
    df.to_csv("resultados.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(6, 4))
    df["sentimento"].value_counts().plot(kind="bar", title="Distribuição dos Sentimentos")
    plt.xlabel("Sentimento")
    plt.ylabel("Quantidade")
    plt.show()

    print(" Análise concluída. Resultados salvos em 'resultados.csv'.")

if __name__ == "__main__":
    main()
