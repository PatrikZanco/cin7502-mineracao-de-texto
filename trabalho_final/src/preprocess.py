import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text)
    return text.strip()

def load_samples(file_path="/trabalho_final/data/"):
    with open(file_path, "r", encoding="utf-8") as f:
        return [clean_text(line) for line in f if line.strip()]
