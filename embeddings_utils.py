"""Embedding utilities used by the RAG resume screening flow."""

from __future__ import annotations

from typing import List


class TfidfEmbeddings:
    """Offline TF-IDF fallback for environments without Hugging Face model downloads."""

    def __init__(self, max_features: int = 512):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            self.fit([text])
        return self.vectorizer.transform([text]).toarray()[0].tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


def _load_hf_embedder():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        model.embed_query("test")
        return model
    except Exception:
        return None


def get_embedder(prefer_hf: bool = True):
    """Return the best embedder that is available for the current environment."""
    if prefer_hf:
        model = _load_hf_embedder()
        if model is not None:
            print("Using HuggingFace sentence-transformers embeddings.")
            return model
        print("Could not load HuggingFace embeddings. Falling back to TF-IDF.")

    return TfidfEmbeddings()
