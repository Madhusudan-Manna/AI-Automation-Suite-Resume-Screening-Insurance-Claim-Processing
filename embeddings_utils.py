"""Embedding utilities used by the RAG resume screening flow."""

from __future__ import annotations

import re
from collections import Counter
from typing import List


class SimpleKeywordEmbeddings:
    """Pure-Python fallback when scikit-learn is unavailable in cloud environments."""

    def __init__(self, max_features: int = 512):
        self.max_features = max_features
        self.vocabulary_ = {}
        self._fitted = False

    def fit(self, texts: List[str]):
        counts = Counter()
        for text in texts:
            tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
            counts.update(token for token in tokens if len(token) > 1)

        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: self.max_features]
        self.vocabulary_ = {token: idx for idx, (token, _) in enumerate(ranked)}
        self._fitted = True

    def _vectorize(self, text: str) -> List[float]:
        if not self._fitted:
            self.fit([text])

        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        token_counts = Counter(token for token in tokens if len(token) > 1)
        vector = [0.0] * len(self.vocabulary_)

        for token, count in token_counts.items():
            idx = self.vocabulary_.get(token)
            if idx is not None:
                vector[idx] = float(count)

        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vectorize(text)

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


class TfidfEmbeddings:
    """Offline TF-IDF fallback for environments without Hugging Face model downloads."""

    def __init__(self, max_features: int = 512):
        self._using_sklearn = False
        self.vectorizer = SimpleKeywordEmbeddings(max_features=max_features)

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
            self._using_sklearn = True
        except Exception:
            self.vectorizer = SimpleKeywordEmbeddings(max_features=max_features)
            self._using_sklearn = False

        self._fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)
        if self._using_sklearn:
            return self.vectorizer.transform(texts).toarray().tolist()
        return self.vectorizer.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted:
            self.fit([text])
        if self._using_sklearn:
            return self.vectorizer.transform([text]).toarray()[0].tolist()
        return self.vectorizer.embed_query(text)

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
    """Return the best embedder available for the current environment."""
    if prefer_hf:
        model = _load_hf_embedder()
        if model is not None:
            print("Using HuggingFace sentence-transformers embeddings.")
            return model
        print("Could not load HuggingFace embeddings. Falling back to local embedding logic.")

    return TfidfEmbeddings()
