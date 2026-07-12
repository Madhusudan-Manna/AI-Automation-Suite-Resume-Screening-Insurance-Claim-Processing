"""
embeddings_utils.py
--------------------
One job: give us an "embed text into numbers" tool that LangChain can use.

We first try the free HuggingFace sentence-transformer model (best quality,
needs internet the first time to download it). If that isn't available
(no internet, model not cached, etc.) we quietly fall back to a simple
TF-IDF embedder built with scikit-learn, which needs no internet at all.

Either way, the rest of the app doesn't need to know which one is active -
both follow LangChain's Embeddings interface: embed_documents() and embed_query().
"""

from typing import List


class TfidfEmbeddings:
    """A tiny, fully-offline stand-in for a real embedding model.

    It learns a vocabulary from whatever text we give it and turns each
    piece of text into a vector of word-importance scores. It's not as
    smart as a neural embedding model, but it needs zero downloads and
    zero API keys, which makes it a good fallback for a student project.
    """

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


def get_embedder(prefer_hf: bool = True):
    """Return the best embedder we can actually use right now.

    prefer_hf=True -> try the free HuggingFace model first, fall back to TF-IDF.
    """
    if prefer_hf:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # quick smoke test - will throw if the model can't actually be loaded
            model.embed_query("test")
            print("Using HuggingFace sentence-transformers embeddings (all-MiniLM-L6-v2).")
            return model
        except Exception as e:
            print(f"Could not load HuggingFace embeddings ({e}). Falling back to TF-IDF.")

    return TfidfEmbeddings()
