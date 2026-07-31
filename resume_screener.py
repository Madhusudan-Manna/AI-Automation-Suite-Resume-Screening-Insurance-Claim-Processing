"""Resume screening using LangChain + RAG and a FAISS vector store."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    PyPDFLoader = None
    FAISS = None
    PromptTemplate = None
    RecursiveCharacterTextSplitter = None

if PromptTemplate is None:
    class _SimplePromptTemplate:
        def __init__(self, template: str):
            self.template = template

        @classmethod
        def from_template(cls, template: str):
            return cls(template)

        def format(self, **kwargs):
            return self.template.format(**kwargs)

    PromptTemplate = _SimplePromptTemplate

if RecursiveCharacterTextSplitter is None:
    class _SimpleTextSplitter:
        def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str):
            chunks = []
            i = 0
            while i < len(text):
                chunks.append(text[i:i + self.chunk_size])
                i += self.chunk_size - self.chunk_overlap
            return chunks

    RecursiveCharacterTextSplitter = _SimpleTextSplitter

if FAISS is None:
    class _SimpleDoc:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class _SimpleRetriever:
        def __init__(self, docs):
            self._docs = docs

        def invoke(self, query: str):
            return self._docs[:4]

    class _SimpleFAISS:
        def __init__(self, docs):
            self._docs = docs

        @classmethod
        def from_texts(cls, texts, embedding, metadatas=None):
            docs = [
                _SimpleDoc(t, m if metadatas is not None else {})
                for t, m in zip(texts, metadatas or [{}] * len(texts))
            ]
            return cls(docs)

        def as_retriever(self, search_kwargs=None):
            return _SimpleRetriever(self._docs)

        def similarity_search(self, query: str, k: int = 4):
            return self._docs[:k]

    FAISS = _SimpleFAISS

from embeddings_utils import get_embedder


EVAL_PROMPT = PromptTemplate.from_template(
    """You are a strict but fair recruiter. Using ONLY the resume excerpts below,
 evaluate the candidate against the Job Description.

Job Description:
{jd}

Relevant resume excerpts:
{context}

Reply in EXACTLY this format (one line per field, no extra text):
MATCH_SCORE: <number 0-100>
MATCHING_SKILLS: <comma separated>
MISSING_SKILLS: <comma separated>
SUMMARY: <1-2 sentence summary of the candidate>
STRENGTHS: <comma separated>
WEAKNESSES: <comma separated>
RECOMMENDATION: <Strongly Recommend | Recommend | Consider | Not Recommended>
"""
)


@dataclass
class ResumeEvaluation:
    resume_name: str
    match_score: int
    matching_skills: List[str]
    missing_skills: List[str]
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str


def load_resume_text(pdf_path: str) -> str:
    """Read a resume PDF into plain text."""
    if PyPDFLoader is not None:
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            return "\n".join(getattr(p, "page_content", "") for p in pages)
        except Exception:
            pass

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(pdf_path)
        texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                texts.append(text)
        return "\n".join(texts)
    except Exception:
        return ""


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def build_vectorstore(resume_texts: dict, embedder=None) -> FAISS:
    """Create a FAISS vector store for resume chunks with metadata for source attribution."""
    embedder = embedder or get_embedder()
    all_chunks, metadatas = [], []

    for name, text in resume_texts.items():
        for chunk in split_text(text):
            all_chunks.append(chunk)
            metadatas.append({"source": name})

    try:
        return FAISS.from_texts(all_chunks, embedding=embedder, metadatas=metadatas)
    except TypeError:
        return FAISS.from_texts(all_chunks, embedder, metadatas=metadatas)


def parse_llm_output(resume_name: str, raw_text: str) -> ResumeEvaluation:
    def grab(field: str, default: str = "") -> str:
        match = re.search(rf"{field}:\s*(.+)", raw_text, re.IGNORECASE)
        return match.group(1).strip() if match else default

    def as_list(field: str) -> List[str]:
        raw = grab(field)
        return [s.strip() for s in raw.split(",") if s.strip()]

    score_raw = grab("MATCH_SCORE", "0")
    digits = re.search(r"\d+", score_raw)
    score = int(digits.group()) if digits else 0

    return ResumeEvaluation(
        resume_name=resume_name,
        match_score=min(max(score, 0), 100),
        matching_skills=as_list("MATCHING_SKILLS"),
        missing_skills=as_list("MISSING_SKILLS"),
        summary=grab("SUMMARY", "No summary produced."),
        strengths=as_list("STRENGTHS"),
        weaknesses=as_list("WEAKNESSES"),
        recommendation=grab("RECOMMENDATION", "Consider"),
    )


def evaluate_resume(resume_name: str, vectorstore: FAISS, jd: str, llm, k: int = 4) -> ResumeEvaluation:
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(jd)
    except Exception:
        docs = vectorstore.similarity_search(jd, k=k)

    if docs and hasattr(docs[0], "metadata"):
        filtered = [d for d in docs if d.metadata.get("source") == resume_name]
        if filtered:
            docs = filtered

    context = "\n---\n".join(getattr(d, "page_content", str(d)) for d in docs) or "(no matching content found)"
    prompt_text = EVAL_PROMPT.format(jd=jd, context=context)
    raw_response = llm.invoke(prompt_text)
    raw_text = raw_response if isinstance(raw_response, str) else getattr(raw_response, "content", str(raw_response))
    return parse_llm_output(resume_name, raw_text)


def compare_resumes(resume_names: List[str], vectorstore: FAISS, jd: str, llm) -> List[ResumeEvaluation]:
    results = [evaluate_resume(name, vectorstore, jd, llm) for name in resume_names]
    return sorted(results, key=lambda r: r.match_score, reverse=True)
