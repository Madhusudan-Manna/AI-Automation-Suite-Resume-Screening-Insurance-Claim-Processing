"""
resume_screener.py
-------------------
Q1: AI Resume Screening Assistant (LangChain + RAG)

Plain-English flow:
1. Read the resume PDF(s) into text.
2. Chop the text into small chunks (splitting).
3. Turn each chunk into a vector (embedding) and store it (FAISS = our vector database).
4. When we want to evaluate a resume against a Job Description, we "retrieve"
   the most relevant chunks and hand them to the LLM along with the JD.
5. The LLM writes back a structured evaluation, which we parse into a
   plain Python dictionary so the app can display it nicely.
"""

from dataclasses import dataclass
from typing import List
import re

# LangChain imports can be heavy and sometimes fail in trimmed environments
_langchain_import_error = None
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
except Exception as e:  # pragma: no cover - graceful fallback for environments without these packages
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None
    FAISS = None
    PromptTemplate = None
    _langchain_import_error = e

# Provide small, safe fallbacks so the notebook can be executed in minimal
# environments without crashing at import-time. These fallbacks are
# intentionally simple and meant for demonstration / testing only.
if PromptTemplate is None:
    class _SimplePromptTemplate:
        def __init__(self, template: str):
            self.template = template

        @classmethod
        def from_template(cls, template: str):
            return cls(template)

        def format(self, **kwargs) -> str:
            return self.template.format(**kwargs)

    PromptTemplate = _SimplePromptTemplate

if RecursiveCharacterTextSplitter is None:
    class _SimpleTextSplitter:
        def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str):
            # naive fixed-size split as a fallback
            chunks = []
            i = 0
            n = len(text)
            while i < n:
                chunks.append(text[i : i + self.chunk_size])
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
            # return up to k docs (no semantic search) for demonstrations
            return self._docs[:4]

    class _SimpleFAISS:
        def __init__(self, docs):
            self._docs = docs

        @classmethod
        def from_texts(cls, texts, embedder, metadatas=None):
            docs = [
                _SimpleDoc(t, m if metadatas is not None else {})
                for t, m in zip(texts, metadatas or [{}] * len(texts))
            ]
            return cls(docs)

        def as_retriever(self, search_kwargs=None):
            return _SimpleRetriever(self._docs)

    FAISS = _SimpleFAISS

from embeddings_utils import get_embedder


# ---------------------------------------------------------------------------
# 1. Loading + splitting
# ---------------------------------------------------------------------------

def load_resume_text(pdf_path: str) -> str:
    """Read a resume PDF and return it as one big string."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# 2. Vector store (retriever)
# ---------------------------------------------------------------------------

def build_vectorstore(resume_texts: dict, embedder=None) -> FAISS:
    """resume_texts: {resume_name: full_text}. Returns a FAISS vector store
    where every chunk is tagged with which resume it came from, so we can
    later filter / attribute answers back to the right candidate."""
    embedder = embedder or get_embedder()

    all_chunks, metadatas = [], []
    for name, text in resume_texts.items():
        for chunk in split_text(text):
            all_chunks.append(chunk)
            metadatas.append({"source": name})

    return FAISS.from_texts(all_chunks, embedder, metadatas=metadatas)


# ---------------------------------------------------------------------------
# 3. Prompt + structured "output parser"
# ---------------------------------------------------------------------------

EVAL_PROMPT = PromptTemplate.from_template(
    """You are a strict but fair recruiter. Using ONLY the resume excerpts
below, evaluate this candidate against the Job Description.

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


def parse_llm_output(resume_name: str, raw_text: str) -> ResumeEvaluation:
    """Turn the LLM's fixed-format reply into a proper Python object.
    This is our 'output parser' - simple and readable instead of relying
    on the LLM to produce perfect JSON (small free models often don't)."""

    def grab(field: str, default: str = "") -> str:
        m = re.search(rf"{field}:\s*(.+)", raw_text, re.IGNORECASE)
        return m.group(1).strip() if m else default

    def as_list(field: str) -> List[str]:
        raw = grab(field)
        return [s.strip() for s in raw.split(",") if s.strip()]

    score_raw = grab("MATCH_SCORE", "0")
    score_digits = re.search(r"\d+", score_raw)
    score = int(score_digits.group()) if score_digits else 0

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


# ---------------------------------------------------------------------------
# 4. Putting it together: evaluate one resume against a JD
# ---------------------------------------------------------------------------

def evaluate_resume(resume_name: str, vectorstore: FAISS, jd: str, llm, k: int = 4) -> ResumeEvaluation:
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, "filter": {"source": resume_name}}
    )
    docs = retriever.invoke(jd)
    context = "\n---\n".join(d.page_content for d in docs) or "(no matching content found)"

    prompt_text = EVAL_PROMPT.format(jd=jd, context=context)
    raw_response = llm.invoke(prompt_text)
    raw_text = raw_response if isinstance(raw_response, str) else getattr(raw_response, "content", str(raw_response))

    return parse_llm_output(resume_name, raw_text)


def compare_resumes(resume_names: List[str], vectorstore: FAISS, jd: str, llm) -> List[ResumeEvaluation]:
    """Evaluate several resumes for the same JD and rank them by match score."""
    results = [evaluate_resume(name, vectorstore, jd, llm) for name in resume_names]
    return sorted(results, key=lambda r: r.match_score, reverse=True)
