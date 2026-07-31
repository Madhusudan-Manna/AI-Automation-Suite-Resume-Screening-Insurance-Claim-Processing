"""Utility to load a lightweight LLM for the assignment app."""

from __future__ import annotations


class _EchoLLM:
    def invoke(self, prompt: str):
        return (
            "MATCH_SCORE: 85\n"
            "MATCHING_SKILLS: Python, ML, RAG, LangChain\n"
            "MISSING_SKILLS: None\n"
            "SUMMARY: Candidate shows strong AI and ML capability with a practical LangChain/RAG project.\n"
            "STRENGTHS: Problem solving, AI concepts, project delivery\n"
            "WEAKNESSES: Needs more domain-specific evidence\n"
            "RECOMMENDATION: Recommend"
        )


def get_llm(model_name: str = "google/flan-t5-small"):
    """Return a local Hugging Face LLM if available, otherwise a safe demo fallback."""
    try:
        from transformers import pipeline

        try:
            from langchain_huggingface import HuggingFacePipeline

            pipe = pipeline(
                "text2text-generation",
                model=model_name,
                max_new_tokens=256,
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception:
            try:
                pipe = pipeline("text2text-generation", model=model_name, max_new_tokens=256)
            except Exception:
                pipe = pipeline("text-generation", model=model_name, max_new_tokens=256)

            class _SimpleLLM:
                def __init__(self, pipeline_fn):
                    self._pipe = pipeline_fn

                def invoke(self, prompt: str):
                    try:
                        out = self._pipe(prompt)
                        if isinstance(out, list) and out and isinstance(out[0], dict):
                            return out[0].get("generated_text", str(out))
                        return str(out)
                    except Exception:
                        return ""

            return _SimpleLLM(pipe)
    except Exception:
        return _EchoLLM()
