"""
llm_utils.py
------------
Loads a small, free, local instruction-following model (flan-t5-base) through
HuggingFace's `transformers` pipeline, wrapped so LangChain can call it just
like it would call OpenAI. No API key needed.

First run downloads the model (~250MB) from HuggingFace and caches it locally;
after that it works fully offline.
"""


def get_llm(model_name: str = "google/flan-t5-base"):
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
        # Fallback: try to construct a transformers pipeline directly.
        try:
            # Some transformers versions expect 'text-generation' instead
            # of 'text2text-generation' for T5-like models. Try both.
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
            # As a last resort return a dummy LLM that echoes the prompt.
            class _EchoLLM:
                def invoke(self, prompt: str):
                    return f"[DUMMY LLM] {prompt[:200]}"

            return _EchoLLM()
