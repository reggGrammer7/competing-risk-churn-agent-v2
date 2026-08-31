"""
Public retrieval interface -- this is what agent.py imports, and the only
module anything outside backend/rag/ should import from.

Prefers the real vector-DB implementation (FAISS + sentence-transformer
embeddings -- matches on MEANING). Falls back to TF-IDF (matches on shared
vocabulary only) if the embedding model can't be loaded: no internet, a
firewalled sandbox, the very first run before HuggingFace's cache is warm,
or -- the case this env var exists for -- a memory-constrained host where
even ATTEMPTING to import torch could get the whole process OOM-killed
before it finishes starting (this is a real failure mode, not a
hypothetical one: it's what happens on Render's free tier, whose RAM limit
sits well below what torch + sentence-transformers need just to import).
Either way, retrieve(query, corpus, top_k) has the exact same signature and
return shape, so nothing calling it needs to know or care which backend is
actually active -- check BACKEND below if you need to know which one loaded.
"""
import os
import warnings

if os.environ.get("DISABLE_VECTOR_RETRIEVER", "").lower() in ("1", "true", "yes"):
    # Skip even ATTEMPTING the torch/sentence-transformers import -- on a
    # memory-constrained host, the import itself (not a failed download)
    # can trigger an OOM kill, which the try/except below can't catch
    # because the OS kills the whole process, not a catchable Python
    # exception. This check runs BEFORE that import is ever reached.
    from backend.rag.tfidf_retriever import retrieve  # noqa: F401
    BACKEND = "tfidf"
else:
    try:
        from backend.rag.vector_retriever import retrieve  # noqa: F401
        BACKEND = "faiss+embeddings"
    except Exception as e:
        warnings.warn(
            f"Vector retriever unavailable ({e.__class__.__name__}: {e}); falling back to TF-IDF retrieval. "
            "This is expected with no internet access, since the embedding model downloads on first use. "
            "Retrieval still works end-to-end, just by shared vocabulary instead of meaning until the "
            "embedding model can be downloaded.",
            stacklevel=2,
        )
        from backend.rag.tfidf_retriever import retrieve  # noqa: F401
        BACKEND = "tfidf"
