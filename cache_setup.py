# cache_setup.py
import os
import shutil
import re
import unicodedata
from typing import Any, Dict, Tuple

from gptcache import cache as GLOBAL_CACHE
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# ---- Embeddings (Sentence Transformers) ----
# If you don't have sentence-transformers installed, install:
#   pip install sentence-transformers
from sentence_transformers import SentenceTransformer as ST

# Single global model instance (fast + shared)
_ST = ST("sentence-transformers/all-MiniLM-L6-v2")


def to_embeddings(x, **kwargs):
    """
    Convert input (string or list of strings) to normalized 384-d vectors (list or list-of-lists).
    We return .tolist() because GPTCache serializes Python lists cleanly.
    """
    if isinstance(x, str):
        v = _ST.encode([x], convert_to_numpy=True, normalize_embeddings=True)[0]
        return v.tolist()
    if isinstance(x, (list, tuple)):
        v = _ST.encode(list(x), convert_to_numpy=True, normalize_embeddings=True)
        return v.tolist()
    # fallback: stringify
    v = _ST.encode([str(x)], convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.tolist()


# ---------- Canonicalization for exact-duplicate stability ----------

def _canonicalize(s: str) -> str:
    # 1) Unicode normalize (fold accents / compatibility)
    s = unicodedata.normalize("NFKC", s)
    # 2) Standardize quotes/dashes (common dataset noise)
    s = (s.replace("“", '"').replace("”", '"')
           .replace("’", "'").replace("–", "-").replace("—", "-"))
    # 3) Collapse whitespace and strip
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preproc(x: Any, **kwargs) -> str:
    """
    Pre-embedding preprocessor. Accept **kwargs because GPTCache can pass extras.
    If you pass dict/messages elsewhere, you can wrap gptcache.processor.pre.get_prompt here.
    """
    return _canonicalize(str(x))


# ---------- Cache builder ----------

def build_cache(
    run_tag: str,
    similarity_max_distance: float = 0.6,
    dim: int = 384,
    reset_files: bool = True,
) -> Tuple[Any, Dict[str, str]]:
    """
    Prepare GPTCache to use per-policy files under cachedata/<run_tag>/.

    We disable GPTCache's internal cleanup (clean_size=None) and set a huge max_size
    because we do eviction logically in our policy (denylist), which is stable on Windows + FAISS.
    """
    base_dir = os.path.abspath(os.path.join("cachedata", run_tag))
    sqlite_path = os.path.join(base_dir, "sqlite.db")
    faiss_path = os.path.join(base_dir, "faiss.index")

    if reset_files and os.path.isdir(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)

    # Explicit absolute paths remove any ambiguity about default filenames
    cache_base = CacheBase("sqlite", url=f"sqlite:///{sqlite_path}")
    vector_base = VectorBase(
        "faiss",
        dimension=dim,
        index_file_path=faiss_path,
        top_k=1,
    )

    # Important: turn off internal "cleanup" that can trigger FAISS deletes with wrong IDs
    dm = get_data_manager(
        cache_base=cache_base,
        vector_base=vector_base,
        max_size=10**9,     # large cap; we enforce policy externally
        clean_size=None,    # disable internal delete cycles to avoid FAISS ID issues
    )

    sim = SearchDistanceEvaluation(max_distance=similarity_max_distance, positive=False)

    GLOBAL_CACHE.init(
        pre_embedding_func=preproc,
        embedding_func=to_embeddings,
        data_manager=dm,
        similarity_evaluation=sim,
    )

    print(f"[build_cache] run_tag={run_tag}")
    print(f"[build_cache] sqlite = {sqlite_path}")
    print(f"[build_cache] faiss  = {faiss_path}")
    return GLOBAL_CACHE, {"sqlite": sqlite_path, "faiss": faiss_path}
