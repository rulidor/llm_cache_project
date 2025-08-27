from gptcache import cache as GLOBAL_CACHE
from cache_setup import build_cache
_active = {"run_tag": None, "sim": None}

def switch_cache(run_tag: str, similarity_max_distance: float = 0.6):
    if _active["run_tag"] == run_tag and _active["sim"] == similarity_max_distance:
        return GLOBAL_CACHE
    cache, _ = build_cache(run_tag, similarity_max_distance, reset_files=False)
    _active["run_tag"] = run_tag
    _active["sim"] = similarity_max_distance
    return cache
