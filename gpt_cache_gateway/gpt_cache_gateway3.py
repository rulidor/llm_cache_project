# --- GPTCache setup (your fork's API) ---
from gptcache import Cache
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.processor.pre import get_prompt

# embeddings: use sentence-transformers directly to avoid onnx/numpy hassles
from sentence_transformers import SentenceTransformer as ST

# LangChain cache bridge + Ollama LLM
from langchain_community.cache import GPTCache as LC_GPTCache
from langchain.globals import set_llm_cache
from langchain_community.llms import Ollama


def build_cache():
    # 1) embedding function
    st = ST("sentence-transformers/all-MiniLM-L6-v2")   # 384-dim
    def to_embeddings(texts):
        if isinstance(texts, str):
            texts = [texts]
        return st.encode(texts, convert_to_numpy=True).tolist()

    # 2) data manager: SQLite (scalar) + FAISS (vector)
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=384, index_file_path="faiss.index", top_k=1)
    dm = get_data_manager(cache_base=cache_base, vector_base=vector_base,
                          max_size=2000, clean_size=200)

    # 3) similarity evaluator (older API uses max_distance/positive)
    sim = SearchDistanceEvaluation(max_distance=1.0, positive=False)

    # 4) init GPTCache
    gcache = Cache()
    gcache.init(
        pre_embedding_func=get_prompt,
        embedding_func=to_embeddings,
        data_manager=dm,
        similarity_evaluation=sim,
    )
    return gcache


def build_llm_with_cache(model_name: str = "phi3:mini"):
    # Create your cache
    gcache = build_cache()

    # Tell LangChain to use it
    set_llm_cache(LC_GPTCache(init_func=lambda: gcache))

    # Create the Ollama LLM handle (calls http://localhost:11434)
    llm = Ollama(model=model_name)
    return llm


if __name__ == "__main__":
    # Build LLM (loads/starts phi3:mini automatically if needed)
    llm = build_llm_with_cache("phi3:mini")

    # Send prompts
    prompt1 = "Explain BLE advertising briefly."
    prompt2 = "What is BLE advertising?"

    # First call -> cache miss (will query Ollama), then store
    print("1st call:", llm.invoke(prompt1))

    # Second call (same prompt) -> cache hit (fast)
    print("2nd call (same):", llm.invoke(prompt1))

    # Semantically similar prompt may hit depending on threshold
    print("Similar prompt:", llm.invoke(prompt2))
