from gptcache import Cache
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.manager.vector_data.faiss import Faiss
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.processor.pre import get_prompt
from sentence_transformers import SentenceTransformer as ST

# 1) Embedding function
_st = ST("sentence-transformers/all-MiniLM-L6-v2")
def _to_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    return _st.encode(texts, convert_to_numpy=True).tolist()

# 2) Define cache + vector bases
cache_base = CacheBase("sqlite")    # scalar storage
vector_base = VectorBase("faiss", dimension=384, index_file_path="faiss.index", top_k=1)

# 3) Data manager
dm = get_data_manager(
    cache_base=cache_base,
    vector_base=vector_base,
    max_size=2000,
    clean_size=200
)

# 4) Similarity evaluator
sim = SearchDistanceEvaluation(max_distance=1.0, positive=False)

# 5) Init cache
cache = Cache()
cache.init(
    pre_embedding_func=get_prompt,
    embedding_func=_to_embeddings,
    data_manager=dm,
    similarity_evaluation=sim,
)
