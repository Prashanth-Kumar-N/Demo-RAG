
import os

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss


persist_dir = "../faiss_index"
_cached_index = None


def get_cached_index():
    global _cached_index
    return _cached_index


def set_cached_index(index):
    global _cached_index
    _cached_index = index


def load_index_from_storage_if_exists(model_info):
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context=storage_context, embed_model=model_info[0])
    return None


def create_and_store_vectorStore_index(nodes, model_info):
    embed_model, model, dimension = model_info
    if "text-" in model or "multi-qa-MiniLM-L6-cos-v1" in model:
        # OPENAI models use cosine similarity, which is equivalent to inner product on normalized vectors
        # text-embedding-3-small is already normalized, so we can use inner product directly

        # sentence-transformers/multi-qa-MiniLM-L6-cos-v1 is cosine trained but not normalized, so we will use inner product without normalization, 
        # which is not ideal but should still work reasonably well for demonstration purposes
        #✅ Inner Product + cosine-trained embeddings
        # ⇒ behaves as cosine similarit
        faiss_index = faiss.IndexFlatIP(dimension)
    else:
        # For other models, we can use L2 distance, but we need to normalize vectors first
        # This is a simplified approach; in practice, you might want to normalize the vectors before adding them to the index
        # For demonstration, we'll use L2 distance without normalization
        faiss_index = faiss.IndexFlatL2(dimension)

    vector_store = FaissVectorStore(faiss_index = faiss_index)
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        vector_store=vector_store
    )

    # Persist the index to disk
    index.storage_context.persist(persist_dir=persist_dir)
    return index


def build_index(nodes, model_info):
    embed_model, model, dimension  = model_info
    #check if index exists in storage, if yes, load and return, if no, create and store, then return
    
    if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)

    if os.listdir(persist_dir):
        index = load_index_from_storage_if_exists(model_info)
    else:
        index = create_and_store_vectorStore_index(nodes, model_info)

    set_cached_index(index)
    return index


def check_and_return_index(model_info):
    cached_index = get_cached_index()
    if cached_index is not None:
        return {"present": True, "index": cached_index}

    index = load_index_from_storage_if_exists(model_info)
    if index is not None:
        set_cached_index(index)
        return {"present": True, "index": index}

    return {"present": False, "index": None}
     