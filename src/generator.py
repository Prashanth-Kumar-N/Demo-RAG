from src.retriever import retrieve_only, classic_rag



def generate_response(index, query: str, top_k: int = 5, retrieval_only: bool = False, files=None):
    """
    Retrieves relevant chunks and generates a grounded answer.
    """
    # CLASSIC RAG
    if retrieval_only:
        return retrieve_only(index=index, query=query, top_k=top_k)
    else:
        return classic_rag(index=index, query=query, top_k=top_k, files=files)
    