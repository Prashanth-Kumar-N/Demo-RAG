from src.retriever import retrieve_only, classic_rag



def generate_response(index, nodes_for_bm25, query: str, top_k: int = 5, retrieval_only: bool = False, files=None, use_llm_fusion: bool = False):
    """
    Retrieves relevant chunks and generates a grounded answer.
    """
    # CLASSIC RAG
    if retrieval_only:
        return retrieve_only(index=index, nodes_for_bm25=nodes_for_bm25, query=query, top_k=top_k, use_llm_fusion=use_llm_fusion)["rows"]
    else:
        return classic_rag(index=index, nodes_for_bm25=nodes_for_bm25, query=query, top_k=top_k, files=files, use_llm_fusion=use_llm_fusion)
    