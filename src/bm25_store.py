import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from llama_index.core.schema import TextNode

PROJECT_ROOT = Path(__file__).parent.parent
persist_dir = f"{PROJECT_ROOT}/nodes_for_bm25"
_cached_nodes_for_bm25 = None


def get_cached_nodes_for_bm25():
    """Get the cached BM25 index from memory."""
    global _cached_nodes_for_bm25
    return _cached_nodes_for_bm25


def set_cached_nodes_for_bm25(nodes_for_bm25):
    """Cache nodes for BM25 search in memory."""
    global _cached_nodes_for_bm25
    _cached_nodes_for_bm25 = nodes_for_bm25


def store_nodes_for_bm25(chunks: List[TextNode]) -> None:
    """
    Build a BM25 index from chunks and persist to disk.
    
    Args:
        chunks: List of TextNode objects with text and metadata
    
    Returns: saves nodes to local for BM25 search and local cache
    """
    if not chunks:
        raise ValueError("No chunks provided to build BM25 index")
    
    # Create persist directory if it doesn't exist
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
        
    # Persist BM25 index
    bm25_path = os.path.join(persist_dir, "bm25_nodes.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Nodes for BM25 persisted to {persist_dir}")


def load_nodes_for_bm25_from_storage() -> Dict[str, Any] | None:
    """
    Load BM25 index and chunk metadata from disk.
    
    Returns:
        Nodes -> List[TextNode]
        or None if not found
    """
    bm25_path = os.path.join(persist_dir, "bm25_nodes.pkl")
    
    if not os.path.exists(bm25_path) :
        return None
    
    try:
        # Load Nodes for BM25
        with open(bm25_path, "rb") as f:
            nodes_for_bm25 = pickle.load(f)        
        print(f"Nodes for BM25 loaded from {persist_dir}")        
        return nodes_for_bm25 
    except Exception as e:
        print(f"Error loading BM25 index: {e}")
        return None


def build_bm25_storage(chunks: List[TextNode]) -> Dict[str, Any]:
    nodes_for_bm25 = load_nodes_for_bm25_from_storage()
    if nodes_for_bm25 is not None:
        set_cached_nodes_for_bm25(nodes_for_bm25)
        return {
            "present": True,
            "nodes": nodes_for_bm25
        }
    
    if chunks is not None:
        try:
            store_nodes_for_bm25(chunks)
            set_cached_nodes_for_bm25(chunks)
            return {
                "present": True,
                "nodes": chunks
            }
        except Exception as e:
            print(f"Error creating BM25 index: {e}")
            return {
                "present": False,
                "nodes": None
            }
    else:
        print("No chunks provided to build BM25 index")
        return {
            "present": False,
            "nodes": None
        }


def check_and_return_nodes_for_bm25(chunks: List[TextNode] = None) -> Dict[str, Any]:
    """
    Check if nodes exists in cache or storage.
        
    Args:
        chunks: List of TextNode objects (required if index doesn't exist)
    
    Returns:
        {
            "present": bool,
            "nodes": List[TextNode],
        }
    """
    # Check cache first
    cached_nodes_for_bm25 = get_cached_nodes_for_bm25()
    
    if cached_nodes_for_bm25 is not None:
        print("Using cached BM25 index")
        return {
            "present": True,
            "nodes": cached_nodes_for_bm25
        }
    
    # Check storage
    stored_nodes_for_bm25 = load_nodes_for_bm25_from_storage()
    if stored_nodes_for_bm25 is not None:
        set_cached_nodes_for_bm25(stored_nodes_for_bm25)
        return {
            "present": True,
            "nodes": stored_nodes_for_bm25
        }
    
    # Not found and no chunks to build from
    return {
        "present": False,
        "nodes": None
    }


def bm25_search(query: str, bm25_index: BM25Okapi, chunk_metadata: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Search for top_k most relevant chunks using BM25.
    
    Args:
        query: Query string
        bm25_index: BM25Okapi instance
        chunk_metadata: List of chunk metadata dicts
        top_k: Number of top results to return
    
    Returns:
        List of dicts with keys: chunk_id, text, metadata, score, rank
    """
    if not bm25_index or not chunk_metadata:
        return []
    
    # Tokenize query
    query_tokens = query.lower().split()
    
    # Get BM25 scores
    scores = bm25_index.get_scores(query_tokens)
    
    # Get top_k indices
    top_indices = sorted(
        range(len(scores)), 
        key=lambda i: scores[i], 
        reverse=True
    )[:top_k]
    
    # Build results
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        if idx < len(chunk_metadata):
            result = {
                "rank": rank,
                "score": float(scores[idx]),
                "chunk_id": chunk_metadata[idx]["chunk_id"],
                "text": chunk_metadata[idx]["text"],
                "metadata": chunk_metadata[idx]["metadata"],
                "index": chunk_metadata[idx]["index"]
            }
            results.append(result)
    
    return results
