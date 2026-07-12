
import os
import sys
import logging
from pathlib import Path

# Add workspace root to Python path before importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import success, error, files
from src.ingestion import ingest
from src.generator import generate_response

logger = logging.getLogger(__name__)

def main():
    print("Starting ingestion process...")
    # load documents
    response = ingest(files)
    if response["status"] == success:
        logger.info(response["message"])
        index = response["index"]
        nodes_for_bm25 = response["nodes_for_bm25"]
        response = generate_response(index=index, nodes_for_bm25=nodes_for_bm25, query="Explain the sequence of machine setup and operation of a JLG Lift", top_k=5, retrieval_only=False, files=files, use_llm_fusion=False)
        logger.info(f"Response: {response.response_text}")
    else:
        logger.error("Ingestion failed: %s", response["message"])



if __name__ == '__main__':
    main()