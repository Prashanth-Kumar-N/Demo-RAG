
import os
import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import success, error, files
from src.ingestion import ingest
from src.generator import generate_response

def main():
    # load documents
    response = ingest(files)
    if response["status"] == success:
        print(response["message"])
        index = response["index"]
        nodes_for_bm25 = response["nodes_for_bm25"]
        response = generate_response(index=index, nodes_for_bm25=nodes_for_bm25, query="What is the CAB VISIBILITY of Striker 4x4?", top_k=5, retrieval_only=True, files=files, use_llm_fusion=False)
        print("Response:")
        print(response)
    else:
        print("Ingestion failed: ", response["message"])



if __name__ == '__main__':
    main()