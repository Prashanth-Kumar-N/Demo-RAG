
import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import success, error

def loadAllDocs(files):
    from src.loader import loadDocuments

    response = loadDocuments(files)
    status = success
    total_docs = []

    if response["status"] == success:
        for file in response["docs"]:
            total_docs.extend(response["docs"][file])

        print("Documents loaded successfully, Total pages loaded --->", len(total_docs))
    else:
        status = error
        print("Error loading documents:", response["message"])

    return {"status": status, "message": response["message"], "docs": total_docs}


def splitDocs(docs):
    from src.splitter_table_aware_with_headings import split_documents

    response = split_documents(docs)
    return response


def getEmbeddingModel():
    from src.embeddings import get_embed_model_info
    
    model_info = get_embed_model_info()
    return model_info


def get_vector_index(nodes, model_info):
    from src.vectorstore import check_and_return_index
    index = check_and_return_index(nodes, model_info)
    return index


def ingest(files):
    # load documents
    response = loadAllDocs(files)   

    if response["status"] == success:
        # Split and chunk documents
        split_response = splitDocs(response["docs"])
        
        if split_response["status"] == success:
            print("Documents split successfully, Total Nodes created --->", len(split_response["chunks"]))

            # Check embedding model and dimension
            model_info = getEmbeddingModel()
            print(f"Using embedding model: {model_info[1]} with dimension {model_info[2]}")

            # Create vector index and store
            try:
                 index = get_vector_index(split_response["chunks"], model_info)
                 
                 # Testing if ingested properly by printing out some nodes and their metadata
                 docstore = index.docstore
                 print(f"Total documents in docstore: {len(index.storage_context.docstore.docs)}")
                #  for node_id, node in docstore.docs.items():
                #     print(node.metadata)
                 return {"status": success, "message": "Ingestion completed successfully", "index": index}

            except Exception as e:
                 print("Error with index", e)
                 return {"status": error, "message": f"Error creating/loading index: {str(e)}", "index": None}
        else:
            print("Error splitting documents")
            return {"status": error, "message": "Error splitting documents", "index": None}
    else:
        print("Error loading")
        return {"status": error, "message": "Error loading documents", "index": None}



