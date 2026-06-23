
# llamaIndex supports all these parsing techniques
#  https://www.lancedb.com/blog/chunking-techniques-with-langchain-and-llamaindex

import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import success, error
from src.vectorstore import check_and_return_index, build_index
from src.bm25_store import check_and_return_nodes_for_bm25, build_bm25_storage

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
    index = build_index(nodes, model_info)
    return index

def store_nodes_to_storage_for_bm25(nodes):
    nodes =  build_bm25_storage(nodes)["nodes"]


def ingest(files):

    model_info = getEmbeddingModel()
    index_info = check_and_return_index(model_info)
    nodes_for_bm25_info = check_and_return_nodes_for_bm25()
    if index_info["present"] and nodes_for_bm25_info["present"]:
        print("Index already exists in storage, loading index...")
        return {"status": success, "message": "Index loaded from storage", "index": index_info["index"], "nodes_for_bm25": nodes_for_bm25_info["nodes"]}
    else:
        print("No existing index found in storage, creating new index...")
        # load documents
        response = loadAllDocs(files)   

        if response["status"] == success:
            # Split and chunk documents
            split_response = splitDocs(response["docs"])
            
            if split_response["status"] == success:
                print("Documents split successfully, Total Nodes created --->", len(split_response["chunks"]))

                # Check embedding model and dimension
                print(f"Using embedding model: {model_info[1]} with dimension {model_info[2]}")

                # Create vector index and store
                try:
                    index = get_vector_index(split_response["chunks"], model_info)
                    nodes = store_nodes_to_storage_for_bm25(split_response["chunks"])
                    # Testing if ingested properly by printing out some nodes and their metadata
                    docstore = index.docstore
                    print(f"Total documents in docstore: {len(index.storage_context.docstore.docs)}")
                    #  for node_id, node in docstore.docs.items():
                    #     print(node.metadata)
                    return {"status": success, "message": "Ingestion completed successfully, index created", "index": index, "nodes_for_bm25": nodes}

                except Exception as e:
                    print("Error with index", e)
                    return {"status": error, "message": f"Error creating/loading index: {str(e)}", "index": None, "nodes_for_bm25": None}
            else:
                print("Error splitting documents")
                return {"status": error, "message": "Error splitting documents", "index": None, "nodes_for_bm25": None}
        else:
            print("Error loading")
            return {"status": error, "message": "Error loading documents", "index": None, "nodes_for_bm25": None}



