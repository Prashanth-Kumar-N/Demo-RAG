
import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import success, error

def loadAllDocs():
    from src.loader import loadDocuments

    files = [
        "./data/3_JLG_Boom_lifts_Catalog.pdf",
    ]

    response = loadDocuments(files)
    return response


def splitDocs(docs):
    from src.splitter import split_documents
    response = split_documents(docs)
    return response

def main():
    # load documents
    response = loadAllDocs()
    if response["status"] == success:
        print(response["docs"]["3_JLG_Boom_lifts_Catalog.pdf"][0])
        split_response = splitDocs(response["docs"]["3_JLG_Boom_lifts_Catalog.pdf"])
        if split_response["status"] == success:
            print(f"Split {len(split_response['chunks'])} chunks")
        else:
            print("Error splitting documents")
    else:
        print("Error loading")



if __name__ == '__main__':
    main()