import langchain
from langchain_community.document_loaders import PyMuPDFLoader
from src.constants import success, error


def loadDocuments(filesList: list[str]) -> dict:
    docsList = {}
    status = success
    message = 'Documents loaded successfully'
    try:
        for file in filesList:
            loader = PyMuPDFLoader(file)
            docs = loader.load()
            file_name = file.split("/")[-1]
            for doc in docs:
                doc.metadata["file_name"] = file_name 
            docsList[file_name] = docs
    except Exception as e:
        print(f"Error loading documents: {e}")
        status = error
        message = str(e)
        docsList = None

    return {"status": status, "message": message, "docs": docsList}
