import langchain
from langchain_community.document_loaders import PyMuPDFLoader

print("langchain", langchain.__version__)

files = [
    "./data/3_JLG_Boom_lifts_Catalog.pdf",
]


docs = []
for file in files:
    loader = PyMuPDFLoader(file)
    pdf = loader.load()
    docs = pdf
    for doc in docs:
        doc.metadata["name"] = file.split("/")[-1] 

print(len(docs))
print(docs[1].page_content);