from pathlib import Path
import fitz
from src.constants import success, error

PROJECT_ROOT = Path(__file__).parent.parent

def loadDocuments(filesList: list[str]) -> dict:
    docsList = {}
    status = success
    message = 'Documents loaded successfully'
    try:
        for file in filesList:
            file_path = PROJECT_ROOT / file
            doc = fitz.open(str(file_path))
            file_name = file.split("/")[-1]
            docs = []
            for i in range(doc.page_count):
                page = doc.load_page(i)
                text = page.get_text()
                if type(text) == str and text.strip():
                    docs.append({"page_content": text, "metadata": {"file_name": file_name, "page_number": i + 1}})
            docsList[file_name] = docs
    except Exception as e:
        print(f"Error loading documents: {e}")
        status = error
        message = str(e)
        docsList = None

    return {"status": status, "message": message, "docs": docsList}
