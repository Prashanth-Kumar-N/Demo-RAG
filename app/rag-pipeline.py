
import sys
from pathlib import Path

# Add workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import success, error
from src.ingestion import ingest

def get_query_embeddings():
    print("here")

def main():
    # load documents
    files = [
        "./data/2_Striker_4x4_Oshkosh_Airport_Products.pdf",
        "./data/2_Striker_6x6_Oshkosh_Airport_Products.pdf",
        "./data/2_Striker_8x8_Oshkosh_Airport_Products.pdf",
        "./data/3_JLG_Boom_lifts_Catalog.pdf",
        "./data/4_Hydraulic_components_Catalog.pdf",
        "./data/5_Racks_wareshouse_catalog.pdf",
        "./data/6_JLG_12SP_Operational_Safety_manual.pdf",
        "./data/7_jaco-quality-system-manual.pdf",
    ]
    response = ingest(files)
    if response["status"] == success:
        print(response["message"])
    else:
        print("Ingestion failed: ", response["message"])



if __name__ == '__main__':
    main()