from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/build_index")
async def build_index():
    from src.ingestion import ingest
    from src.constants import files
    try:
        logger.info("Checking index...")
        
        response = ingest(files)
        if response["status"] == "success":
            return JSONResponse(status_code=200, content={"message": response["message"], "status": "success"})
        else:
            raise Exception(response["message"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))