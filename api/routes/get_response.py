
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse


router = APIRouter()

@router.post("/get_response")
async def get_response(body: dict = Body(...)) -> JSONResponse:
    try: 
        query = body.get("query")
        from src.vectorstore import check_and_return_index
        from src.bm25_store import check_and_return_nodes_for_bm25
        from src.load_models import load_embedding_model
        from src.generator import generate_response
        from src.constants import files
        
        model_info = load_embedding_model()
        index_info = check_and_return_index(model_info=model_info)
        nodes_for_bm25_info = check_and_return_nodes_for_bm25()
        
        if index_info["present"]:
            index = index_info["index"]
            nodes_for_bm25 = nodes_for_bm25_info["nodes"]
            response = generate_response(index=index, nodes_for_bm25=nodes_for_bm25, query=query, top_k=5, retrieval_only=False, files=files, use_llm_fusion=True)
            return JSONResponse(status_code=200, content=response)
        else:
            raise HTTPException(status_code=404, detail="Index not found. Please build the index first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))