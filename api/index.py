import os
import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from logging import basicConfig, getLogger, INFO
from routes.create_index import router as create_index_router
from routes.get_response import router as get_response_router

basicConfig(level=INFO)
logger = getLogger(__name__)

app = FastAPI(debug=True, title="Oshkosh RAG", summary="RAG application for oshkosh publicly available data")


#------------------------------------------------
# add logging middleware and CORS middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()

    logger.info(
        json.dumps({
            "event": "request",
            "method": request.method,
            "url": str(request.url)
        })
    )

    response = await call_next(request)

    process_time = round((time.time() - start_time) * 1000, 2)

    logger.info(
        json.dumps({
            "event": "response",
            "status_code": response.status_code,
            "duration_ms": process_time
        })
    )

    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#-------------------------------------------------
# add routers for the endpoints
@app.get("/")
def root() -> dict:
    return {"message": "Welcome to the Oshkosh RAG API"}

app.include_router(create_index_router, tags=["Create Index"])
app.include_router(get_response_router, tags=["Get Response"])


#-------------------------------------------------
# create server and port 
def start_server():
    import uvicorn
    
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

#-------------------------------------------------
# Exception handlers

#HTTP exception handler, to catch all HTTP exceptions and return a JSON response with the error message
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail
        }
    )

# add global exception handler here, to catch all unhandled exceptions and return a JSON response with the error message, print to console as well
@app.exception_handler(Exception)
async def global_exception_handler(req: Request, exc: Exception):
    print(f"Error: {exc}")
    return JSONResponse(status_code=500, content={"message": str(exc)})




if __name__ == "__main__":
    start_server()