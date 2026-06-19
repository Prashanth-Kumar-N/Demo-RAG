
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import pandas as pd

from src.load_models import load_llm


def retrieve_only(index, query: str, top_k: int = 5):
    """
    Returns retrieved chunks only, without synthesized final answer.
    """
    rows = []

    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    # print(f"\nQUERY: {query}")
    # print(f"Retrieved {len(nodes)} chunks\n")

    # Node: <class 'llama_index.core.schema.NodeWithScore'>
    # A NodeWithScore is simply a retrieved chunk of text plus the relevance score assigned to it.

    for i, node in enumerate(nodes, start=1):
        rows.append({
            "rank": i,
            "score": node.score,
            "metadata": node.metadata,
            "response": node.text
        })

    return {"nodes": nodes, "table": pd.DataFrame(rows) }
 

def classic_rag(index, query: str, top_k: int = 5, files=None):
    """
    Retrieves relevant chunks and generates a grounded answer.
    """
    # Retrieval
    matching_files = get_matching_files(query, files)
    llm = load_llm()
    nodes = retrieve_only(index=index, query=query, top_k=top_k * 2)["nodes"]


    # Filter
    if matching_files and len(matching_files) > 0:
        filtered_nodes = [node for node in nodes if node.metadata.get("file_name").lower() in matching_files]
    else:
        filtered_nodes = nodes
    
    # Synthesize response

    # Parameters for response_mode:
    #   * compact:          Produces a concise answer by merging all chunks into a single prompt before synthesis.
    #   * tree_summarize:   Hierarchical summarization (for large documents)
    #   * refine:           Iteratively improves answer, chunk by chunk
    #   * simple_summarize: Basic summarization by concatenation of all chunks
    #   * accumulate:       Returns raw or lightly processed chunk outputs without deep synthesis
    #   * structure_refine: Similar to refine but enforces structured output (JSON-like) while iteratively
    #                          improving the answer

    
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT, llm=llm)
    response = response_synthesizer.synthesize(nodes=filtered_nodes, query= query)

    # Synthesized Response
    print(f"\nQUESTION: {query}")
    print("\nANSWER:")
    print(response)

    # Retrieved Chunks Data
    print("\nSOURCES:")
    rows = []
    for i, src in enumerate(response.source_nodes, start=1):
        # print("=" * 90)
        rows.append({
            "rank": i,
            "score": src.score,
            "metadata": src.metadata,
            "response": src.text[:700]
        })
    
    print(pd.DataFrame(rows))
 

def get_matching_files(query, files):

    matchingFiles = []

    if any(key.lower() in query.lower() for key in ['4x4', '4 x 4', "Striker",'Striker 4']) :
        matchingFiles.extend([file for file in files if "striker 4x4" in file.lower()])

    if any(key.lower() in query.lower() for key in ['8x8',  "8 x 8", "Striker",'Striker 8']) :
        matchingFiles.extend([file for file in files if "striker 8x8" in file.lower()])
    
    if any(key.lower() in query.lower() for key in ['6x6',  "6 x 6", "Striker",'Striker 6']) :
        matchingFiles.extend([file for file in files if "striker 6x6" in file.lower()])

    if any(key in query.lower() for key in ["boom lifts", "boom", "lifts"]):
        matchingFiles.extend([file for file in files if "boom_lifts" in file.lower()]) 
    
    if any(key in query.lower() for key in ["hydraulic", "hydraulic components"]):
        matchingFiles.extend([file for file in files if "hydraulic" in file.lower()])

    if any(key in query.lower() for key in ["racks", "warehouse"]):
        matchingFiles.extend([file for file in files if "racks" in file.lower()])

    if any(key in query.lower() for key in ["safety manual", "operational safety", "safety", "manual"]):
        matchingFiles.extend([file for file in files if "safety" in file.lower()])

    if any(key in query.lower() for key in ["quality system manual", "quality manual", "quality system", "jaco"]):
        matchingFiles.extend([file for file in files if "quality" in file.lower()])
    
    return matchingFiles
    

"""
def create_metadata_filters(query, files):
    
    Example function to create metadata filters based on query and file information.
    This is a placeholder and should be customized based on actual metadata structure and filtering needs.
    
    matchingFiles = {}
    
    # Example: If query contains specific keywords, set filters accordingly
    if any(key in query.lower()  for key in ["Striker 4x4", "4x4", "4 x 4"]):
        matchingFiles = [file for file in files if "striker_4x4" in file.lower()]

    elif any(key in query.lower()  for key in ["Striker 8x8", "8x8", "8 x 8"]):
        matchingFiles = [file for file in files if "striker_8x8" in file.lower()]

    elif any(key in query.lower()  for key in ["Striker 6x6", "6x6", "6 x 6"]):
        matchingFiles = [file for file in files if "striker_6x6" in file.lower()]

    elif any(key in query.lower() for key in ["boom lifts", "boom", "lifts"]):
        matchingFiles = [file for file in files if "boom_lifts" in file.lower()]
    
    elif any(key in query.lower() for key in ["hydraulic", "hydraulic components"]):
        matchingFiles = [file for file in files if "hydraulic" in file.lower()]

    elif any(key in query.lower() for key in ["racks", "warehouse"]):
        matchingFiles = [file for file in files if "racks" in file.lower()]

    elif any(key in query.lower() for key in ["safety manual", "operational safety", "safety", "manual"]):
        matchingFiles = [file for file in files if "safety" in file.lower()]

    elif any(key in query.lower() for key in ["quality system manual", "quality manual", "quality system", "jaco"]):
        matchingFiles = [file for file in files if "quality" in file.lower()]
    
    # Example: Add file-based filters if needed
    # for file in files:
    #     if "Striker_4x4" in file:
    #         filters["source_file"] = file
    
    ###### DOESN"T WORK BECAUSE WHEN FILTERS IS AN ARRAY, IT IS A AND CONDITION BETWEEN THEM AND ONE CHUNK CANNOT SATISFY MULTIPLE FILES. NEED TO FIGURE OUT HOW TO DO OR CONDITION BETWEEN FILES. ######
    if matchingFiles and len(matchingFiles) > 0:
        return MetadataFilters(filters=[ExactMatchFilter(key="file_name", value= file) for file in matchingFiles]) 
    else:
        return None
"""

""" def build_query_engine_md(reference=None, top_k: int = 2) -> RetrieverQueryEngine:
    index = build_or_load_index()

    filters = None
    if reference is not None:
        filters = MetadataFilters( filters=[
                    MetadataFilter(key="reference",value=reference,operator=FilterOperator.EQ) ])

    # retriever = VectorIndexRetriever( index=index, similarity_top_k=top_k, filters=filters )
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    response_synthesizer = get_response_synthesizer(response_mode="compact")
    # return RetrieverQueryEngine( retriever=retriever, response_synthesizer=response_synthesizer)
    return RetrieverQueryEngine.from_args( retriever=retriever, response_synthesizer=response_synthesizer)


def classic_rag(query: str, reference=None, top_k: int = 2):
    # CLASSIC RAG
    query_engine = build_query_engine_md(reference=reference,top_k=top_k)
    response = query_engine.query(query)

    # Synthesized Response
    print(f"\nQUESTION: {query}")
    print("\nANSWER:")
    print(response)

    print("\nSOURCES:")
    for i, src in enumerate(response.source_nodes, start=1):
        print(f"SOURCE {i} | score={src.score}")
        print(src.metadata)
        print() 
        


def build_query_engine(index, top_k: int = 5, llm=None) -> RetrieverQueryEngine:
    # This creates a component responsible for generating the final answer from retrieved chunks.
    # retrieved chunks will be like CHUNK 1, CHUNK 2, CHUNK 3
    # Synthesizer controls the final display
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    # Parameters for response_mode:
    #   * compact:          Produces a concise answer by merging all chunks into a single prompt before synthesis.
    #   * tree_summarize:   Hierarchical summarization (for large documents)
    #   * refine:           Iteratively improves answer, chunk by chunk
    #   * simple_summarize: Basic summarization by concatenation of all chunks
    #   * accumulate:       Returns raw or lightly processed chunk outputs without deep synthesis
    #   * structure_refine: Similar to refine but enforces structured output (JSON-like) while iteratively
    #                          improving the answer

    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT, llm=llm)

    return RetrieverQueryEngine( retriever=retriever, response_synthesizer=response_synthesizer,)
 """