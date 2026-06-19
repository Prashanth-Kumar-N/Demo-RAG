
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def load_embedding_model():
    from src.embeddings import get_embed_model_info
    return get_embed_model_info()

def load_llm():    
    #from llama_index.llms.huggingface import HuggingFaceLLM

    # llm = HuggingFaceLLM(
    #     model_name="microsoft/phi-2",   # small, works in Codespaces
    #     tokenizer_name="microsoft/phi-2",
    #     max_new_tokens=256,
    #     device_map="auto"
    # )
    # llm = HuggingFaceLLM(
    #     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     max_new_tokens=256,
    #     device_map="auto",
    # )

    llama_3_3_70B = "llama-3.3-70b-versatile"
    Llama_4_Scout_17B = "meta-llama/llama-4-scout-17b-16e-instruct"
    from llama_index.llms.groq import Groq
    llm = Groq(model=llama_3_3_70B, api_key=os.getenv("GROQ_API_KEY"))

    return llm
