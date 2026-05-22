from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from sentence_transformers import SentenceTransformer


# for OpenAI text-embedding-3-small, dimension is 1536
# if we embed using openai, we will need api-key for both creating and querying, 
# they are normalizedvectors, so cosine similarity can be used directly with inner product in Faiss
# but since we dont have the kyey for now, we will use the sentence-transformers/multi-qa-MiniLM-L6-cos-v1 model, which is free to use locally
# dimension = 1536
# model = "text-embedding-3-small"


# embed_model = OpenAIEmbedding(
#     model="text-embedding-3-small",
#     api_key=""    
# )

# for local, free huggingface models like sentence-transformers/multi-qa-MiniLM-L6-cos-v1,
# has dimension of 384, but the vectors are not normalized, so we will normalize vectors before adding to Faiss, and use cosine similarity (inner product) for retrieval
# normalizing using sentence-transformers, which is the same as the model we are using for embedding, so it should be consistent
# multi-qa-MiniLM-L6-cos-v1 (already normalized for cosine)
# faiss.IndexFlatIP(384) (cosine similarity)
# No manual normalization required in LlamaIndex.


dimension = 384
model = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

embed_model = HuggingFaceEmbedding(model_name=model)


get_embed_model_info = lambda: (embed_model, model, dimension)