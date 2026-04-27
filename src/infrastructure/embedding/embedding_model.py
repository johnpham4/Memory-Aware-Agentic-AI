from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings

@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )