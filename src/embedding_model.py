from langchain_community.embeddings import OllamaEmbeddings
from typing import List

def get_ollama_embeddings(model_name: str, base_url: str = "http://localhost:11434") -> OllamaEmbeddings:
    print(f"Initializing OllamaEmbeddings with model: '{model_name}' at '{base_url}'")
    try:
        embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
        _ = embeddings.embed_query("test embedding connection")
        print("OllamaEmbeddings initialized and connected successfully.")
        return embeddings
    except Exception as e:
        print(f"Error initializing OllamaEmbeddings: {e}")
        print("Please ensure Ollama is running and the specified embedding model is pulled.")
        raise
