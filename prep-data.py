import yaml
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from text_splitter import run_split_pipeline

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ollama_cfg = config["ollama"]
data_cfg = config["data_ingestion"]

embeddings = OllamaEmbeddings(
    model=ollama_cfg["embedding_model"],
    base_url=ollama_cfg["host"]
)

def build_vector_store(chunks):
    vs_cfg = data_cfg["vector_store"]
    persist_dir = vs_cfg["persist_directory"]

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=vs_cfg["collection_name"],
        persist_directory=persist_dir
    )
    db.persist()
    print(f"[INFO] ✅ ChromaDB built at {persist_dir}")

if __name__ == "__main__":
    docs, chunks = run_split_pipeline()
    build_vector_store(chunks)
