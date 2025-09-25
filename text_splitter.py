import os
import re
import yaml
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_cfg = config["data_ingestion"]

def clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.extract()
    return soup.get_text(separator=" ", strip=True)

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return text.strip()

def deduplicate_chunks(chunks):
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c.page_content not in seen:
            seen.add(c.page_content)
            unique_chunks.append(c)
    return unique_chunks

def load_documents():
    docs = []
    for source in data_cfg["document_sources"]:
        if source["type"] == "pdf":
            folder = source["path"]
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.endswith(".pdf"):
                        loader = PyPDFLoader(os.path.join(folder, file))
                        docs.extend(loader.load())

        elif source["type"] == "csv":
            if os.path.exists(source["path"]):
                loader = CSVLoader(file_path=source["path"])
                docs.extend(loader.load())

        elif source["type"] == "web":
            for url in source["urls"]:
                loader = UnstructuredURLLoader(urls=[url])
                docs_raw = loader.load()
                # Clean HTML content
                for d in docs_raw:
                    d.page_content = clean_html(d.page_content)
                docs.extend(docs_raw)

    print(f"[INFO] Total documents loaded: {len(docs)}")
    return docs

def preprocess_documents(docs):
    """Normalize and clean all documents."""
    for d in docs:
        d.page_content = normalize_text(d.page_content)
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=data_cfg["chunking"]["chunk_size"],
        chunk_overlap=data_cfg["chunking"]["chunk_overlap"]
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Total chunks created: {len(chunks)}")
    return chunks

def run_split_pipeline():
    docs = load_documents()
    docs = preprocess_documents(docs)
    chunks = chunk_documents(docs)
    chunks = deduplicate_chunks(chunks)
    print(f"[INFO] Final unique chunks: {len(chunks)}")
    return docs, chunks

if __name__ == "__main__":
    run_split_pipeline()
