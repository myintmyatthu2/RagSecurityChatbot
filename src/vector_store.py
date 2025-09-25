import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Any

def get_chroma_vector_store(
    persist_directory: str,
    collection_name: str,
    embedding_function: Any
) -> Chroma:
    print(f"Attempting to connect to ChromaDB at: '{persist_directory}' with collection: '{collection_name}'")
    os.makedirs(persist_directory, exist_ok=True)

    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name
        )
        print("ChromaDB vector store initialized successfully.")
        return vector_store
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        raise

def add_documents_to_vector_store(
    vector_store: Chroma,
    documents: List[Document]
):
    if not documents:
        print("No documents to add to the vector store.")
        return

    print(f"Adding {len(documents)} documents to the vector store...")
    try:
        vector_store.add_documents(documents)
        vector_store.persist()
        print(f"Successfully added {len(documents)} documents to vector store.")
    except Exception as e:
        print(f"Error adding documents to vector store: {e}")
        raise

