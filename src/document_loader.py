import os
from typing import List, Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_core.documents import Document

def load_documents_from_sources(sources_config: List[Dict[str, Any]]) -> List[Document]:
    all_documents = []
    print("Loading documents from configured sources...")

    for source in sources_config:
        source_type = source.get('type')
        source_path = source.get('path')
        source_urls = source.get('urls')

        try:
            if source_type == "pdf" and source_path:
                if os.path.isdir(source_path):
                    print(f"  Loading PDFs from directory: {source_path}")
                    loader = DirectoryLoader(source_path, glob="*.pdf", loader_cls=PyPDFLoader)
                    all_documents.extend(loader.load())
                elif os.path.isfile(source_path):
                    print(f"  Loading PDF file: {source_path}")
                    loader = PyPDFLoader(source_path)
                    all_documents.extend(loader.load())
                else:
                    print(f"  Warning: PDF path '{source_path}' is not a valid file or directory. Skipping.")

            elif source_type == "csv" and source_path:
                if os.path.isfile(source_path):
                    print(f"  Loading CSV file: {source_path}")
                    loader = CSVLoader(file_path=source_path, encoding="utf-8")
                    all_documents.extend(loader.load())
                else:
                    print(f"  Warning: CSV file '{source_path}' not found. Skipping.")

            elif source_type == "website" and source_urls:
                print(f"  Loading documents from websites: {source_urls}")
                loader = WebBaseLoader(web_paths=source_urls)
                all_documents.extend(loader.load())

            elif source_type == "text" and source_path:
                if os.path.isdir(source_path):
                    print(f"  Loading TXT files from directory: {source_path}")
                    loader = DirectoryLoader(source_path, glob="*.txt", loader_cls=TextLoader)
                    all_documents.extend(loader.load())
                elif os.path.isfile(source_path):
                    print(f"  Loading TXT file: {source_path}")
                    loader = TextLoader(source_path)
                    all_documents.extend(loader.load())
                else:
                    print(f"  Warning: Text path '{source_path}' is not a valid file or directory. Skipping.")

            else:
                print(f"  Warning: Unknown or incomplete document source configuration: {source}. Skipping.")
        except Exception as e:
            print(f"  Error loading source {source_type} from '{source_path or source_urls}': {e}")

    print(f"Finished loading documents. Total documents loaded: {len(all_documents)}")
    return all_documents

