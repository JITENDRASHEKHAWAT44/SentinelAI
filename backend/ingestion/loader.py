from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.database.vector_store import add_documents

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# 1. Load document
def load_document(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyMuPDFLoader(str(path))
    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))
    elif extension == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError("Unsupported file format")

    documents = loader.load()

    for document in documents:
        metadata = dict(document.metadata or {})
        metadata.setdefault("source", str(path))
        metadata["filename"] = path.name
        document.metadata = metadata

    print(f"Loaded {len(documents)} document pages/files")
    return documents


# 2. Split document into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    raw_chunks = text_splitter.split_documents(documents)
    # Filter out empty or whitespace-only chunks to protect the embedding model
    chunks = [c for c in raw_chunks if c.page_content and c.page_content.strip()]

    print(f"Created {len(chunks)} non-empty chunks (from {len(raw_chunks)} raw)")
    return chunks


# 3. Store chunks in vector database
def store_documents(chunks):
    ids = add_documents(chunks)
    print(f"Stored {len(ids)} chunks in PostgreSQL")
    return ids


# 4. Complete ingestion pipeline
def ingest_document(file_path: str):
    print("\nStarting document ingestion...")

    # Load
    documents = load_document(file_path)

    # Split
    chunks = split_documents(documents)

    # Store
    ids = store_documents(chunks)

    print("\nDocument ingestion completed successfully!")
    return ids
