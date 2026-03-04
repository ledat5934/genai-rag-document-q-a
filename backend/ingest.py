from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from backend.config import app_settings, validate_environment
from backend.pinecone_utils import initialize_pinecone, create_index
from backend.embeddings import initialize_embeddings


def load_documents(file_path: str) -> List[Document]:
    validate_environment()
    loader = PyPDFLoader(file_path)
    return loader.load()


def ingest_documents(documents: List[Document]) -> None:
    validate_environment()

    pc = initialize_pinecone()
    _ = create_index(pc)

    embeddings = initialize_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=app_settings.pinecone.index_name,
        namespace=app_settings.pinecone.chunks_workspace,
    )

    print(f"Ingested {len(chunks)} chunks into Pinecone.")