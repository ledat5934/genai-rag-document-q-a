import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str = os.getenv("PINECONE_API_KEY", "")
    index_name: str = os.getenv("PINECONE_INDEX_NAME", "document-qa-chatbot")
    namespace_chunks: str = os.getenv("PINECONE_NAMESPACE_CHUNKS", "chunks")
    namespace_docs: str = os.getenv("PINECONE_NAMESPACE_DOCS", "docs")
    cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    region: str = os.getenv("PINECONE_REGION", "us-east-1")
    dimension: int = int(os.getenv("PINECONE_DIMENSION", "3072"))
    similarity_threshold: float = float(os.getenv("DOC_SIMILARITY_THRESHOLD", "0.9"))


@dataclass(frozen=True)
class LLMSettings:
    api: str = os.getenv("GOOGLE_API_KEY", "")
    model: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

@dataclass(frozen=True)
class AppSettings:
    pinecone: PineconeSettings = PineconeSettings()
    llm: LLMSettings = LLMSettings()

def validate_settings() -> None:
    pinecone_settings = PineconeSettings()
    llm_settings = LLMSettings()
    if not pinecone_settings.api_key:
        raise ValueError("PINECONE_API_KEY is not set in environment.")
    if not llm_settings.api:
        raise ValueError("GOOGLE_API_KEY is not set in environment.")

