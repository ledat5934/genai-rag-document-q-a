import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()
@dataclass(frozen = True)
class PineconeConfig:
    api_key: str = os.getenv("PINECONE_API_KEY", "")
    index_name: str = os.getenv("PINECONE_INDEX_NAME", "document-qa-chatbot")
    dimension: int = int(os.getenv("PINECONE_DIMENSION", "3072"))
    chunks_workspace: str = os.getenv("PINECONE_CHUNKS_WORKSPACE", "chunks")
    docs_workspace: str = os.getenv("PINECONE_DOCS_WORKSPACE", "docs")
    region: str = os.getenv("PINECONE_REGION", "us-east-1")
    cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    threshold: float = float(os.getenv("PINECONE_THRESHOLD", "0.5"))

@dataclass(frozen = True)
class GoogleGenaiConfig:
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    model: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    embed_model: str = os.getenv("GOOGLE_EMBED_MODEL", "models/gemini-embedding-001")

@dataclass(frozen = True)
class AppSettings:
    pinecone: PineconeConfig = PineconeConfig()
    google_genai: GoogleGenaiConfig = GoogleGenaiConfig()

app_settings = AppSettings()

def validate_environment() -> None:
    if not app_settings.pinecone.api_key:
        raise ValueError("PINECONE_API_KEY is not set")
    if not app_settings.google_genai.api_key:
        raise ValueError("GOOGLE_API_KEY is not set")
