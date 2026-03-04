from backend.config import app_settings, validate_environment
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def initialize_embeddings() -> GoogleGenerativeAIEmbeddings:
    validate_environment()
    return GoogleGenerativeAIEmbeddings(
        api_key=app_settings.google_genai.api_key,
        model=app_settings.google_genai.embed_model,
    )

