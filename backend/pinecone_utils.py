from pinecone import Pinecone, ServerlessSpec
from backend.config import app_settings, validate_environment
from typing import List, Dict, Any

def initialize_pinecone() -> Pinecone:
    validate_environment()
    return Pinecone(api_key=app_settings.pinecone.api_key)

def create_index(pinecone: Pinecone) -> Any:
    index_name = app_settings.pinecone.index_name
    dimension = app_settings.pinecone.dimension
    spec = ServerlessSpec(
        cloud = app_settings.pinecone.cloud,
        region = app_settings.pinecone.region,
    )
    if not pinecone.has_index(index_name):
        pinecone.create_index(
            name = index_name,
            dimension = dimension,
            spec = spec,
        )
        print(f"Index {index_name} created successfully.")
    else:
        print(f"Index {index_name} already exists.")
    return pinecone.Index(index_name)

def delete_index(pinecone: Pinecone) -> bool:
    index_name = app_settings.pinecone.index_name
    if pinecone.has_index(index_name):
        pinecone.delete_index(index_name)
        print(f"Index {index_name} deleted successfully.")
        return True
    else:
        print(f"Index {index_name} does not exist.")
        return False