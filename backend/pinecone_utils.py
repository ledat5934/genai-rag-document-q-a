from typing import Any

from pinecone import Pinecone, ServerlessSpec

from backend.config import app_settings, validate_environment

def initialize_pinecone() -> Pinecone:
    validate_environment()
    return Pinecone(api_key=app_settings.pinecone.api_key)

def create_index(pc: Pinecone) -> Any:
    index_name = app_settings.pinecone.index_name
    dimension = app_settings.pinecone.dimension
    spec = ServerlessSpec(
        cloud = app_settings.pinecone.cloud,
        region = app_settings.pinecone.region,
    )
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name = index_name,
            dimension = dimension,
            metric="cosine",
            spec = spec,
        )
        print(f"Index {index_name} created successfully.")
    else:
        print(f"Index {index_name} already exists.")
    return pc.Index(index_name)

def delete_index(pc: Pinecone) -> bool:
    index_name = app_settings.pinecone.index_name
    existing = [idx["name"] for idx in pc.list_indexes()]
    if index_name in existing:
        pc.delete_index(index_name)
        print(f"Index {index_name} deleted successfully.")
        return True
    else:
        print(f"Index {index_name} does not exist.")
        return False