"""Vector store management for the agent."""

import logging
from pathlib import Path
from typing import Optional

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from src.utils import logger

# Cache for embedding dimension to avoid repeated API calls
_EMBEDDING_DIM_CACHE: Optional[int] = None


def get_embedding_dimension(embedding_function: OpenAIEmbeddings) -> int:
    """Get the embedding dimension, using cache to avoid repeated API calls.

    Args:
        embedding_function: The OpenAI embeddings instance

    Returns:
        Dimension of the embeddings
    """
    global _EMBEDDING_DIM_CACHE

    if _EMBEDDING_DIM_CACHE is None:
        # Only call the API once to get the dimension
        sample_embedding = embedding_function.embed_query("dimension check")
        _EMBEDDING_DIM_CACHE = len(sample_embedding)
        logger.info(f"Cached embedding dimension: {_EMBEDDING_DIM_CACHE}")

    return _EMBEDDING_DIM_CACHE


def create_faiss_vectorstore(
    load_data: bool = False,
    cache_path: Optional[str] = None,
) -> FAISS:
    """Create a FAISS vector store with OpenAI embeddings.

    Optimizations:
    - Cache embedding dimension to avoid redundant API calls
    - Support loading from saved cache
    - Lazy loading of data ingestion utilities

    Args:
        load_data: If True, load data from data/files.txt into the vectorstore
        cache_path: Optional path to load/save cached vectorstore

    Returns:
        FAISS vectorstore instance
    """
    # Check if we can load from cache
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading vectorstore from cache: {cache_path}")
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            cache_path, embedding, allow_dangerous_deserialization=True
        )
        logger.info("Vectorstore loaded from cache successfully!")
        return vectorstore

    # Create new vectorstore
    embedding = OpenAIEmbeddings()
    embedding_dim = get_embedding_dimension(embedding)

    # Create FAISS index with the correct dimension
    index = faiss.IndexFlatL2(embedding_dim)
    vectorstore = FAISS(
        embedding_function=embedding,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    logger.info("Created new empty vectorstore")

    # Optionally load data from files
    if load_data:
        from src.utils import ingest_data_files

        logger.info("Loading data into vectorstore...")
        vectorstore = ingest_data_files(vectorstore)

        # Save to cache if path provided
        if cache_path:
            logger.info(f"Saving vectorstore to cache: {cache_path}")
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            vectorstore.save_local(cache_path)
            logger.info("Vectorstore saved to cache successfully!")

    return vectorstore
